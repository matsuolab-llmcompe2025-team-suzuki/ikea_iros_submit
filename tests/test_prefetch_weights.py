"""重みの事前取得 (tools/prefetch_weights.py) の test。ネットには出ない。

--check は ramen/ の本物の解決関数を HF_HUB_OFFLINE=1 で呼ぶ。HF が取ったときと同じ配置の
偽の cache で OK になれば、事前取得と実行時の呼び方が cache の上で一致している。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

SUBMIT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SUBMIT_ROOT / "tools" / "prefetch_weights.py"
RAMEN = SUBMIT_ROOT / "ramen"
FAKE_MAIN_COMMIT = "a" * 40
POLICY_CONFIG = RAMEN / "inference/desktop/lower_policy/configs/policy_config.yaml"
#: GB10 で DP を試すときに --variant で足す slot (会場の候補の set には入っていない)
DP_TRIAL_VARIANT = "rotate_table_base_diffusion"


def _module():
    spec = importlib.util.spec_from_file_location("prefetch_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass が自分の module を引けるように
    spec.loader.exec_module(module)
    return module


def _run_check(hf_home: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "HF_HOME": str(hf_home),
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("HF_TOKEN", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--check", *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_list_follows_the_config() -> None:
    """既定の model・YOLO・VLM と外部の backbone が、設定どおりの revision で並ぶ。"""
    module = _module()
    items = {item.repo_id: item for item in module.build_items()}
    config = yaml.safe_load(
        (
            RAMEN / "inference/desktop/lower_policy/configs/policy_config.yaml"
        ).read_text()
    )
    for variant in config["default_variant_by_skill"].values():
        repo_id, _, revision = config["policies"][variant]["ckpt_ref"].rpartition("@")
        assert items[repo_id].revision == revision, variant
    yolo_repo, _, yolo_revision = config["yolo"]["ckpt_ref"].rpartition("@")
    assert items[yolo_repo].revision == yolo_revision
    assert items[yolo_repo].scope == config["yolo"]["ckpt_file"]
    for repo_id in (
        module.LINGBOT_REPO,
        module.COSMOS_REPO,
        "Qwen/Qwen3-VL-8B-Instruct",
    ):
        assert items[repo_id].revision == "main"


def _selected_variants() -> set[str]:
    """既定と variant_sets の全部 (会場で起動の引数だけで切り替えられる model)。"""
    config = yaml.safe_load(POLICY_CONFIG.read_text())
    variants = set(config["default_variant_by_skill"].values())
    for variant_set in config.get("variant_sets", {}).values():
        variants |= set(variant_set.values())
    return variants


def test_every_variant_set_is_prefetched_once_per_file() -> None:
    """会場はネット無しなので、set で切り替える先の重みも取る。同じ file は 1 回だけ。"""
    module = _module()
    items = module.build_items()
    config = yaml.safe_load(POLICY_CONFIG.read_text())
    for variant in _selected_variants():
        spec = config["policies"][variant]
        repo_id, _, revision = spec["ckpt_ref"].rpartition("@")
        matches = [
            item
            for item in items
            if (item.repo_id, item.revision) == (repo_id, revision)
            and (spec.get("ckpt_filename") in (None, item.scope))
        ]
        assert len(matches) == 1, variant
        assert variant in matches[0].label, variant
    keys = [(item.repo_id, item.revision, item.scope) for item in items]
    assert len(keys) == len(set(keys))


def test_a_variant_outside_the_sets_is_added_only_on_request() -> None:
    module = _module()
    assert not any(DP_TRIAL_VARIANT in item.label for item in module.build_items())
    (dp,) = [
        item
        for item in module.build_items([DP_TRIAL_VARIANT])
        if DP_TRIAL_VARIANT in item.label
    ]
    assert dp.repo_id == "Team-RAMEN/IROS2026_RAMEN_hara_rotate_table_base_diffusion_v1"
    try:
        module.build_items(["no_such_slot"])
    except ValueError as exc:
        assert "no_such_slot" in str(exc)
    else:
        raise AssertionError("unknown --variant must be rejected")


def test_the_groot_base_default_matches_the_policy() -> None:
    source = (RAMEN / "inference/desktop/lower_policy/policies/groot.py").read_text()
    assert _module().GROOT_BASE_DEFAULT_REVISION in source


def test_weights_md_and_the_manifest_list_every_item() -> None:
    module = _module()
    weights_md = (SUBMIT_ROOT / "WEIGHTS.md").read_text()
    manifest = (SUBMIT_ROOT / "manifest.yaml").read_text()
    expected = [(item.repo_id, item.revision) for item in module.build_items()]
    expected.append((module.GROOT_BASE_REPO, module.GROOT_BASE_DEFAULT_REVISION))
    for repo_id, revision in expected:
        assert repo_id in weights_md and revision[:7] in weights_md, repo_id
        assert repo_id in manifest and revision[:7] in manifest, repo_id


def test_check_reports_every_weight_missing_on_an_empty_cache(tmp_path) -> None:
    result = _run_check(tmp_path)
    assert result.returncode == 1, result.stderr[-2000:]
    expected = len(_module().build_items()) + 1  # + GR00T の base model
    assert result.stdout.count("MISSING") == expected
    assert f"{expected} missing" in result.stdout


def _lay_out(cache: Path, repo_id: str, revision: str, files: dict[str, str]) -> None:
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    commit = FAKE_MAIN_COMMIT if revision == "main" else revision
    if revision == "main":
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(commit)
    for name, content in files.items():
        path = repo_dir / "snapshots" / commit / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_check_passes_on_a_cache_laid_out_like_a_download(tmp_path) -> None:
    """HF が取ったときと同じ配置なら、本物の解決関数がネット無しで全部見つける。

    既定・set の全部・--variant の DP を、実行時と同じ関数で cache から解決する。
    """
    sys.path.insert(0, str(RAMEN))
    from inference.desktop.lower_policy.policies.config_loader import (
        load_policy_variant,
    )

    module = _module()
    config_path = POLICY_CONFIG
    config = yaml.safe_load(config_path.read_text())
    cache = tmp_path / "hub"
    base_config = json.dumps(
        {"base_model_revision": module.GROOT_BASE_DEFAULT_REVISION}
    )
    for variant in sorted(_selected_variants() | {DP_TRIAL_VARIANT}):
        entry = load_policy_variant(config_path, variant)
        cfg = entry.policy_config
        repo_id, _, revision = str(cfg.ckpt_ref).rpartition("@")
        prefix = f"{cfg.checkpoint_subdir}/" if cfg.checkpoint_subdir else ""
        if entry.policy_type == "ramen_ori":
            files = {cfg.ckpt_filename: "ckpt"}
        elif entry.policy_type == "groot":
            files = {
                f"{prefix}config.json": base_config,
                f"{prefix}model.safetensors": "w",
            }
        elif entry.policy_type == "act_diffusion":  # LeRobot の保存形式 (最終 model は root)
            files = {f"{prefix}config.json": "{}", f"{prefix}model.safetensors": "w"}
        else:  # groot_pick_legs: 実行時に要る file
            files = {
                f"{prefix}{name}": "{}"
                for name in (
                    "config.json",
                    "processor_config.json",
                    "statistics.json",
                    "embodiment_id.json",
                    "model.safetensors.index.json",
                )
            }
        _lay_out(cache, repo_id, revision, files)
    yolo_repo, _, yolo_revision = config["yolo"]["ckpt_ref"].rpartition("@")
    _lay_out(cache, yolo_repo, yolo_revision, {config["yolo"]["ckpt_file"]: "pt"})
    for repo_id in (
        module.LINGBOT_REPO,
        module.COSMOS_REPO,
        "Qwen/Qwen3-VL-8B-Instruct",
    ):
        _lay_out(cache, repo_id, "main", {"config.json": "{}"})
    _lay_out(
        cache,
        module.GROOT_BASE_REPO,
        module.GROOT_BASE_DEFAULT_REVISION,
        {"config.json": "{}"},
    )

    result = _run_check(tmp_path, "--variant", DP_TRIAL_VARIANT)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert "MISSING" not in result.stdout
    assert "all present" in result.stdout
    assert DP_TRIAL_VARIANT in result.stdout
