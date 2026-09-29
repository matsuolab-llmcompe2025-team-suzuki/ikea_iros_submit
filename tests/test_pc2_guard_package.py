import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def test_small_read_only_bundle_matches_generated_source(tmp_path):
    archive = tmp_path / "guard.tar.gz"
    subprocess.run([sys.executable, str(ROOT / "tools/package_pc2_guard.py"), str(archive)],
                   check=True, timeout=15)
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
        assert names == {"venue_state_guard.py", "configs/pc2_g1_3_internal.json",
                         "SHA256SUMS", "RAMEN_SOURCE.txt"}
        assert all(m.isfile() for m in bundle.getmembers())
        for line in bundle.extractfile("SHA256SUMS").read().decode().splitlines():
            digest, name = line.split("  ")
            assert hashlib.sha256(bundle.extractfile(name).read()).hexdigest() == digest
        source = bundle.extractfile("venue_state_guard.py").read().decode()
        assert "ChannelPublisher" not in source
        assert "ChannelSubscriber" in source
        profile = json.load(bundle.extractfile("configs/pc2_g1_3_internal.json"))
        assert profile["hand_type"] == "internal"
        assert "wbc_adapter/deploy/run_wbc_with_dex1.py" in profile["files_sha256"]
    assert archive.stat().st_size < 100_000


def test_operator_environment_and_wrapper_are_the_delivered_rig():
    instructions = (ROOT / "INSTRUCTIONS.md").read_text()
    assert "conda activate g1_wbc\n" not in instructions
    assert "source ~/iros_g1_3/iros_env.sh" in instructions
    assert "source ~/iros_g1_3/iros_env_teleimager.sh" in instructions
    assert "iros_with_wbc_preload python ~/wbc_adapter/deploy/run_wbc_with_dex1.py" in instructions
    assert "--bind-address <PC2_IP>" in instructions


def test_worktree_sync_cannot_be_built_as_release():
    dockerfile = (ROOT / "docker/Dockerfile.thor").read_text()
    check = "! grep -q '^state: uncommitted-worktree$' RAMEN_SOURCE.txt"
    assert check in dockerfile
    assert dockerfile.index("COPY ramen/ ./") < dockerfile.index(check)


def test_the_guard_stays_optional_and_the_default_path_is_unchanged():
    import yaml
    entry = (ROOT / "docker/venue_entry.sh").read_text()
    assert "--boundary-state-guard" not in entry
    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text())
    assert manifest["pc2_read_only_guard"]["enabled_by_default"] is False
    assert manifest["images"]["thor"]["digest"], "the verified image must stay declared"
    assert "--boundary-state-guard" not in manifest["images"]["thor"]["run"]
    assert "5557" in manifest["sockets"] and "OPTIONAL" in manifest["sockets"]["5558"]
