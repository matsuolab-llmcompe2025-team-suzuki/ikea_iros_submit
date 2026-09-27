"""manifest.yaml (運営に出す宣言) と INSTRUCTIONS.md (会場の手順) の image が食い違わないかの test。

会場で打つ docker pull / docker run の digest が manifest と違うと、運営が確かめた image と
別のものを動かす。image を差し替えたら両方を同じ digest に直す。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

SUBMIT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor"


def _thor() -> dict:
    manifest = yaml.safe_load((SUBMIT_ROOT / "manifest.yaml").read_text())
    assert manifest["lane"] == "joint"
    assert "--lane decoupled" in manifest["organizer_stack_on_pc2"]["adapter_go_live"]
    return manifest["images"]["thor"]


def test_the_manifest_names_a_released_image() -> None:
    thor = _thor()
    repo, _, tag = thor["repo_tag"].rpartition(":")
    assert repo == REGISTRY
    assert tag and tag != "TBD"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", thor["digest"]), thor["digest"]
    manifest = yaml.safe_load((SUBMIT_ROOT / "manifest.yaml").read_text())
    assert isinstance(manifest["runtime"]["peak_memory_thor_gb"], (int, float))


def test_the_venue_procedure_pulls_and_runs_the_manifest_digest() -> None:
    instructions = (SUBMIT_ROOT / "INSTRUCTIONS.md").read_text()
    assert "<DIGEST>" not in instructions
    assert "__REBUILD" not in instructions
    pinned = re.findall(rf"{re.escape(REGISTRY)}@(sha256:[0-9a-f]+)", instructions)
    assert len(pinned) >= 2, pinned  # docker pull と docker run
    assert set(pinned) == {_thor()["digest"]}
