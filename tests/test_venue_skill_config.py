"""会場で使う skill_config (venue/skill_config_venue.yaml) の test。

image の skill_config と違うのは Dex1 の到達の許容 (`hand_pre_motion.tolerance_rad`) だけ。
会場の Dex1 は運営の柔らかい P 制御 (kp=5.0) で、全開は手で測った機械の端なので、0.05 rad では
手前で止まって時間切れになり、run が policy の前に終わりうる (INSTRUCTIONS.md 4 章)。
Thor では image から同じ sed で作り、sha256 で同じ物かを確かめる (INSTRUCTIONS.md 1 章)。
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

SUBMIT_ROOT = Path(__file__).resolve().parents[1]
RAMEN_CONFIG = SUBMIT_ROOT / "ramen/inference/desktop/lower_policy/configs/skill_config.yaml"
VENUE_CONFIG = SUBMIT_ROOT / "venue/skill_config_venue.yaml"
IMAGE_CONFIG_PATH = "/app/ramen/inference/desktop/lower_policy/configs/skill_config.yaml"
MOUNT = "/app/venue_skill_config.yaml"
# INSTRUCTIONS.md 1 章の sed と同じ式 (手順書と一字一句同じであることも下で確かめる)
SED_EXPRESSION = (
    r"s/^  tolerance_rad: 0\.05 .*$/"
    r"  tolerance_rad: 0.20  # venue: Dex1 air arrival, ~3.3 mm (INSTRUCTIONS.md sec. 4)/"
)
VENUE_TOLERANCE_RAD = 0.20


def _instructions() -> str:
    return (SUBMIT_ROOT / "INSTRUCTIONS.md").read_text()


def test_the_venue_config_is_the_image_config_with_only_the_dex1_tolerance_widened() -> None:
    image = yaml.safe_load(RAMEN_CONFIG.read_text())
    venue = yaml.safe_load(VENUE_CONFIG.read_text())
    assert image["hand_pre_motion"]["tolerance_rad"] == 0.05
    expected = dict(image)
    expected["hand_pre_motion"] = {**image["hand_pre_motion"], "tolerance_rad": VENUE_TOLERANCE_RAD}
    assert venue == expected


def test_the_thor_sed_reproduces_the_committed_file_byte_for_byte() -> None:
    made = subprocess.run(
        ["sed", SED_EXPRESSION, str(RAMEN_CONFIG)],
        check=True, capture_output=True,
    ).stdout
    assert made == VENUE_CONFIG.read_bytes()


def test_the_procedure_names_the_same_sed_hash_and_image_path() -> None:
    text = _instructions()
    assert SED_EXPRESSION in text
    assert IMAGE_CONFIG_PATH in text
    digest = hashlib.sha256(VENUE_CONFIG.read_bytes()).hexdigest()
    assert digest in text, "INSTRUCTIONS.md の sha256 を venue config に合わせる"
    # 手順書に残った古い sha256 が無いこと (64 桁の 16 進はこの 1 つだけ)
    hashes = set(re.findall(r"\b[0-9a-f]{64}\b", text))
    image_digests = set(re.findall(r"sha256:([0-9a-f]{64})", text))
    assert hashes - image_digests == {digest}


def test_every_venue_run_mounts_the_venue_config() -> None:
    manifest = yaml.safe_load((SUBMIT_ROOT / "manifest.yaml").read_text())
    run = " ".join(manifest["images"]["thor"]["run"].split())
    assert f"-v <skill_config_venue.yaml>:{MOUNT}:ro" in run
    assert run.endswith(f"--actuate --skill-config {MOUNT}")
    step4 = _instructions().split("### Step 4", 1)[1].split("### Step 5", 1)[0]
    assert f"skill_config_venue.yaml:{MOUNT}:ro" in step4
    assert f"--skill-config {MOUNT}" in step4


@pytest.fixture()
def ramen_code(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(SUBMIT_ROOT / "ramen"))
    yield
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]


class _Dex1:
    """運営 relay の 4.2 rad/s で動き、全開の手前 `short` rad で止まる Dex1。"""

    def __init__(self, short: float) -> None:
        import numpy as np

        self.q = np.array([1.0, 1.0])
        self.command = None
        self.short = short

    def send_action(self, command) -> None:
        import numpy as np

        self.command = np.asarray(command, dtype=np.float64)

    def read_last_commanded_positions(self):
        return None if self.command is None else tuple(self.command)

    def advance(self, dt: float) -> None:
        import numpy as np

        if self.command is not None:
            goal = self.command - self.short
            self.q += np.clip(goal - self.q, -4.2 * dt, 4.2 * dt)


def _open_hands(config: dict, short: float) -> str:
    import numpy as np

    from inference.desktop.lower_policy.skills.hand_pre_motion import HandPreMotionSkill
    from inference.desktop.lower_policy.skills.hand_ramp import resolve_hand_opening_rad
    from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES

    # entrypoint と同じく、boundary では全開を運営上限へ解決してから作る
    config = {**config, "hand_pre_motion": {
        **config["hand_pre_motion"],
        "open_rad": resolve_hand_opening_rad(config, action_sink="boundary"),
    }}
    clock = types.SimpleNamespace(t=0.0)
    hand = _Dex1(short)
    skill = HandPreMotionSkill.from_config(
        config, "pick_table_leg", target="open", hand_actuator=hand, time_fn=lambda: clock.t
    )
    skill.start({})
    joints = types.SimpleNamespace(
        name=list(G1_JOINT_NAMES), position=np.zeros(len(G1_JOINT_NAMES))
    )
    for _ in range(30 * 20):
        now = time.monotonic_ns()
        state = types.SimpleNamespace(
            position_rad=hand.q.copy(),
            left_received_monotonic_ns=now, right_received_monotonic_ns=now,
        )
        skill.step({"hand_state": state, "joint_state": joints})
        hand.advance(1 / 30)
        clock.t += 1 / 30
        if skill.is_complete:
            return "complete"
        if skill.timeout_reason or skill.failure_reason:
            return "timeout"
    return "pending"


@pytest.mark.parametrize("short", [0.08, 0.12])
def test_a_dex1_stopping_short_ends_the_run_only_with_the_image_tolerance(ramen_code, short):
    image = yaml.safe_load(RAMEN_CONFIG.read_text())
    venue = yaml.safe_load(VENUE_CONFIG.read_text())
    assert _open_hands(image, short) == "timeout"
    assert _open_hands(venue, short) == "complete"


def test_the_venue_config_still_completes_a_dex1_that_reaches_the_target(ramen_code):
    venue = yaml.safe_load(VENUE_CONFIG.read_text())
    assert _open_hands(venue, 0.0) == "complete"
