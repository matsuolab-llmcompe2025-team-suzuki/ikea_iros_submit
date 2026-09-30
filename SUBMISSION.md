# Team RAMEN Submission

Branch: `main` of https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit

The image contains RAMEN source `1b34e827c88bec1b23a20225428b9200583cac9a`
from [PR #189](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/pull/189).
Later commits on that PR change documentation only; the image retains the pinned source.

## Image

- Platform: `linux/arm64`; one container on Thor, no team container on Orin/PC2.
- Tag: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:gb10-test-1b34e82`
- Use this immutable reference for pull and execution:

```text
ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:6ee78c2f84fe1301d74208b887bb3ab1b29eff7d30a0449dee9909435085ede2
```

- RAMEN source: `1b34e827c88bec1b23a20225428b9200583cac9a`.
- Image build input: `abead33ca61cf6a2e735bfb963bd7088607feea0`.
- Tested organizer interface: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Private GHCR read access is required. Do not put credentials in this repository.
- [Arm64 build and push](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36726824877) succeeded. This exact image has no completed GB10 Stage matrix or physical Thor/G1 verification.
- Merge-time documentation updates do not change the built image or its build inputs.

## Preparation And Operation

1. Pull the digest above on Thor with an authorized GHCR account, and make the venue
   `skill_config` from it (INSTRUCTIONS.md section 1; sha256 checked). It is the image's own
   config with only the Dex1 arrival tolerance widened (0.05 -> 0.20 rad); every run mounts it.
2. Follow [WEIGHTS.md](WEIGHTS.md) to prefetch the configured checkpoints and dependencies
   into the host HF cache (approximately 80 GB). Image download alone does not include weights.
   The HF account needs Team-RAMEN private-repository access and Cosmos model access approval.
3. Run the documented offline `prefetch_weights.py --check` inside the image with the cache mounted.
4. Use [INSTRUCTIONS.md](INSTRUCTIONS.md) for the ordered PC2/Thor startup, the pre-run checks
   (other teams' processes, image, venue config, Stage 0 walking distance) and operator controls.
   On the event PC2 (G1 (3)) the organizer processes run after `source ~/iros_g1_3/iros_env.sh`,
   and the WBC is `~/wbc_adapter/deploy/run_wbc_with_dex1.py` (per-robot Dex1 calibration).
   [CONNECTION_TEST.md](CONNECTION_TEST.md) is the 09-27 connection-test record.

An optional read-only PC2 state guard (`:5558`, RAMEN issue #184) is declared in the manifest as
`pc2_read_only_guard` and is off by default. The image includes the Thor reader, but using it
requires deployment of the separate PC2 bundle with the organizer's approval (INSTRUCTIONS.md section 7).

The default action contract is the joint lane `(T,22)`; the organizer adapter is launched
with `--lane decoupled`. Do not switch to the pose lane: on 2026-09-29 the organizer IK on this
robot accepted 0% of 602 waypoints for another team, while the joint lane rejected none. Ports and launch commands are declared in [manifest.yaml](manifest.yaml).
Organizer code is not modified. No head-camera geometry compensation is applied by this submission.
Runtime reads the preloaded weights offline; the camera/state/action network remains necessary.

## Verification And Limits

The venue arm tracking update includes a 2.1x elbow gravity offset, a bounded
joint-lane tracking assist and new default rotate/pick/insert checkpoints. The
tracking estimates in the source handoff are offline simulations and evaluations.
The image build succeeded, but this digest has not completed the GB10 Stage
matrix, model forward checks, operator-key sequence or a Thor/G1 run. Check
the prefetched weights and venue skill-config hash before use.

[GB10_STATE_GUARD_REPORT.md](GB10_STATE_GUARD_REPORT.md), the
[preparation report](GB10_PREPARATION_REPORT.md) and the
[rebuild5 report](GB10_REBUILD5_REPORT.md) describe older images.

These are software and non-actuating tests, not a guarantee of physical task success on Thor/G1.
Client/transport-loss navigation retention remains tracked in
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).
The owner explicitly excluded it from this merge on 2026-09-27; it is not marked fixed.
Follow the documented on-site E-stop procedure rather than treating disconnection as a stop.
