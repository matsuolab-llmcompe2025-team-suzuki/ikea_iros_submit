# Team RAMEN Submission

Branch: `main` of https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit

The runtime is RAMEN `develop` as of
[PR #182](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/pull/182)
(source `10b8d73` in the image; `develop` `faf73a8` has the same inference code).

## Image

- Platform: `linux/arm64`; one container on Thor, no team container on Orin/PC2.
- Tag: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:gb10-preparation-10b8d73`
- Use this immutable reference for pull and execution:

```text
ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:6db3fb0b23dcf9a1835fc5a8c82b2d50dd2046090b745a6132f09c03f84757f4
```

- Image config ID: `sha256:f2eafc93eec07929e1b39f4b9fb48f3b15e11d606dbad4a005509df0a94b728b`
- RAMEN source: `10b8d736609595bde30c929ddb73ef2f4751c0f1`.
- Image build input: `c56fc693fa6231529a9b38c7fa0f3e6f1c766596`.
- Tested organizer interface: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Private GHCR read access is required. Do not put credentials in this repository.
- [Arm64 build and push](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36479757032) succeeded. GB10 results and untested conditions are recorded below; physical Thor/G1 motion is not verified.
- Merge-time documentation updates do not change the tested image or its build inputs.

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
5. Start the Thor container once and run the remaining stages in it (`--phase3-full`): the
   36-minute evaluation cycle leaves no time to reload models (4-5 min with Stage 1-4) per run.
   [CONNECTION_TEST.md](CONNECTION_TEST.md) is the 09-27 connection-test record.

The default action contract is the joint lane `(T,22)`; the organizer adapter is launched
with `--lane decoupled`. Do not switch to the pose lane: on 2026-09-29 the organizer IK on this
robot accepted 0% of 602 waypoints for another team, while the joint lane rejected none. Ports and launch commands are declared in [manifest.yaml](manifest.yaml).
Organizer code is not modified. No head-camera geometry compensation is applied by this submission.
Runtime reads the preloaded weights offline; the camera/state/action network remains necessary.

## Verification And Limits

[GB10_PREPARATION_REPORT.md](GB10_PREPARATION_REPORT.md) records this image's identity,
verification status, retained failed attempts and limitations. The earlier
[rebuild5 report](GB10_REBUILD5_REPORT.md) is historical evidence, not proof for this image.

These are software and non-actuating tests, not a guarantee of physical task success on Thor/G1.
Client/transport-loss navigation retention remains tracked in
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).
The owner explicitly excluded it from this merge on 2026-09-27; it is not marked fixed.
Follow the documented on-site E-stop procedure rather than treating disconnection as a stop.
