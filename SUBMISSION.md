# Team RAMEN Submission

Branch: `main` of https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit

The runtime is RAMEN `develop` as of
[PR #185](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/pull/185)
(source `1cd5fdf` in the image, including the opt-in guard's separate measurement clock).

## Image

- Platform: `linux/arm64`; one container on Thor, no team container on Orin/PC2.
- Tag: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:gb10-guard-1cd5fdf`
- Use this immutable reference for pull and execution:

```text
ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:19ebe27030179aa04bc9d6f8c167971699fcc6417cbc3a5f23b2d2ae7a0f8262
```

- Image config ID: `sha256:a2fcf98773e4ca738c083cb36f5dec5024867817e4d7a685d3735166e5cf1e87`.
- RAMEN source: `1cd5fdf4d37b29edbccc5a0f0434f6d379d1a469`.
- Image build input: `55c1a55db45a4117f761a99200d4a1977644dbf9`.
- Tested organizer interface: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Private GHCR read access is required. Do not put credentials in this repository.
- [Arm64 build and push](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36567638842) succeeded. GB10 results and untested conditions are recorded below; physical Thor/G1 motion is not verified.
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

GB10 end-to-end validation was interrupted at the owner's request. The owner then
requested main integration without resuming it. Default-path Stage 0 passed;
Stage 1 was interrupted during model preparation. Remaining Stage/guard-path
end-to-end and forward-matrix checks are pending, not passed.

[GB10_STATE_GUARD_REPORT.md](GB10_STATE_GUARD_REPORT.md) records this image's identity,
verification status, retained failed attempts and limitations. The earlier
[preparation report](GB10_PREPARATION_REPORT.md) and
[rebuild5 report](GB10_REBUILD5_REPORT.md) are historical evidence, not proof for this image.

These are software and non-actuating tests, not a guarantee of physical task success on Thor/G1.
Client/transport-loss navigation retention remains tracked in
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).
The owner explicitly excluded it from this merge on 2026-09-27; it is not marked fixed.
Follow the documented on-site E-stop procedure rather than treating disconnection as a stop.
