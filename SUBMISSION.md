# Team RAMEN Submission

Submission branch: `main` of
https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit

## Image

- Platform: `linux/arm64`; one container on Thor, no team container on Orin/PC2.
- Tag: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:20260927-insert-dp100k-01`
- Use this immutable reference for pull and execution:

```text
ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:c98c80c70dccafd418910fe84a2e1ab1685b46d6106a5c77b2a54ce30ff77923
```

- Image config digest: `sha256:859a31949bf2eb290fff51e4b6e8ac1bbcd00d879b61422dce18da2e38de7b59` (arm64 CI build output).
- RAMEN source: `c73c922e01c6f19a6b9dccb483a08ed83ffdd5a7`.
- Organizer interface tested with the previous rebuild5 image: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Private GHCR read access is required. Do not put credentials in this repository.
- [Arm64 build and push](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36318545130) succeeded; this new image has not yet had a GB10 or Thor/G1 validation run.

## Preparation And Operation

1. Pull the digest above on Thor with an authorized GHCR account.
2. Follow [WEIGHTS.md](WEIGHTS.md) to prefetch the configured checkpoints and dependencies
   into the host HF cache (approximately 78 GB, planning estimate). Image download alone does not include weights.
   The HF account needs Team-RAMEN private-repository access and Cosmos model access approval.
3. Run the documented offline `prefetch_weights.py --check` inside the image with the cache mounted.
4. Use [INSTRUCTIONS.md](INSTRUCTIONS.md) for the ordered PC2/Thor startup and operator controls.
   Confirm venue addresses, camera streams, clocks and WBC settings with
   [CONNECTION_TEST.md](CONNECTION_TEST.md) before actuation.

The default action contract is the joint lane `(T,22)`; the organizer adapter is launched
with `--lane decoupled`. Ports and launch commands are declared in [manifest.yaml](manifest.yaml).
Organizer code is not modified. No head-camera geometry compensation is applied by this submission.
Runtime reads the preloaded weights offline; the camera/state/action network remains necessary.

## Verification And Limits

[GB10_REBUILD5_REPORT.md](GB10_REBUILD5_REPORT.md) records the **previous** image's evidence and limitations:
four GPU environments, all Stage preflights, 11 model configurations with 990 forward calls,
continuous Stages 0-5, a 600-second flip soak, fault injection, no observed external connects,
84 organizer tests, and recorded action replay into the official fake backend.
The new insert-Diffusion image passed the arm64 CI build and 69 local lightweight regressions.
The 298 GB10 runtime regressions above were run on rebuild5, not this image.

These are software and non-actuating tests, not a guarantee of physical task success on Thor/G1.
Client/transport-loss navigation retention remains tracked in
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).
The owner explicitly excluded it from this merge on 2026-09-27; it is not marked fixed.
Follow the documented on-site E-stop procedure rather than treating disconnection as a stop.
