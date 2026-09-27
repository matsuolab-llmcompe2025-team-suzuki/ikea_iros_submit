# Team RAMEN Submission

Submission branch: `main` of
https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit

## Image

- Platform: `linux/arm64`; one container on Thor, no team container on Orin/PC2.
- Tag: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:20260927-rebuild5`
- Use this immutable reference for pull and execution:

```text
ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:b58c2cd594955a092a7481a21f866577127c40f178748c64b9216d250a24939f
```

- Image ID: `sha256:78a13b49593979d7f58e763174808a4c0ccca95269830125fdcb0fc7364b801e`
- RAMEN source: `e3a41877ea76bfeb106de7afa991c017ad57ce76`.
- Tested organizer interface: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Private GHCR read access is required. Do not put credentials in this repository.
- Merge-time documentation updates do not change the tested image or its build inputs.

## Preparation And Operation

1. Pull the digest above on Thor with an authorized GHCR account.
2. Follow [WEIGHTS.md](WEIGHTS.md) to prefetch the configured checkpoints and dependencies
   into the host HF cache (approximately 90 GB). Image download alone does not include weights.
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

[GB10_REBUILD5_REPORT.md](GB10_REBUILD5_REPORT.md) records the tested image, evidence and limitations:
four GPU environments, all Stage preflights, 11 model configurations with 990 forward calls,
continuous Stages 0-5, a 600-second flip soak, fault injection, no observed external connects,
84 organizer tests, and recorded action replay into the official fake backend.
Local lightweight regressions: 69 passed; GB10 runtime regressions: 298 passed.

These are software and non-actuating tests, not a guarantee of physical task success on Thor/G1.
Client/transport-loss navigation retention remains tracked in
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).
The owner explicitly excluded it from this merge on 2026-09-27; it is not marked fixed.
Follow the documented on-site E-stop procedure rather than treating disconnection as a stop.
