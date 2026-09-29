# Team RAMEN Submission

Candidate branch: `issue/16-recoverable-joint-preparation` of
https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit

Not merged into `main`. Runtime changes are proposed in
[RAMEN PR #182, targeting develop](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/pull/182).

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

1. Pull the digest above on Thor with an authorized GHCR account.
2. Follow [WEIGHTS.md](WEIGHTS.md) to prefetch the configured checkpoints and dependencies
   into the host HF cache (approximately 80 GB). Image download alone does not include weights.
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

[GB10_PREPARATION_REPORT.md](GB10_PREPARATION_REPORT.md) records this image's identity,
verification status, retained failed attempts and limitations. The earlier
[rebuild5 report](GB10_REBUILD5_REPORT.md) is historical evidence, not proof for this image.

These are software and non-actuating tests, not a guarantee of physical task success on Thor/G1.
Client/transport-loss navigation retention remains tracked in
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).
The owner explicitly excluded it from this merge on 2026-09-27; it is not marked fixed.
Follow the documented on-site E-stop procedure rather than treating disconnection as a stop.
