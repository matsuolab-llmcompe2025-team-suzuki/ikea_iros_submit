# Optional State Guard Validation

Status: interrupted at the user's request on 2026-09-29 21:58 JST.
Do not treat this partial validation as full test or physical release approval.
The owner subsequently requested merging submission PR #21 into main without
resuming tests. That integration decision does not complete the pending checks.

## Identity

- Source: `1cd5fdf4d37b29edbccc5a0f0434f6d379d1a469` (RAMEN develop, PR #185).
- Build input: `55c1a55db45a4117f761a99200d4a1977644dbf9` (submission PR #21).
- Image tag: `gb10-guard-1cd5fdf`.
- Image digest: `sha256:19ebe27030179aa04bc9d6f8c167971699fcc6417cbc3a5f23b2d2ae7a0f8262`.
- Image config: `sha256:a2fcf98773e4ca738c083cb36f5dec5024867817e4d7a685d3735166e5cf1e87`.
- ARM64 build: [36567638842](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36567638842), successful.
- Organizer fixture: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Local evidence directory: `outputs/gb10_validation/20260929_issue20/` in the source repository.

## Scope

The default state stream remains the organizer PUB on :5557. The guard is opt-in,
using REQ/REP :5558 and a separately deployed, organizer-approved read-only PC2
subscriber. No organizer files are modified and no live robot is contacted.

Freshness uses a conservative local-time lower bound. Velocity and preparation
dwell use the guard's measurement clock, not request latency. The new test helper
runs the production attestor and relay with synthetic lowstate on loopback;
source/DDS cessation and bridge cessation are injected independently.

## Completed Local Checks

- Source regression suite: 434 passed, including variable-latency measurement/dwell.
- Focused submission suite: 43 passed, including guard protocol and lifecycle.
- Broad local submission suite: 89 passed, 2 unavailable because this local Python
  lacks huggingface_hub; both were repeated successfully on GB10 below.
- The checked source is normally synced from a published commit. Worktree-only
  snapshots are rejected by Dockerfile and CI.

## Completed GB10 Checks

- Runtime regression suite: 743 passed, 1 integration test initially skipped.
  That real teacher-trajectory test passed separately after fetching its pinned
  parquet file; recorded in `teacher-trajectory.xml`.
- Submission tests: 94 passed (`submission-tests.xml`).
- Pinned organizer joint-lane/measurement-seeding/head-geometry tests: 84 passed.
- Runtime, desktop, VLM and pick Python environments: CUDA matrix multiplication
  passed on NVIDIA GB10 (`aarch64`, compute capability 12.1).
- Every configured weight and the Diffusion variant was prefetched remotely and
  checked with `HF_HUB_OFFLINE=1`. Temporary HF authentication was removed.
- Production file hashes match the built source. Test tools and approved
  organizer dependencies live outside the shipped Python environments/image.

The first setup attempt omitted test files; the first submission-suite attempt
omitted `WEIGHTS.md`. These were test-fixture transfer omissions, corrected before
the passing reruns. No production code was changed to make these tests pass.

## Pending Validation

- GB10 offline model inference, Stage matrix for default and guarded streams,
  operator controls, injected failures, and resource/network audit.

## Interruption

- Default-path Stage 0 passed. Stage 1 was interrupted during model preparation;
  it is not a pass regardless of the harness's raw interrupted-result fields.
- Remaining Stage, guard-path end-to-end, operator-fault and forward-matrix
  checks were not completed. The passed unit/organizer tests above remain valid.
- Test and GPU processes were stopped. No live robot was contacted.
- Logs were saved locally as `interrupted-evidence.tar.gz` in the evidence
  directory above, SHA256
  `c655b02fc2a7580ac08c3758c0f6792697ba63bcae5b4ac0f5b491ff771df57a`.
- Disposable GB10 instance `53360147` was destroyed; the provider returned no
  instance on the final status check. Lifecycle confirmation is recorded in
  `provision.json` in the evidence directory.

## Limits

These tests cannot certify real DDS load, sensor transport latency, physical
tracking, contacts, collisions, or task success on Thor/G1. Guard use at the venue
still needs organizer approval and rig preflight. The existing transport-loss
navigation issue is explicitly out of scope, not fixed by this guard.
