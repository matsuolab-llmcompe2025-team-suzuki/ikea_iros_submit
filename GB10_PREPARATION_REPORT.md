# Recoverable Joint Preparation: GB10 Validation

## Scope and Identity

2026-09-29 JST. Source Issue 181 / submission Issue 16. This is software validation
on an isolated Vast GB10, not a venue test, physics simulation, or robot success claim.
No connection or command was sent to Thor, PC2, DDS, or a physical robot.

- Source: `10b8d736609595bde30c929ddb73ef2f4751c0f1`.
- Image build input: submission `c56fc693fa6231529a9b38c7fa0f3e6f1c766596`.
- Candidate tag: `gb10-preparation-10b8d73`.
- Registry digest: `sha256:6db3fb0b23dcf9a1835fc5a8c82b2d50dd2046090b745a6132f09c03f84757f4`.
- Image config: `sha256:f2eafc93eec07929e1b39f4b9fb48f3b15e11d606dbad4a005509df0a94b728b`.
- CI: [36479757032](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36479757032).

Registry manifest/config hashes, ARM64 architecture and the running container's
source revision have been checked. All 301 production files matched their SHA-256
inventory at setup. Final completion/audit status is stated separately below;
earlier image results must not be substituted for this candidate's results.

The verification helpers are transferred separately and are not image contents.
Vast's SSH bootstrap and the isolated test dependencies modify only the disposable
instance overlay. The production Python environments and organizer code are not patched.

## Earlier Image Checks

These results are for source `809c8c1`, image `sha256:75f45e0c32648b014554e647c7b6f5bf28dc4002a78b5393003f9b16a174ef53`,
build input `201abf8`, CI [36459109393](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36459109393).
Hardware was GB10 ARM64, approximately 121 GiB shared RAM, driver 595.71.05.
All 301 generated production files matched their source-side SHA-256 inventory.

| Check | Evidence | Result |
|---|---|---|
| Four Python environments, CUDA/bf16 | `setup/environments.log` | Pass |
| Default models, variant sets and DP cache | `setup/offline-weights.log` | Offline resolution passed |
| Source regression suite | `runtime-unit.xml` | 675 passed, 1 optional-data skip |
| Previously skipped teacher trajectory | `teacher-trajectory.xml` | 1 passed in existing desktop environment |
| Organizer adapter and bridge | `organizer-tests.xml` | 84 passed |
| Official goto/follower integration | `preparation-adapter.json` | 7 cases passed |
| Submission tests | `submission-tests.xml` | 76 passed |
| Actual model forwards | `forward/matrix.json` | 11 variants, 90 calls each, all passed |
| Stage 0 with `--actuate` | `stage-0/` | Passed; 31 joint / 9 goto packets accepted |
| Stage 1 with `--actuate` | `stage-1/` | Passed; 678 joint / 17 goto packets accepted |
| Stage 2 with filtered tracing | `stage-2-filtered/` | Passed; 904 joint / 20 goto packets accepted |
| Stage 3 with filtered tracing | `stage-3-filtered/` | Passed; 890 joint / 20 goto packets accepted |
| Stage 4 with filtered tracing | `stage-4-filtered/` | Passed; 910 joint / 20 goto packets accepted |
| Stage 5 with filtered tracing | `stage-5-filtered/` | Passed; 109 joint / 11 goto packets accepted |

The seven preparation cases cover nominal arrival, a split exceeding the 15-second
wire limit, late arrival (0.1162 to 0.08445 rad), dropped command plus explicit retry,
position clamp, trajectory interruption, and stale state. An unreachable target must
hold rather than advance. The follower has lag and the same assumed gravity model;
it cannot establish that the real robot has that response.

Model forwards use synthetic images and measured-state fixtures. Stage tests exercise
the production entrypoint, operator keys and real model workers against loopback-only
camera/state publishers and the pinned official adapter with fake robot I/O. They do
not establish visual task recognition, physical grasping, or competition success.

## Test Dependencies and Data

- Organizer: `497f3ab93e5baa706311daebd31c7a9798258450`.
- WBC assets: `a0732b642c0333077e127a2f56ab0014c196bca4`; 65 LFS files, 52,642,960 bytes,
  verified against their LFS object hashes and sizes.
- Approved isolated additions: msgpack-numpy 0.4.8, loop-rate-limiters 1.2.0,
  pin-pink 4.3.0, qpsolvers 4.13.0, quadprog 0.1.13, wheel 0.45.1, Cython 3.0.12.
  Distribution hashes, dry-run and install logs are under `setup/`.
- strace 6.8-0ubuntu2: Ubuntu package, disposable overlay only; no dependency upgrades.
- Existing numpy 1.26.4, scipy 1.15.2 and pinocchio 3.1.0 were reused. This is not an
  exact recreation of the organizer's pinocchio 2.7 environment.
- Automated malware scanning was unavailable; this limitation was disclosed before
  package installation. Package provenance/hash checks are not a malware guarantee.
- Teacher fixture: `Team-RAMEN/IROS2026_RAMEN_HARA_rotate_table_base_merged_v1`
  at `012034b1060816049108bbb108a05bf701716d5e`, `data/chunk-000/file-000.parquet`.
  SHA-256: `ec6f85e509a515a28069202dcda152356a1401c7cf0545f7ba8955896343b92c`.

## Failed Attempts Retained

The first Stage 2 run (`stage-2/`) stopped when head-camera age reached 0.508 seconds,
exceeding the unchanged 0.5-second freshness gate. It is not counted as a pass.
Forced fixture cleanup also required removal of that run's separate VLM process group.

Ordinary `strace -f` stops on unrelated CUDA syscalls even when the printed syscall
set is narrow. A rotate forward comparison measured median wall times of 117.81 ms
with ordinary tracing (90 calls), 31.76 ms without tracing (30 calls), and 55.48 ms with
`--seccomp-bpf` filtered tracing (30 calls). These runs have different sample counts
and are diagnostic timings, not a controlled real-time performance benchmark.

Subsequent interactive probes use `strace --seccomp-bpf -f -e trace=connect,execve`.
Every connect call remains in scope. No camera threshold, production scheduler or
model setting was changed to accommodate the tracer.

The earlier continuous test correctly rejected one Enter while the measured state
was not fresh/stationary. The original test driver never retried, so it timed out;
that run is not counted as a pass. The driver now issues a new Enter only after an
observed rejection, with at most five retries and the original deadline unchanged.

Review also reproduced an input ownership bug: an obsolete confirmation waiter
could consume a later safety-decision Enter or R. Source `1fd8955` invalidates the
old waiter before dequeuing and explicitly cancels gates during shutdown. Targeted
overlay regression tests passed (208 tests, then 13 gate/console tests); these are
not final-image results. New `gate-camera` and `retry-camera` probes cover this
fault through the real entrypoint.

The intermediate image (`1fd8955`, digest `b229830a...`) passed both new fault
probes, 679 source tests, the separate teacher test, 84 organizer tests and 76
submission tests. Its continuous run exposed a separate usability defect: an
accepted Enter could be revoked by the fresh-state recheck without an explanation.
The controller correctly refused to start, but the operator had no retry cue.
Source `10b8d73` adds that cue without changing arrival/freshness thresholds.
Thirteen gate regression tests and a complete 16-policy continuous run passed on
a separate source-copy overlay. Those overlay results are diagnostic evidence,
not validation of the final image.

Setup failures were retained: an incomplete test-only archive was replaced with
all nested test directories; organizer test import paths were corrected; the
teacher fixture hash check was changed from Python 3.11-only `file_digest` to
streamed SHA-256 for Python 3.10. A submission test collided with the active
probe's loopback ports and passed on a serial rerun. Final network tests are
strictly sequential, and these failed attempts are not counted as passes.

On the final image, the first `retry` probe used an obsolete fixture sequence:
after the second R it waited for a leg-placement gate that flip does not have.
The application had correctly reached `flip_table`'s start gate, but the fixture
timed out. The failure and wire capture are retained under
`failed-attempts/retry-obsolete-fixture/`. The probe now follows the actual flip
contract: R, wait for open hands, R, wait for the initial-pose start gate, then
a new Enter. No application code or arrival threshold was changed for this retry.

Vast's update/recycle API reported the new image UUID while the container still
contained source `809c8c1`. The instance was destroyed instead of accepting that
metadata as proof. A subsequent loading-only instance was also destroyed. The final
candidate is provisioned by its exact digest and checked against all source hashes.

## Final-Image Results

GB10 ARM64, driver 595.71.05, 121.7 GiB shared RAM, 350 GB disk.
The image's four Python environments passed CUDA/bf16 computation.

| Check | Result |
|---|---|
| Source regression | 679 passed, 1 optional-data skip; the skipped test then passed separately with the pinned teacher fixture (680 unique passes) |
| Organizer adapter / camera geometry | 84 passed |
| Submission regression | 76 passed; local lightweight subset 25 passed |
| Preparation / official adapter | 7 cases passed: nominal, split, late, drop/retry, clamp, interrupt, stale |
| Real model forwards | 11 variants x 90 calls = 990, all passed |
| Continuous Stages 0-5 | 16 policy intervals x 30 seconds, normal return, exit 0 |
| Stage 0-5 individually | All six passed, including normal return |
| Operator / fault probes | R retry, N next, camera loss, state loss, killed worker, camera loss at start gate and R2 gate all passed |
| Long run | 600-second flip run, normal return |
| Conformance | Pass |

All 15 operator cases recorded zero external `connect()` calls. The continuous
run rejected two Enter attempts when the start conditions were not satisfied;
both recovered with a fresh Enter. No arrival threshold or limit was relaxed.
Peak GPU-process memory in that run was 44.36 GiB; minimum `MemAvailable` was
39.78 GiB. These are sampled GB10 measurements, not Thor memory/latency bounds.

The recorded packets were replayed into the pinned official adapter with fake
robot I/O. All cases passed:

| Case | Joint packets | Goto packets |
|---|---:|---:|
| Continuous 0-5 | 3609 | 57 |
| Stage 0 | 31 | 9 |
| Stage 1 | 716 | 17 |
| Stage 2 | 943 | 20 |
| Stage 3 | 900 | 20 |
| Stage 4 | 906 | 20 |
| Stage 5 | 148 | 11 |
| R retry | 74 | 14 |
| N next | 119 | 14 |
| Camera loss | 36 | 11 |
| State loss | 41 | 11 |
| Worker exit | 38 | 11 |
| Camera loss at Enter gate | 31 | 8 |
| Camera loss at R2 gate | 31 | 8 |
| 600-second run | 2549 | 11 |

Default preflights for Stages 0, 1, 2 and 5 passed. The user then requested PR
completion instead of further validation. The active Stage 5 preflight was
allowed to finish, and the matrix scheduler was deliberately terminated (exit
143) before the next test. That scheduler exit is not an application failure.

The additional untraced Stage 1 comparison, all6 Stage 2/5 preflights, rotate-DP
Stage 2 preflight and pose-fallback Stage 2 preflight were **not run on this final
image**. Their model forwards and the existing unit coverage do not substitute
for those missing entrypoint checks. They are not counted as passes.
`setup/validation-scope.json` records this scope decision, and the final audit
reports `all_planned_preflights_completed=false` explicitly.

Final audit passed for the completed scope: all 301 production hashes remained
unchanged, the final submission regression rerun passed all 76 tests, no runtime
processes or GPU workers remained, and no listener remained on ports
5555/5556/5557/8000. The listener audit used `/proc/net/tcp` and `/proc/net/tcp6`
because this image does not include `ss`. No HF token file remained.

The completed default preflights took 6 / 253 / 267 / 26 seconds for Stages
0 / 1 / 2 / 5 respectively. These include model startup and are not control-loop
latency measurements.

## Evidence and Cleanup

- Local archive: source repo
  `outputs/gb10_validation/20260929_issue181/final-image-evidence.tar.gz`.
- SHA-256: `6aa1cd86866a31b287319d2f8f477dbc0f06707dbc4c471be3e94d910b40a0f2`.
- Size: 18,459,751 bytes; 472 files. The local copy matched the remote hash.
- Includes successful and failed logs, wire captures, traces, resource samples,
  test XML, package provenance, helper hashes, final audit and scope decision.
  Model weights and authentication files are excluded.
- Known token/private-key pattern scan: zero findings. This is not an exhaustive
  secret audit or malware scan. Generated evidence is not committed to Git.
- Owned GB10 instance `53253780` was created at 2026-09-28 20:57:30 UTC and
  destroyed; API absence verified at 2026-09-28 23:11:11 UTC (09-29 08:11:11 JST).
  Earlier owned instances `53226149`, `53243139`, `53244562` were also destroyed.
  Other account instances were not modified.

The source PR targets `develop`; neither it nor the submission branch has been
merged. Subsequent changes to runtime files require a new image and validation.

## Boundaries

- No real tracking, clearance, contact, balance or task-success guarantee.
- Fake adapter I/O is not actual WBC control of the lower body.
- Goto publication has no acceptance ACK. Real missing packets require measured
  arrival monitoring and an operator retry, not an assumption of delivery.
- Organizer-side retransmission of old physical state with a new timestamp cannot
  be identified using receiver age alone.
- Client-loss walking-velocity retention is explicitly outside this issue's scope.
- Initial VLM startup remains several minutes; initialization is completed before
  preparation motion. This is not a latency-free launch.
