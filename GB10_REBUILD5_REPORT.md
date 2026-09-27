# Rebuild5 GB10 Validation

## Identity and Scope

- Image: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:20260927-rebuild5`
- Registry digest: `sha256:b58c2cd594955a092a7481a21f866577127c40f178748c64b9216d250a24939f`
- Image config ID: `sha256:78a13b49593979d7f58e763174808a4c0ccca95269830125fdcb0fc7364b801e`
- RAMEN source: `e3a41877ea76bfeb106de7afa991c017ad57ce76`; all 300 copied files match by SHA-256.
- Organizer source: `497f3ab93e5baa706311daebd31c7a9798258450`.
- Hardware: Vast instance `52806874`, NVIDIA GB10, ARM64, integrated GPU, compute capability 12.1, driver 595.91.07.
- Creation: 2026-09-26 19:29:43 UTC / 2026-09-27 04:29:43 JST.
- Quoted compute/storage rate: USD 0.7157407407/hour; ingress USD 0.0390625/GB. This is not an invoice.
- Robot communication: none. Cameras, state and action transport use an isolated mock environment.
- No production source or organizer file was modified inside the tested image. Extra test files/tools were uploaded separately.
- Vast adds SSH/bootstrap utilities above the pinned image. This is not a pristine Thor host.

## First-Round Completed Checks

- All four environments execute finite GPU bf16 matrix multiplication: runtime (torch 2.12.1), desktop (2.11.0), VLM (2.13.0), pick (2.11.0), CUDA 13.0.
- All default/candidate checkpoints and extra DP weights download and pass the offline resolver check. The temporary HF token was removed afterward.
- Runtime regression tests: 298 passed on GB10, covering entrypoint, operator console, boundary sink, gravity compensation, operator gate and boundary state.
- Default Stage 0/1/2/3/4/5 preflights passed with exit 0 and an explicit `validation passed; NO command sent` marker. The default startup times were 3/214/189/202/193/12 seconds, respectively. VLM cold loading accounts for most of the Stage 1-4 startup time.
- Organizer camera-geometry and measured-seed tests: 20 passed. The separate organizer joint-lane suite was not run successfully because its `msgpack_numpy` dependency is not in the client image; it is not silently counted as passed.
- Real model forward tests: 11 configurations x 90 calls, with finite 19D public actions and correct state assembly. Where supported, reset is exercised halfway through.
- Operator retry: Enter -> policy -> R -> open hands/hold -> R -> initial pose -> Enter -> policy -> Ctrl+C -> return; passed.
- Operator next: Stage 2 rotate policy -> N -> transition -> pick initial-pose confirmation gate -> Ctrl+C -> return; passed with exit 0. This tests the transition, not physical picking success.
- Camera and state interruptions: both enter safety decision hold and return only after explicit Enter; passed with exit 2.
- Unreachable initial pose: the standard non-following mock triggers safety decision hold without starting the policy. Explicit Enter attempts return; return also cannot converge against this mock and is logged as failed. Exit 2 after 65 seconds is the expected fault-test result, **not a successful physical return**.
- Worker failure: SIGKILL to the owned GR00T worker enters safety decision hold in approximately 0.2 seconds, before any return motion; explicit Enter returns and exits with the expected error status 1.
- Joint packet observers reject malformed shape/dtype, nonfinite values, invalid hand commands and incorrect default base height. Retry/camera/state tests respectively observed 394/319/316 valid messages with no packet errors.
- Retry/camera/state/worker traces all contain 16-row chunks, zero navigation commands, base height 0.74 m and nondecreasing `issued_at`. These Stage 5 traces do not test braking from an active walk, nor clock alignment between two real hosts.
- Local lightweight test-tool/manifest/conformance regression: 61 passed. No local checkpoint loading or local GPU testing was performed.

The nonphysical follower uses bounded motion only to exercise application sequencing. It does not model gravity, contacts, WBC dynamics, physical torque, measured hardware latency or grasp success.

## Forward Timing

Synthetic 640x480 BGR inputs and fixed FK-consistent state. Times include the probe's preprocessing/YOLO and policy call. Cached actions and asynchronous workers mean these are **not** independent GPU inference latency or a control-rate guarantee. The isolated pick API returns a chunk each call; other APIs may return queued single rows.

| Variant | Calls | Load + warmup (s) | Call wall p50 / p95 / max (ms) |
|---|---:|---:|---|
| rotate_table_base_ramen_ori_141_c32_state_dropout | 90 | 3.52 | 25.20 / 84.68 / 115.14 |
| groot_pick_legs_v1 | 90 | 6.98 | 162.93 / 166.02 / 168.18 |
| groot_insert_leg_200k | 90 | 8.71 | 2.61 / 4.38 / 231.07 |
| groot_rotate_leg_200k | 90 | 8.81 | 2.07 / 3.52 / 235.81 |
| groot_flip_table_n17_2_legacy_fk | 90 | 8.61 | 2.19 / 3.47 / 267.86 |
| insert_table_leg_ramen_ori_141_c32_state_dropout | 90 | 3.68 | 25.31 / 84.79 / 115.04 |
| rotate_table_base_ramen_ori_all6_400k | 90 | 4.24 | 24.94 / 83.38 / 115.21 |
| insert_table_leg_ramen_ori_all6_400k | 90 | 3.48 | 24.74 / 84.82 / 121.06 |
| rotate_leg_to_tighten_ramen_ori_all6_400k | 90 | 3.56 | 25.01 / 83.35 / 112.37 |
| flip_table_ramen_ori_all6_400k | 90 | 3.49 | 24.89 / 72.62 / 112.98 |
| rotate_table_base_diffusion | 90 | 5.20 | 61.06 / 66.41 / 226.78 |

## Stage Variants and Memory

| Preflight | Exit | Seconds |
|---|---:|---:|
| Stage 2 all6_400k | 0 | 173 |
| Stage 2 Diffusion rotate override | 0 | 203 |
| Stage 5 all6_400k | 0 | 6 |
| Stage 5 pose lane | 0 | 14 |
| Stage 5 pose lane + wrist clamp | 0 | 11 |
| Stage 0 walk-lowering-check converged | 0 | 3 |

Every row includes the explicit preflight completion marker. These checks do not command physical hardware. Among the default Stage runs, the sampled GPU process memory peaks at 42.69 GiB and `MemAvailable` stays at or above 44.94 GiB. These are GB10 measurements, not guaranteed Thor memory or latency bounds.

## First-Round Pending Checks

At the end of the first round, new-image outbound-connect tracing was incomplete. `strace` installation was not approved during that run, so it was not installed. The instance was destroyed after evidence recovery rather than left billing while waiting. The follow-up below records the additional checks; the first round alone did not satisfy the merge gate.

The first-round organizer joint-lane suite was blocked by missing dependencies. Real Thor/G1 transport, WBC gains, gravity-offset tuning, braking from a walk, physical clearances and task success remain venue checks. A software follower cannot certify these.

The summarizer now distinguishes an absent network trace from a zero-external-connect trace. A successful process exit without the preflight completion marker also fails validation. Test helpers are outside the Docker build inputs; they do not change the tested production image.

## Interpreting the Evidence

- The first Stage 0 attempt preceded YOLO download and correctly failed offline. Its log is retained; the complete-cache rerun passed.
- The first forward probe incorrectly expected pick's native 38D output. The current worker correctly exposes a decoded 19D output; the probe was corrected, not the model.
- The first automated operator attempt sent Enter inside the console's intentional 150 ms suppression window. The test now waits 400 ms and preserves the production guard.
- Worker fault injection enumerates children of every thread in the owned process tree, since pixi can spawn Python from a non-main thread. It does not kill workers by a global name search.
- During safety decision hold, there may be no new client packets. The organizer adapter refreshes its last accepted goal via its own keepalive. This does not establish correctness of the real WBC configuration; confirm that adapter at the venue.
- GB10 (sm_121) is not Thor (sm_110). CUDA execution here does not prove real camera mounting, DDS transport, PD gains, balancing, contact dynamics, collision clearance or task success.

Evidence is collected under `outputs/gb10_validation/20260927_rebuild5/` in the local `iros_2026_ramen` workspace. Raw model weights and generated logs are not Git content. All earlier image measurements in `VERIFY.md` remain historical and separate.

## Cleanup and Evidence

- Final process check: no application workers, VLM, test listeners on ports 5555/5556/5557/8000, or GPU processes remained.
- Archive: `gb10-evidence.tar.gz`, 1,282,510 bytes, 517 entries, SHA-256 `31f1fb5b64a4550cb6ab75f763849de95e2a69a8680484d7271bdd74706f06f6`.
- The archive includes failed first attempts as well as successful reruns, operator events, wire arrays, model-call traces, resource samples and VLM/application logs. It excludes weights and credential files.
- Vast instance `52806874` deletion succeeded at 2026-09-26 20:35:08 UTC (2026-09-27 05:35:08 JST), and absence was confirmed at 20:35:19 UTC. The unrelated L4 instance was left running and unchanged.
- Creation-to-deletion interval was approximately 65.4 minutes. Quoted compute/storage estimate is USD 0.78, **excluding ingress/egress and other account usage**. No invoice total is claimed.
- No production-image inputs changed after build commit `56c69c5`; subsequent changes are verification tools and documentation.

## Follow-Up Validation

The same image digest was additionally validated on isolated GB10 instance `52870045`.
Only loopback mock endpoints are used. No real robot commands are sent.

### Organizer Adapter Tests

- Organizer checkout is unchanged at `497f3ab93e5baa706311daebd31c7a9798258450`.
- WBC model assets are from `NVlabs/GR00T-WholeBodyControl@a0732b642c0333077e127a2f56ab0014c196bca4`.
  All 65 G1 LFS assets (52,642,960 bytes) were downloaded and checked against their LFS SHA-256 and size.
- A separate `--system-site-packages` validation venv reuses the image's numpy 1.26.4, scipy 1.15.2 and
  pinocchio 3.1.0. It does not alter the runtime environment or the production image.
- User-approved validation-only packages: msgpack-numpy 0.4.8, loop-rate-limiters 1.2.0,
  pin-pink 4.3.0, qpsolvers 4.13.0, quadprog 0.1.13; build tools wheel 0.45.1 and Cython 3.0.12.
  The solver/Pink/qpsolvers versions match the organizer environment file. Pinocchio remains 3.1.0,
  rather than the organizer's 2.7.0: this is a compatibility test, not an exact clone of the Orin environment.
- PyPI metadata, hashes, dependencies and build entrypoints were checked. Automated malware scanning was unavailable.
  Unrelated dependencies proposed by a broad dry run were not installed; installs used `--no-deps`.
- `reference/wbc_adapter/tests reference/orin_bridge/tests`: **84 passed in 110.17 seconds**.
- The earlier compatible-environment attempt was **83 passed / 1 failed** with Pink 3.4.0,
  qpsolvers 4.8.1 and proxqp. The optional wrist-roll hard-cap IK case exceeded its limit by
  approximately 1.45e-6 rad. The failure disappeared with organizer-specified solver/library versions;
  no safety limit, test expectation or organizer source was loosened to pass it.

### Additional Runtime Checks

- Continuous Stage 0-5 passed with exit 0: all 16 policy visits ran for 30 seconds each, with
  operator Enter/N gates and Ctrl+C return. The active interval from the first confirmation prompt
  to the final Ctrl+C was 693.76 seconds (model loading is excluded).
  N intentionally advances each visit; this does not establish autonomous completion or execution
  of every perception-dependent hybrid pick/IK/handover branch on real observations.
- The full run recorded 5,298 valid 16-row joint packets. Decoded rows were replayed into the unchanged
  organizer adapter's fake backend: 5,298 accepted, zero rejected/stale/clamped. Left/right arms,
  Dex1, navigation, base height and waypoint schedule all matched. This legacy decoded-row replay
  reconstructs the envelope and refreshes its timestamp; it is not an inter-host clock/latency test.
- Full-run strace observed only loopback and Unix connections, with zero external connect attempts.
  Its result was separately checked for all 16 visits and for application/return-path errors.
- A second traced model matrix passed all 11 configurations x 90 calls, or 990 calls. It made
  16 Unix-socket connections and zero external connect attempts. Default and configured candidate
  sets, plus the DP override, were exercised sequentially.
- Sampled full-run peak GPU process memory was 49.46 GiB; minimum `MemAvailable` was 31.99 GiB.
  Keeping every Stage's expert resident increases the peak over individual-Stage preflights.
- The flip policy ran for 600 seconds, then returned through its normal Ctrl+C shutdown and exited 0.
  Its 5,306 original wire packets passed the organizer decoder and fake-backend replay, with zero
  rejected/stale/clamped packets and matching arm, hand, navigation, height and waypoint values.
- Soak strace observed only loopback/Unix connections and zero external attempts. Peak GPU process
  memory was 7.12 GiB and minimum `MemAvailable` 87.37 GiB; middle-half GPU samples stayed at exactly
  7,288 MiB. This finite test did not show GPU-memory growth; it is not an unbounded leak guarantee.
- Local lightweight regression: 69 passed. This includes new framed-capture and required-trace checks.
- Final application processes, GPU processes and listeners on 5555/5556/5557/8000: none.
- The completion audit deliberately records `normal_path_checks_passed=true` and `all_faults_passed=false`,
  because the client-loss finding below remains unresolved.

### Follow-Up Evidence and Cleanup

- Local directory: `outputs/gb10_validation/20260927_rebuild5_completion/` in `iros_2026_ramen`.
- Archive: `gb10-completion-evidence.tar.gz`, 14,978,997 bytes, 113 entries,
  SHA-256 `fb3fa4ab445260fa8f6c19def68e467ba871680e5d3ae84ca64482503b8f4b07`.
- Archive includes official-test failures and successful reruns, traces, raw packets, row replay,
  resource samples and application records. No weights are included. Transfer hash was verified,
  and no credential patterns were found in text artifacts. Temporary HF credentials were absent.
- No Docker COPY inputs changed. The tested digest remains valid; verification tools and documents
  are outside the image. The source repository's tracked files were left unchanged.
- Instance `52870045` was deleted and absence confirmed at 2026-09-27 03:53:58 UTC
  (12:53:58 JST). The unrelated L4 instance `52331653` remained untouched.
  Creation-to-deletion compute/storage estimate is USD 0.59, excluding network charges and other usage.

### Client-Loss Safety Finding

An additional fake-backend test sends one waypoint with `navigate_cmd=[0.08,0,0]`, then stops sending
client packets. After the scheduled waypoint expires, the unchanged organizer adapter continues
publishing `[0.08,0,0]` through three keepalives at approximately 0.2-second intervals. It does **not**
automatically replace navigation with zero. This is recorded in `client-loss-keepalive.json` and follows
`wbc_driver._publish_trajectory` storing `nav_cmds[-1]` and `_publish_keepalive` reusing that template.

The same behavior was reproduced using the actual production run's message 24: a 16-row joint chunk
whose final navigation is `[0.185,0,0]`. After its 0.8-second schedule expired, all three keepalives
still contained `[0.185,0,0]`. See `client-loss-production-packet.json` and
[Issue #14](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/issues/14).

This is not a demonstrated real-robot walk or injury outcome. It is a concrete limit on the claimed
fault coverage: normal Ctrl+C, camera/state faults handled by the live application, and loss of the
entire client/transport are different cases. The latter requires organizer-side safety confirmation
and a present E-stop operator. We did not modify the organizer adapter, weaken its tests or claim an
automatic stop. A production safety guarantee or unattended walking approval is not supported.
