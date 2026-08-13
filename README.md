# IKEA IROS Assembly Challenge Submission Template — Unitree G1 Policy Showcase

> ## Team RAMEN submission
>
> - **Lane:** `decoupled` ((T,25) task-space)
> - **Images** (linux/arm64, GHCR、digest 指定):
>   - Thor (server): `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:a32494cd2c8fd23b49cc1f4209f7c49e1d4645acd90092ce8b8028a352e3170a`
>   - Orin (client): `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-orin@sha256:8ad1fe45343756d85d3409f3c3af8d1a0cf4b5a64a6350efe8ac3ec1046fb938`
> - **Base images:** Thor `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`、Orin `nvcr.io/nvidia/l4t-jetpack:r35.3.1`
> - **Status (onboarding):** `boundary/` unmodified、`conformance.py --lane decoupled` PASS、別コンテナ e2e (server+client+mock) で action 疎通確認済。現状 Policy は hold-still (RAMEN-Ori は後続で差し替え)。
> - **Run / manifest:** `INSTRUCTIONS.md` と `manifest.yaml` を参照。

You submit **two containers**: a policy server for the Jetson AGX Thor and a
policy client for the Jetson Orin NX onboard the G1. What runs inside them is
entirely yours — framework, model, architecture, whether you use a VLA at all.

The organizer owns three endpoints on the Orin and an independent e-stop.
Those, and nothing else, are the contract.

---

## 1 · The Boundary

```
JETSON AGX THOR                          JETSON ORIN NX (onboard the G1)
192.168.100.1                            192.168.100.2
┌──────────────────────┐                 ┌────────────────────────────────┐
│ components/server.py │                 │ components/client.py           │
│                      │◄── ethernet ───►│                                │
│   your model         │  your transport │   boundary.CameraStream  :5555 │◄─ cameras
│                      │                 │   boundary.StateStream   :5557 │◄─ state
└──────────────────────┘                 │   boundary.ActionSink    :5556 │──► WBC
        YOURS                            └────────────────────────────────┘
                                              YOURS          ORGANIZER'S
```

| Endpoint | Direction | Port | Format |
|---|---|---|---|
| Cameras | you subscribe | `:5555` | msgpack `{"timestamps": {...}, "images": {key: jpeg}}` |
| State | you subscribe | `:5557` | topic `g1_debug` + msgpack `{body_q, base_quat, ...}` |
| Actions | **you bind**, the controller dials in | `:5556` | lane-dependent — see §4 |

**You bind `:5556`.** That is backwards from habit and catches everyone once.
The controller is the long-lived process; your client comes and goes.

Use `boundary/` as shipped. Do not edit it — a local change passes your tests
and fails on the robot.

---

## 2 · Layout

```
ikea_iros_submit/
├── boundary/            ORGANIZER-OWNED — do not modify
│   ├── cameras.py         SUB  :5555
│   ├── states.py          SUB  :5557
│   └── actions.py         BIND :5556   ← the only lane-aware module
├── components/          YOURS — this is what you edit
│   ├── server.py          runs on the Thor. Replace Policy.act().
│   ├── client.py          runs on the Orin. Adapt the loop.
│   └── transport.py       Thor↔Orin link. Replace it if you like.
├── mocks/               develop with no robot
│   ├── mock_orin.py       fake cameras + fake state
│   └── mock_wbc.py        fake controller; validates what you publish
├── conformance.py       run this before you ship
└── requirements.txt
```

Realistically you will edit **`components/server.py`** and little else.

| Path | Owner | What it does |
|---|---|---|
| **`boundary/`** | organizer | The contract. Three endpoints, nothing else. Editing anything here makes your tests pass and the robot fail. |
| `boundary/__init__.py` | organizer | Re-exports `CameraStream`, `StateStream`, `ActionSink`. Import from here, not from the submodules. |
| `boundary/cameras.py` | organizer | Subscribes `:5555`, decodes JPEG, flips BGR→RGB, hands you `(480,640,3)` uint8. Newest frame wins — a slow policy skips frames instead of falling behind. |
| `boundary/states.py` | organizer | Subscribes `:5557` topic `g1_debug`, validates the schema, returns `RobotState` with `body_q (29,)` and `base_quat (4,)`. Same on both lanes. |
| `boundary/actions.py` | organizer | **Binds** `:5556` and publishes your actions. The only lane-aware file: `SonicSink` frames latents for the deploy, `DecoupledSink` sends `(T,25)` chunks. Validates every action and raises `ActionError` rather than publish a malformed one. |
| **`components/`** | **you** | Your submission. Both files ship as working references — replace as much as you like. |
| `components/server.py` | **you** | **The file you replace.** Runs on the Thor. Load your model in `__init__`, put inference in `Policy.act()`, keep `metadata`/`act`/`reset`. Ships a hold-still policy so the pipeline runs before your model exists. `--delay-ms` fakes inference time. |
| `components/client.py` | **you** | Runs on the Orin. Reads both endpoints, calls the Thor, publishes actions. Already pipelined: next inference starts while the current chunk plays, and rows made stale by latency are skipped. Warns when your chunk is too short for your latency. |
| `components/transport.py` | **you** | The Thor↔Orin link — WebSocket + msgpack with a numpy hook. Entirely inside your submission; the organizer never speaks it. Swap it for gRPC or anything else. |
| **`mocks/`** | organizer | Fake robot, so you can develop with no hardware. |
| `mocks/mock_orin.py` | organizer | Stands in for the Orin: publishes synthetic cameras on `:5555` and state on `:5557` in the exact real formats. Defaults to a Dex1-1 rig (no hand state). |
| `mocks/mock_wbc.py` | organizer | Stands in for the whole-body controller: dials into `:5556`, decodes what you publish, prints `ACTION REJECTED` with a reason. If this is happy, the real controller will parse you. |
| **`conformance.py`** | organizer | Runs all four processes and gives one PASS/FAIL. Run it before every submission and before every bench session. |
| `requirements.txt` | organizer | Template deps only — numpy, msgpack, pyzmq, opencv-python, websockets. Your model's deps are yours. |
| `.gitignore` | — | Keeps `__pycache__` and model weights out of git. |

---

## 3 · Quickstart

```bash
pip install -r requirements.txt

# Prove the unmodified template passes, so you know what a pass looks like.
python conformance.py --lane sonic
```

Then develop against the mocks, in four terminals:

```bash
python mocks/mock_orin.py                              # fake cameras + state
python components/server.py --lane sonic --port 8765   # your model
python components/client.py --lane sonic --thor 127.0.0.1 --orin 127.0.0.1
python mocks/mock_wbc.py --lane sonic                  # validates your actions
```

`mock_orin.py` publishes a Dex1-1 rig by default — **no hand state**, matching
the real robot. `--with-hands` simulates a Dex3 rig if you need it.

---

## 4 · Lanes

Your lane follows from your model. It is a declaration, not a preference.

| Your model | Lane | Action format |
|---|---|---|
| GR00T N1.7 + `UNITREE_G1_SONIC` | `sonic` | `motion_token (T,64)` + `left/right_hand_joints (T,7)` |
| Pi 0.5, GR00T N1.6, MolmoAct2, non-VLA methods | `decoupled` | `actions (T,25)` |

Only GR00T N1.7 with the SONIC embodiment emits a 64-dim motion token. If
yours does not, you are on `decoupled`.

Declare the lane in your manifest. The bench brings up a **different
controller** for each, so a wrong declaration is caught before you get robot
time — not during your slot.

### `sonic`

Rows stream to the controller at 50 Hz. `boundary` rejects any chunk with
`max|motion_token| > 1.25` — a training-range bound, not a correctness check.
A latent inside the bound can still be nonsense; nothing outside the paired
decoder can tell.

### `decoupled`

One `(T,25)` chunk per message. The organizer runs inverse kinematics and
drives the controller, so you need neither pinocchio nor ROS 2 in your
container. Fixed row layout:

| Columns | Meaning |
|---|---|
| `[0:2]` | left hand, 2 finger joints, `-1` open → `+1` closed |
| `[2:4]` | right hand, same convention |
| `[4:7]` | left end-effector position, xyz, metres |
| `[7:11]` | left end-effector quaternion, **`(w, x, y, z)`** |
| `[11:14]` | right end-effector position |
| `[14:18]` | right end-effector quaternion, `(w, x, y, z)` |
| `[18:21]` | `navigate_cmd` — vx, vy, yaw rate |
| `[21]` | `base_height_cmd` |
| `[22:25]` | torso orientation, roll-pitch-yaw |

Quaternions must be unit length and **`w`-first**. `boundary` checks the norm,
but a `(x,y,z,w)`-ordered quaternion is still unit length and passes — so
nothing catches wrong ordering for you. Verify it by hand; on the robot it
shows up as a rotated end-effector, not an error message.

---

## 5 · Observations

### Cameras (`:5555`)

| Key | Shape | Notes |
|---|---|---|
| `ego_view` | `(480,640,3)` uint8 RGB | head camera — **always present** |
| `left_wrist` | `(480,640,3)` uint8 RGB | RealSense D405 — may be absent |
| `right_wrist` | `(480,640,3)` uint8 RGB | RealSense D405 — may be absent |

JPEGs are BGR on the wire; `boundary/cameras.py` flips to RGB for you. The
server does not publish until `ego_view` is live, and drops individual wrist
keys when those cameras fail. Survive a missing wrist key.

### State (`:5557`)

| Key | Shape | Notes |
|---|---|---|
| `body_q` | `(29,)` float32 | canonical G1 body joints, radians |
| `base_quat` | `(4,)` float32 | base orientation, wxyz |
| `left_hand_q` | `(7,)` float32 | **usually absent** — Dex1-1 rig |
| `right_hand_q` | `(7,)` float32 | usually absent |

One schema for both lanes, whichever controller the organizer is running —
so your client never needs ROS 2.

`body_q` follows Unitree's canonical G1 29-DoF ordering (`G1JointIndex`).
Slice it however your model was trained — no subset or reordering is imposed.

Angles are **radians**. Limits are the mechanical ranges, listed so you can
sanity-check your own outputs; the controller enforces them, not `boundary`.

| # | Joint | Limit (rad) | | # | Joint | Limit (rad) |
|---|---|---|---|---|---|---|
| 0 | `L_LEG_HIP_PITCH` | −2.5307 … 2.8798 | | 15 | `L_SHOULDER_PITCH` | −3.0892 … 2.6704 |
| 1 | `L_LEG_HIP_ROLL` | −0.5236 … 2.9671 | | 16 | `L_SHOULDER_ROLL` | −1.5882 … 2.2515 |
| 2 | `L_LEG_HIP_YAW` | −2.7576 … 2.7576 | | 17 | `L_SHOULDER_YAW` | −2.618 … 2.618 |
| 3 | `L_LEG_KNEE` | −0.0873 … 2.8798 | | 18 | `L_ELBOW` | −1.0472 … 2.0944 |
| 4 | `L_LEG_ANKLE_PITCH` | −0.8727 … 0.5236 | | 19 | `L_WRIST_ROLL` | −1.9722 … 1.9722 |
| 5 | `L_LEG_ANKLE_ROLL` | −0.2618 … 0.2618 | | 20 | `L_WRIST_PITCH` | −1.6144 … 1.6144 |
| 6 | `R_LEG_HIP_PITCH` | −2.5307 … 2.8798 | | 21 | `L_WRIST_YAW` | −1.6144 … 1.6144 |
| 7 | `R_LEG_HIP_ROLL` | −2.9671 … 0.5236 | | 22 | `R_SHOULDER_PITCH` | −3.0892 … 2.6704 |
| 8 | `R_LEG_HIP_YAW` | −2.7576 … 2.7576 | | 23 | `R_SHOULDER_ROLL` | −2.2515 … 1.5882 |
| 9 | `R_LEG_KNEE` | −0.0873 … 2.8798 | | 24 | `R_SHOULDER_YAW` | −2.618 … 2.618 |
| 10 | `R_LEG_ANKLE_PITCH` | −0.8727 … 0.5236 | | 25 | `R_ELBOW` | −1.0472 … 2.0944 |
| 11 | `R_LEG_ANKLE_ROLL` | −0.2618 … 0.2618 | | 26 | `R_WRIST_ROLL` | −1.9722 … 1.9722 |
| 12 | `WAIST_YAW` | −2.618 … 2.618 | | 27 | `R_WRIST_PITCH` | −1.6144 … 1.6144 |
| 13 | `WAIST_ROLL` | −0.52 … 0.52 | | 28 | `R_WRIST_YAW` | −1.6144 … 1.6144 |
| 14 | `WAIST_PITCH` | −0.52 … 0.52 | | | | |

Groups: legs `0–11` (6 per leg), waist `12–14`, left arm `15–21`, right arm
`22–28`.

**Ankles have two names for the same indices.** `4/5` and `10/11` appear as
both `ANKLE_PITCH`/`ANKLE_ROLL` and `ANKLE_B`/`ANKLE_A` depending on whether
the controller is in PR or AB mode. Same slots, different convention.

**Waist roll and pitch (`13`, `14`) can be mechanically locked** on some G1
builds, leaving yaw-only control. Do not assume those two carry meaningful
signal until you have seen live data from the competition robot.

---

## 6 · Hardware

| | Jetson AGX Thor | Jetson Orin NX |
|---|---|---|
| **Role** | policy server — your model | policy client — your control loop |
| **Container** | yours | yours |
| **Address** | `192.168.100.1` | `192.168.100.2` |
| **OS** | JetPack 7, aarch64 | JetPack 5.1.1 (L4T R35.3.1), aarch64 |
| **CPU** | Arm Neoverse-V3AE, 14 cores, 2.6 GHz | Arm Cortex-A78AE, 8 cores / 8 threads, 2.0 GHz |
| **Cache** | 1 MB L2 per core + 16 MB shared L3 | 2 MB L2 + 4 MB L3 |
| **GPU** | Blackwell, 2560 CUDA cores, 5th-gen tensor cores, **sm_110** | Ampere, 1024 CUDA cores, 32 tensor cores, 918 MHz, **sm_87** |
| **CUDA** | 13.0 | 11.4 |
| **Python** | 3.12 | 3.8 default |
| **Memory** | **128 GB unified** LPDDR5X, 256-bit, 273 GB/s | **16 GB unified** (shared CPU + GPU) |
| **Storage** | NVMe over PCIe | 2 TB |
| **Power** | 40–130 W | — |
| **Graphics** | — | OpenGL 4.6, OpenCL 3.0 |

**Both machines use unified memory** — CPU and GPU share one physical pool. On
the Thor that is a luxury (128 GB, no host-to-device copy). On the Orin it is a
constraint: **16 GB total, not 16 CPU + 16 GPU.** Keep the client light — it
moves images and actions, it does not infer. Your model belongs on the Thor.

**The two GPUs are different architectures.** Thor is Blackwell `sm_110`, Orin
NX is Ampere `sm_87`. A wheel built for one will not load on the other, so
build your two containers separately.

NVIDIA quotes **2070 TFLOPS (FP4, sparse)** for the Thor. That is a ceiling for
a quantized sparse workload, not throughput you will see from an unquantized
checkpoint — size your model against the 128 GB and your own measured latency.

Robot: **Unitree G1 EDU**, 29 body DoF, **Dex1-1** two-finger grippers (not
Dex3). Cameras: 1× head + 2× RealSense D405, all 480×640×3.

---

## 7 · Safety

- `boundary/` validates every action before publishing and refuses malformed
  ones. That is a floor, not a safety system.
- Submissions are **video pre-screened**: show your policy working in
  simulation or on your own G1 before it runs on ours.

---

## 8 · Submitting

1. `python conformance.py --lane <your lane>` passes against the **current**
   template.
2. Policy server (Thor) and policy client (Orin).
3. A manifest declaring your lane, your entrypoints, and your ports.
4. Your pre-screen video.