# G1 football sim2sim / deployment parity

The Python sim2sim implementation mirrors the `Football_MJLab` policy contract
and the shared vision logic in `klavier_rl_deploy-isaacsim5.1`.

## Matched behavior

| Contract | Python sim2sim | C++ deployment |
|---|---|---|
| Policy rate | 50 Hz (`dt=0.005`, decimation 4) | 50 Hz (`step_dt=0.02`) |
| Policy input/output | `obs[1,520]` / `actions[1,29]` | same migrated MJLab ONNX |
| History layout | term-major, five frames | same |
| Phase | first policy step at phase zero; zero while stopped | `mjlab_phase` |
| Action | raw clip, scale/offset, joint clip, 0.12 rad/step rate clip | same |
| PD gains | MJLab model gains | deployment YAML/ONNX metadata values |
| Camera parent | `torso_link` | `torso_link` |
| Camera pose | football deployment pose/quaternion | same |
| RGB image | 640×480, vertical FOV 42.5° | 640×480 RGB stream |
| Depth | rendered from the same camera, pixel-aligned to RGB | RealSense depth aligned to RGB |
| YOLO input | 320×320 half-pixel bilinear letterbox | same ONNX preprocessing |
| YOLO output | generic layout decode, class filter, NMS, size-prior selection | same |
| Depth sample | 4 px center ROI, then whole-box fallback, upper median | same |
| Ball surface correction | add radius 0.1098 m along camera ray | same |
| Robot coordinates | optical → torso camera → world → gravity-aligned yaw | equivalent FK transform |
| Lost detection | keep last value for 0.5 s | keep until 0.5 s FSM ball-lost timeout |
| Command shaping | direct command, deployment range clamp | direct joystick command |

## Deliberate physical/runtime differences

- MuJoCo provides ideal aligned floating-point depth; RealSense supplies noisy
  quantized depth with device calibration.
- Python uses ONNX Runtime CPU. Deployment normally uses TensorRT/CUDA.
- Sim2sim renders and infers synchronously once per policy step. Deployment has
  separate camera, inference, shared-memory publication, and policy threads and
  consumes the newest completed result.
- Sim2sim does not emulate DDS transport latency, camera frame drops, USB
  stalls, or the controller FSM outside the football state.

The sim2sim command now follows the direct deployment path only. Research-only
command generators, artificial disturbances, and kinematics export utilities
are intentionally kept outside this runtime.
