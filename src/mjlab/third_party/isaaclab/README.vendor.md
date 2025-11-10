# NVIDIA Isaac Lab (partial copy)

- **Source repository:** https://github.com/isaac-sim/IsaacLab
- **License:** BSD-3-Clause (see [LICENSE](LICENSE))
- **Commit/tag:** `5f71ff479eb121d8aff6c37caaf6768927e1e5c9`

## Local Modifications

- 2025-11-01: `isaaclab_tasks/utils/parse_cfg.py`: Removed verbose print
  statements for configuration parsing to reduce startup output noise
- 2025-10-27: `isaaclab_rl/rsl_rl/exporter.py`: Added `dynamo=False` parameter
  to `torch.onnx.export()` calls for PyTorch 2.9 compatibility
- 2025-10-05: `isaaclab/utils/math.py`: Replaced deprecated
  `quat_rotate_inverse` with `quat_apply_inverse` for newer PyTorch versions

We include only these files to support our code. For full source,
documentation, and updates, refer to the upstream repository.
