MUJOCO_GL=glfw MJLAB_WARP_QUIET=1 uv run play \
  --task Mjlab-Velocity-Flat-Unitree-Go1-Play \
  --checkpoint-file logs/rsl_rl/go1_velocity/2025-09-30_22-12-13/model_150.pt \
  --viewer viser \
  # --export \
  # --motion-file not-a-real-file.npz \
