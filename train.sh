MUJOCO_GL=glfw MJLAB_WARP_QUIET=1 uv run train \
  Mjlab-Velocity-Flat-Unitree-Go1 \
  --env.scene.num-envs 1024 \
  --device cpu \
  --agent.logger tensorboard
