MUJOCO_GL=glfw MJLAB_WARP_QUIET=1 uv run train \
  Mjlab-Velocity-Flat-CCBR-Leo \
  --env.scene.num-envs 1024 \
  --gpu-ids None \
  --agent.logger tensorboard \
  --video True \
  --video-interval 50
