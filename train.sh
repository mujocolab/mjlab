MUJOCO_GL=egl MJLAB_WARP_QUIET=1 uv run train \
  Mjlab-Velocity-Flat-CCBR-Leo \
  --env.scene.num-envs 4096 \
  --gpu-ids 0 \
  --agent.logger tensorboard \
  --video True \
  --video-interval 2000 \
