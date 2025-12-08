MUJOCO_GL=egl MJLAB_WARP_QUIET=1 uv run train \
  Mjlab-Velocity-Flat-CCBR-Leo \
  --env.scene.num-envs 4096 \
  --gpu-ids 0 \
  --agent.logger tensorboard \
  --video True \
  --video-interval 2000 \
  --agent.load-checkpoint logs/rsl_rl/ccbr_leo_velocity/2025-12-08_07-14-00/model_1300.pt
