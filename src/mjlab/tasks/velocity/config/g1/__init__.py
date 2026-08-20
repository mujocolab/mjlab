from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOffPolicyRunner, VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_flashsac_runner_cfg, unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


def _unitree_g1_flat_flashsac_env_cfg(play: bool = False):
  """Flat G1 velocity env cfg with the default 512-env count for FlashSAC."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 512
  return cfg


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-FlashSAC",
  env_cfg=_unitree_g1_flat_flashsac_env_cfg(),
  play_env_cfg=_unitree_g1_flat_flashsac_env_cfg(play=True),
  rl_cfg=unitree_g1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)
