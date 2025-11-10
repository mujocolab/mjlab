from mjlab.tasks.registry import register

from .env_cfgs import UNITREE_GO1_FLAT_ENV_CFG, UNITREE_GO1_ROUGH_ENV_CFG

register(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1",
  env_cfg=UNITREE_GO1_ROUGH_ENV_CFG,
  rl_cfg_entry_point=f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
)

register(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1",
  env_cfg=UNITREE_GO1_FLAT_ENV_CFG,
  rl_cfg_entry_point=f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
)
