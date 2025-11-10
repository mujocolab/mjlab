from mjlab.tasks.registry import register

from .env_cfgs import UNITREE_G1_FLAT_ENV_CFG, UNITREE_G1_ROUGH_ENV_CFG

register(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=UNITREE_G1_ROUGH_ENV_CFG,
  rl_cfg_entry_point=f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
)

register(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=UNITREE_G1_FLAT_ENV_CFG,
  rl_cfg_entry_point=f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
)
