from mjlab.tasks.registry import register

from .env_cfgs import (
  G1_FLAT_TRACKING_ENV_CFG,
  G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)

register(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=G1_FLAT_TRACKING_ENV_CFG,
  rl_cfg_entry_point=f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
)

register(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
  rl_cfg_entry_point=f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
)
