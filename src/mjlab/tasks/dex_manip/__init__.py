from __future__ import annotations

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import dex_manip_env_cfg
from .rl_cfg import dex_manip_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Dex-Manip",
  env_cfg=dex_manip_env_cfg(),
  play_env_cfg=dex_manip_env_cfg(play=True),
  rl_cfg=dex_manip_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
