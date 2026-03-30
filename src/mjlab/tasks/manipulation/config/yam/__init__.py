from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  yam_lift_cube_env_cfg,
  yam_lift_cube_vision_env_cfg,
  yam_multi_cube_seg_env_cfg,
)
from .rl_cfg import (
  yam_lift_cube_ppo_runner_cfg,
  yam_lift_cube_vision_ppo_runner_cfg,
  yam_multi_cube_seg_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam",
  env_cfg=yam_lift_cube_env_cfg(),
  play_env_cfg=yam_lift_cube_env_cfg(play=True),
  rl_cfg=yam_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam-Rgb",
  env_cfg=yam_lift_cube_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=yam_lift_cube_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=yam_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam-Depth",
  env_cfg=yam_lift_cube_vision_env_cfg(cam_type="depth"),
  play_env_cfg=yam_lift_cube_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=yam_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Multi-Cube-Seg-Yam",
  env_cfg=yam_multi_cube_seg_env_cfg(num_cubes=3),
  play_env_cfg=yam_multi_cube_seg_env_cfg(num_cubes=3, play=True),
  rl_cfg=yam_multi_cube_seg_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
