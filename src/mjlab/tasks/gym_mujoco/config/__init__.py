from .ant_env_cfg import ant_env_cfg
from .half_cheetah_env_cfg import half_cheetah_env_cfg
from .hopper_env_cfg import hopper_env_cfg
from .humanoid_env_cfg import humanoid_env_cfg
from .humanoid_standup_env_cfg import humanoid_standup_env_cfg
from .inverted_double_pendulum_env_cfg import inverted_double_pendulum_env_cfg
from .inverted_pendulum_env_cfg import inverted_pendulum_env_cfg
from .pusher_env_cfg import pusher_env_cfg
from .reacher_env_cfg import reacher_env_cfg
from .swimmer_env_cfg import swimmer_env_cfg
from .walker2d_env_cfg import walker2d_env_cfg

TASK_CONFIGS = {
  "Mjlab-Gym-Ant-v5": ant_env_cfg,
  "Mjlab-Gym-HalfCheetah-v5": half_cheetah_env_cfg,
  "Mjlab-Gym-Hopper-v5": hopper_env_cfg,
  "Mjlab-Gym-Humanoid-v5": humanoid_env_cfg,
  "Mjlab-Gym-HumanoidStandup-v5": humanoid_standup_env_cfg,
  "Mjlab-Gym-InvertedDoublePendulum-v5": inverted_double_pendulum_env_cfg,
  "Mjlab-Gym-InvertedPendulum-v5": inverted_pendulum_env_cfg,
  "Mjlab-Gym-Pusher-v5": pusher_env_cfg,
  "Mjlab-Gym-Reacher-v5": reacher_env_cfg,
  "Mjlab-Gym-Swimmer-v5": swimmer_env_cfg,
  "Mjlab-Gym-Walker2d-v5": walker2d_env_cfg,
}
