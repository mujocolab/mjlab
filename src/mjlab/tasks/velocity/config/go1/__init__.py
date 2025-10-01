# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)
