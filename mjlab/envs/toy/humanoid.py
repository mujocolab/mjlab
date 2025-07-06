from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mjlab.core import entity, mjx_task
from mjlab.envs.toy import reward
from mjlab.entities.robots.mujoco_humanoid import MujocoHumanoid

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10


@dataclass(frozen=True)
class _HumanoidConfig(mjx_task.TaskConfig):
  """Humanoid configuration."""

  move_speed: float = 0.0
  sim_dt: float = 0.005
  ctrl_dt: float = 0.025
  solver_iterations: int = 3
  solver_ls_iterations: int = 5
  euler_damping: bool = False
  max_episode_length: int = 1_000
  integrator: str = "implicitfast"


class _Humanoid(mjx_task.MjxTask[_HumanoidConfig]):
  """A humanoid task."""

  def __init__(self, config: _HumanoidConfig = _HumanoidConfig()):
    root, entities = _Humanoid.build_scene(config)
    self.humanoid: MujocoHumanoid = cast(MujocoHumanoid, entities["humanoid"])
    super().__init__(config, root.spec, entities=entities)

  @property
  def observation_names(self) -> Tuple[str, ...]:
    return (
      "joint_angles",
      "head_height",
      "extremities",
      "torso_vertical_orientation",
      "com_vel",
      "qvel",
    )

  @property
  def robot(self) -> entity.Entity:
    return self.humanoid

  @staticmethod
  def build_scene(
    cfg: _HumanoidConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    root_spec = MujocoHumanoid()
    cfg.apply_defaults(root_spec.spec)
    return root_spec, {"humanoid": root_spec}

  def initialize_episode(
    self, data: mjx.Data, rng: jax.Array
  ) -> Tuple[mjx.Data, Dict[str, Any], Dict[str, Any]]:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    data = data.bind(self.mjx_model, self.humanoid.joints).set(
      "qvel",
      0.1 * jax.random.normal(rng1, (self.mjx_model.nv - 6,)),
    )
    data = data.bind(self.mjx_model, self.humanoid.joints).set(
      "qpos",
      0.25 * jax.random.normal(rng2, (self.mjx_model.nq - 7,)),
    )
    info = {"rng": rng}
    metrics = {
      "reward/standing": jp.zeros(()),
      "reward/upright": jp.zeros(()),
      "reward/stand": jp.zeros(()),
      "reward/small_control": jp.zeros(()),
      "reward/move": jp.zeros(()),
    }
    return data, info, metrics

  def get_observation(self, data, state):
    del state  # Unused.
    return jp.concatenate(
      [
        self.humanoid.joint_angles(self.mjx_model, data),
        self.humanoid.head_height(self.mjx_model, data).reshape(1),
        self.humanoid.extremities(self.mjx_model, data).ravel(),
        self.humanoid.torso_vertical_orientation(self.mjx_model, data),
        self.humanoid.center_of_mass_velocity(self.mjx_model, data),
        data.qvel,
      ]
    )

  def get_reward(self, data, state, action, done):
    del done  # Unused.

    standing = reward.tolerance(
      self.humanoid.head_height(self.mjx_model, data),
      bounds=(_STAND_HEIGHT, float("inf")),
      margin=_STAND_HEIGHT / 4,
    )
    state.metrics["reward/standing"] = standing

    upright = reward.tolerance(
      self.humanoid.torso_upright(self.mjx_model, data),
      bounds=(0.9, float("inf")),
      sigmoid="linear",
      margin=1.9,
      value_at_margin=0,
    )
    state.metrics["reward/upright"] = upright

    stand_reward = standing * upright
    state.metrics["reward/stand"] = stand_reward

    small_control = reward.tolerance(
      action, margin=1, value_at_margin=0, sigmoid="quadratic"
    ).mean()
    small_control = (4 + small_control) / 5
    state.metrics["reward/small_control"] = small_control

    horizontal_velocity = self.humanoid.center_of_mass_velocity(self.mjx_model, data)[
      :2
    ]
    if self._config.move_speed == 0.0:
      move_reward = reward.tolerance(horizontal_velocity, margin=2).mean()
    else:
      move = reward.tolerance(
        jp.linalg.norm(horizontal_velocity),
        bounds=(self._config.move_speed, float("inf")),
        margin=self._config.move_speed,
        value_at_margin=0,
        sigmoid="linear",
      )
      move_reward = (5 * move + 1) / 6
    state.metrics["reward/move"] = move_reward

    return small_control * stand_reward * move_reward


@dataclass(frozen=True)
class HumanoidStandConfig(_HumanoidConfig):
  """Humanoid stand configuration."""

  move_speed: float = 0.0


class HumanoidStand(_Humanoid):
  def __init__(self, config: HumanoidStandConfig = HumanoidStandConfig()):
    super().__init__(config=config)


@dataclass(frozen=True)
class HumanoidWalkConfig(_HumanoidConfig):
  """Humanoid walk configuration."""

  move_speed: float = _WALK_SPEED


class HumanoidWalk(_Humanoid):
  def __init__(self, config: HumanoidWalkConfig = HumanoidWalkConfig()):
    super().__init__(config=config)


@dataclass(frozen=True)
class HumanoidRunConfig(_HumanoidConfig):
  """Humanoid run configuration."""

  move_speed: float = _RUN_SPEED


class HumanoidRun(_Humanoid):
  def __init__(self, config: HumanoidRunConfig = HumanoidRunConfig()):
    super().__init__(config=config)


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: HumanoidRunConfig):
    root, _ = HumanoidRun.build_scene(cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(HumanoidRunConfig))
