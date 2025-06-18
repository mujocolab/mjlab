from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mjlab.core import entity, mjx_task
from mjlab.envs.toy import reward

_HERE = Path(__file__).parent
_HUMANOID_XML = _HERE / "xmls" / "humanoid.xml"

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10


class Humanoid(entity.Entity):
  """Humanoid entity."""

  def post_init(self):
    self._torso_body = self.spec.body("torso")
    self._head_body = self.spec.body("head")
    self._joints = self.get_non_root_joints()
    self._com_vel_sensor = self.spec.sensor("torso_subtreelinvel")

    extremities = []
    for side in ("left", "right"):
      for limb in ("hand", "foot"):
        extremities.append(self.spec.body(f"{side}_{limb}"))
    self._extremities = tuple(extremities)

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._joints

  def torso_upright(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns projection from z-axes of torso to the z-axes of the world."""
    return data.bind(model, self._torso_body).xmat[2, 2]

  def head_height(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the height of the head above the ground."""
    return data.bind(model, self._head_body).xpos[2]

  def center_of_mass_position(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns position of the center-of-mass."""
    return data.bind(model, self._torso_body).subtree_com

  def center_of_mass_velocity(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns velocity of the center-of-mass."""
    return data.bind(model, self._com_vel_sensor).sensordata

  def torso_vertical_orientation(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the z-projection of the torso orientation matrix."""
    return data.bind(model, self._torso_body).xmat[2]

  def joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the state without global orientation or position."""
    return data.bind(model, self._joints).qpos

  def extremities(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns end-effector positions in egocentric frame."""
    torso_frame = data.bind(model, self._torso_body).xmat
    torso_pos = data.bind(model, self._torso_body).xpos
    torso_to_limb = data.bind(model, self._extremities).xpos - torso_pos
    return torso_to_limb @ torso_frame


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
    self.humanoid: Humanoid = cast(Humanoid, entities["humanoid"])
    super().__init__(config, root.spec, entities=entities)

  @staticmethod
  def build_scene(
    cfg: _HumanoidConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    root_spec = Humanoid.from_file(_HUMANOID_XML)
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
