from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mjlab._src.types import State
from mjlab._src import entity, mjx_task, reward

_HERE = Path(__file__).parent
_CARTPOLE_XML = _HERE / "cartpole.xml"


class Cartpole(entity.Entity):
  """A cartpole entity."""

  def __init__(self, spec: mujoco.MjSpec):
    super().__init__(spec)
    self._slider_joint = spec.joint("slider")
    self._hinge_joint = spec.joint("hinge_1")

  @property
  def slider_joint(self) -> mujoco.MjsJoint:
    return self._slider_joint

  @property
  def hinge_joint(self) -> mujoco.MjsJoint:
    return self._hinge_joint

  def cart_position(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the position of the cart."""
    return data.bind(model, self._slider_joint).qpos

  def angular_vel(self, data: mjx.Data) -> jax.Array:
    """Returns the angular velocity of the pole."""
    return data.qvel[1:]

  def pole_angle_cos(self, data: mjx.Data) -> jax.Array:
    """Returns the cosine of the pole angle."""
    return data.xmat[2, 2, 2]

  def pole_angle_sin(self, data: mjx.Data) -> jax.Array:
    """Returns the sine of the pole angle."""
    return data.xmat[2, 0, 2]

  def bounded_position(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the state, with pole angle split into sin/cos."""
    return jp.hstack(
      [
        self.cart_position(model, data),
        self.pole_angle_cos(data),
        self.pole_angle_sin(data),
      ]
    )


@dataclass(frozen=True)
class CartpoleConfig(mjx_task.TaskConfig):
  """Cartpole configuration."""

  sim_dt: float = 0.01
  ctrl_dt: float = 0.01
  solver_iterations: int = 1
  solver_ls_iterations: int = 4
  euler_damping: bool = False
  max_episode_length: int = 1_000
  integrator: str = "euler"


class Swingup(mjx_task.MjxTask[CartpoleConfig]):
  """Swing up a pole on a cart and balance it."""

  def __init__(self, config: CartpoleConfig = CartpoleConfig()):
    root, entities = Swingup.build_scene(config)
    self.cartpole: Cartpole = cast(Cartpole, entities["cartpole"])
    super().__init__(config, root.spec, entities=entities)

  @staticmethod
  def build_scene(
    cfg: CartpoleConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    root_spec = Cartpole.from_file(_CARTPOLE_XML)
    cfg.apply_defaults(root_spec.spec)
    return root_spec, {"cartpole": root_spec}

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    cart_body_id = self.spec.body("cart").id

    @jax.vmap
    def _randomize(rng):
      rng, key = jax.random.split(rng)
      cart_mass = model.body_mass[cart_body_id]
      body_mass = model.body_mass.at[cart_body_id].set(
        cart_mass * jax.random.uniform(key, (), minval=0.5, maxval=1.5)
      )
      return body_mass

    body_mass = _randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"body_mass": 0})
    model = model.tree_replace({"body_mass": body_mass})
    return model, in_axes

  def initialize_episode(
    self, data: mjx.Data, rng: jax.Array
  ) -> Tuple[mjx.Data, Dict[str, Any], Dict[str, Any]]:
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    data = data.bind(self.mjx_model, self.cartpole.slider_joint).set(
      "qpos", 0.01 * jax.random.normal(rng1)
    )
    data = data.bind(self.mjx_model, self.cartpole.hinge_joint).set(
      "qpos", jp.pi + 0.01 * jax.random.normal(rng2)
    )
    qvel = 0.01 * jax.random.normal(rng3, (self.mjx_model.nv,))
    data = data.replace(qvel=qvel)
    metrics = {
      "reward/upright": jp.zeros(()),
      "reward/centered": jp.zeros(()),
      "reward/small_control": jp.zeros(()),
      "reward/small_velocity": jp.zeros(()),
      "reward/cart_in_bounds": jp.zeros(()),
      "reward/angle_in_bounds": jp.zeros(()),
    }
    info = {"rng": rng}
    return data, info, metrics

  def get_observation(self, data: mjx.Data, state: State):
    del state  # Unused.
    return jp.concatenate(
      [self.cartpole.bounded_position(self.mjx_model, data), data.qvel]
    )

  def get_reward(
    self, data: mjx.Data, state: State, action: jax.Array, done: jax.Array
  ):
    del done  # Unused.

    pole_angle_cos = self.cartpole.pole_angle_cos(data)
    upright = (pole_angle_cos + 1) / 2
    state.metrics["reward/upright"] = upright

    cart_position = self.cartpole.cart_position(self.mjx_model, data)
    centered = reward.tolerance(cart_position, margin=2)
    centered = (1 + centered) / 2
    state.metrics["reward/centered"] = centered

    small_control = reward.tolerance(
      action[0], margin=1, value_at_margin=0, sigmoid="quadratic"
    )
    small_control = (4 + small_control) / 5
    state.metrics["reward/small_control"] = small_control

    angular_vel = data.qvel[1:]
    small_velocity = reward.tolerance(angular_vel, margin=5).min()
    small_velocity = (1 + small_velocity) / 2
    state.metrics["reward/small_velocity"] = small_velocity

    return upright * small_control * small_velocity * centered


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: CartpoleConfig):
    root, _ = Swingup.build_scene(cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(CartpoleConfig))
