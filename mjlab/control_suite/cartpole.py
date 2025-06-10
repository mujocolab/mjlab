from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mjlab._src import entity, mjx_env, mjx_task, reward

_HERE = Path(__file__).parent
_CARTPOLE_XML = _HERE / "cartpole.xml"


class Cartpole(entity.Entity):
    def post_init(self) -> None:
        self._slider_joint = self.spec.joint("slider")
        self._hinge_joint = self.spec.joint("hinge_1")

    @property
    def slider_joint(self) -> mujoco.MjsJoint:
        return self._slider_joint

    @property
    def hinge_joint(self) -> mujoco.MjsJoint:
        return self._hinge_joint

    def cart_position(self, data: mjx.Data) -> jax.Array:
        """Returns the position of the cart."""
        return data.bind(self.mjx_model, self._slider_joint).qpos

    def angular_vel(self, data: mjx.Data) -> jax.Array:
        """Returns the angular velocity of the pole."""
        return data.qvel[1:]

    def pole_angle_cos(self, data: mjx.Data) -> jax.Array:
        """Returns the cosine of the pole angle."""
        return data.xmat[2, 2, 2]

    def pole_angle_sin(self, data: mjx.Data) -> jax.Array:
        """Returns the sine of the pole angle."""
        return data.xmat[2, 0, 2]

    def bounded_position(self, data: mjx.Data) -> jax.Array:
        """Returns the state, with pole angle split into sin/cos."""
        return jp.hstack(
            [
                self.cart_position(data),
                self.pole_angle_cos(data),
                self.pole_angle_sin(data),
            ]
        )


@dataclass
class CartpoleConfig(mjx_task.TaskConfig):
    """Cartpole configuration."""

    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    solver_iterations: int = 1
    solver_ls_iterations: int = 4
    euler_damping: bool = False
    max_episode_length: int = 1_000


class Swingup(mjx_task.MjxTask[CartpoleConfig]):
    """Swing up a pole on a cart and balance it."""

    def __init__(self, config: CartpoleConfig = CartpoleConfig()):
        entity = Cartpole.from_file(_CARTPOLE_XML)
        super().__init__(config, entity.spec, entities={"cartpole": entity})

    def after_compile(self) -> None:
        super().after_compile()
        self.cartpole: Cartpole = self._entities["cartpole"]

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

    def get_observation(self, data, state):
        del state  # Unused.
        return jp.concatenate([self.cartpole.bounded_position(data), data.qvel])

    def get_reward(self, data, state, action, done):
        del done  # Unused.

        pole_angle_cos = self.cartpole.pole_angle_cos(data)
        upright = (pole_angle_cos + 1) / 2
        state.metrics["reward/upright"] = upright

        cart_position = self.cartpole.cart_position(data)
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
    import mediapy as mp
    from tqdm.auto import tqdm

    rng = jax.random.PRNGKey(0)
    task = Swingup()
    env = mjx_env.MjxEnv(Swingup())
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng, reset_rng = jax.random.split(rng)
    states = [state := reset_fn(reset_rng)]
    for _ in tqdm(range(task.cfg.max_episode_length)):
        rng, rng_step = jax.random.split(rng)
        action = jax.random.uniform(
            rng_step,
            (task.action_size,),
            minval=-1,
            maxval=1,
        )
        state = step_fn(state, action)
        states.append(state)
    frames = env.render(states, height=480, width=640)
    mp.write_video("cartpole.mp4", frames, fps=(1.0 / task.dt))
