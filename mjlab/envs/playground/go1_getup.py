from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from mjlab.utils import reset as reset_utils
from mjlab.core import entity, mjx_env, mjx_task
from mjlab.entities.arenas import FlatTerrainArena
from mjlab.entities.go1 import UnitreeGo1, get_assets, GO1_XML
from mjlab.entities.go1 import go1_constants as consts

_UNWANTED_COLLISIONS = [
  "FR_hip_collision2",
  "FL_hip_collision2",
  "RR_hip_collision2",
  "RR_hip_collision3",
  "RL_hip_collision2",
  "RL_hip_collision3",
]


class Go1(UnitreeGo1):
  """Go1 with custom collision pairs."""

  def post_init(self):
    self.add_pd_actuators_from_patterns(consts.ACTUATOR_SPECS)
    super().post_init()

    # Disable self-collisions, enable collisions with the floor.
    for geom in self.spec.geoms:
      if geom.classname.name == "visual" or geom.name in _UNWANTED_COLLISIONS:
        continue
      geom.contype = 0
      geom.conaffinity = 1
      geom.condim = 1

    self.spec.add_numeric(name="max_contact_points", data=np.array([30]))
    self.spec.add_numeric(name="max_geom_pairs", data=np.array([12]))


@dataclass(frozen=True)
class RewardScales:
  orientation: float = 1.0
  torso_height: float = 1.0
  posture: float = 1.0
  stand_still: float = 1.0
  action_rate: float = -0.001
  dof_pos_limits: float = -0.1
  torques: float = -1e-5
  dof_acc: float = -2.5e-7
  dof_vel: float = -0.1


@dataclass(frozen=True)
class RewardConfig:
  scales: RewardScales = RewardScales()


@dataclass(frozen=True)
class NoiseConfig:
  joint_pos: float = 0.03
  joint_vel: float = 1.5
  gyro: float = 0.2
  projected_gravity: float = 0.05


@dataclass(frozen=True)
class Go1GetupConfig(mjx_task.TaskConfig):
  """Go1 configuration."""

  sim_dt: float = 0.004
  ctrl_dt: float = 0.02
  solver_iterations: int = 5
  solver_ls_iterations: int = 8
  integrator: str = "implicitfast"
  euler_damping: bool = False
  max_episode_length: int = 300
  noise_level: float = 1.0
  action_scale: float = 0.5
  soft_joint_pos_limit_factor: float = 0.95
  reward_config: RewardConfig = RewardConfig()
  noise_config: NoiseConfig = NoiseConfig()


class Go1GetupEnv(mjx_task.MjxTask[Go1GetupConfig]):
  """Go1 fall recovery task."""

  def __init__(self, config: Go1GetupConfig = Go1GetupConfig()):
    root, entities = Go1GetupEnv.build_scene(config)
    super().__init__(config, root.spec, entities=entities)
    self.go1: Go1 = cast(Go1, entities["go1"])
    self._reward_scales = asdict(self.cfg.reward_config.scales)

    self._torso_body_id = self.model.body("trunk").id
    self._floor_geom_id = self.model.geom("floor").id
    self._imu_site_id = self.model.site("imu").id
    self._init_q = jp.array(self.model.keyframe("home").qpos)
    self._default_pose = jp.array(self.model.keyframe("home").qpos[7:])
    jnt_ids = [j.id for j in self.go1.joints]
    lowers, uppers = self.model.jnt_range[jnt_ids].T
    c = (lowers + uppers) / 2
    r = uppers - lowers
    self._soft_lowers = c - 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._settle_steps = int(0.5 / self.sim_dt)
    self._z_des = 0.275
    self._up_vec = jp.array([0.0, 0.0, -1.0])

  @staticmethod
  def build_scene(
    config: Go1GetupConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    del config  # Unused.
    assets = get_assets()
    go1_entity = Go1.from_file(GO1_XML, assets=assets)

    arena = FlatTerrainArena()
    arena.add_skybox()
    arena.floor_geom.contype = 1
    arena.floor_geom.conaffinity = 0
    arena.floor_geom.priority = 1
    arena.floor_geom.condim = 3
    arena.floor_geom.friction[0] = 0.6

    arena.spec.stat.meansize = 0.03

    frame = arena.spec.worldbody.add_frame()
    arena.spec.attach(go1_entity.spec, prefix="", frame=frame)

    return arena, {"go1": go1_entity, "arena": arena}

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    joint_qpos_ids = jp.array([model.bind(j).qposadr for j in self.go1.joints])

    @jax.vmap
    def _randomize(rng):
      # Floor friction: *U(0.4, 1.0).
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=0.6, maxval=1.0)
      geom_friction = model.geom_friction.at[self._floor_geom_id, 0].set(friction)

      # Link masses: *U(0.9, 1.1).
      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
      body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

      # Link center of masses: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      dpos = jax.random.uniform(key, shape=(model.nbody, 3), minval=-0.02, maxval=0.02)
      body_ipos = model.body_ipos.at[:].set(model.body_ipos + dpos)

      # Joint calibration offsets: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      dqpos = jax.random.uniform(
        key, shape=(len(joint_qpos_ids),), minval=-0.05, maxval=0.05
      )
      qpos0 = model.qpos0.at[joint_qpos_ids].set(model.qpos0[joint_qpos_ids] + dqpos)

      return geom_friction, body_mass, body_ipos, qpos0

    geom_friction, body_mass, body_ipos, qpos0 = _randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
      {
        "geom_friction": 0,
        "body_mass": 0,
        "body_ipos": 0,
        "qpos0": 0,
      }
    )
    model = model.tree_replace(
      {
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_ipos": body_ipos,
        "qpos0": qpos0,
      }
    )
    return model, in_axes

  def before_step(self, action: jax.Array, state: mjx_env.State) -> mjx.Data:
    motor_targets = self._default_pose + action * self.cfg.action_scale
    return super().before_step(motor_targets, state)

  def initialize_episode(self, data: mjx.Data, rng: jax.Array):
    qpos = jp.zeros(self.mjx_model.nq)
    qvel = jp.zeros(self.mjx_model.nv)

    rng, key = jax.random.split(rng)
    joint_angles = jax.random.uniform(
      key, (12,), minval=self._soft_lowers, maxval=self._soft_uppers
    )
    qpos = qpos.at[7:].set(joint_angles)

    # Initialize height and orientation of the root body.
    rng, key = jax.random.split(rng)
    height = 0.5
    qpos = qpos.at[2].set(height)
    quat = jax.random.normal(key, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)

    data = data.replace(qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    # Initialize root body velocity.
    rng, key = jax.random.split(rng)
    data = reset_utils.reset_root_state(
      rng=rng,
      robot=self.go1,
      model=self.mjx_model,
      data=data,
      pose_range={},
      velocity_range={
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
      },
    )

    def settle(_, data: mjx.Data) -> mjx.Data:
      return mjx.step(self.mjx_model, data)

    data = jax.lax.fori_loop(0, self._settle_steps, settle, data)
    data = data.replace(time=0.0)

    info = {
      "rng": rng,
      "last_act": jp.zeros(self.mjx_model.nu),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_scales.keys()}
    metrics["metrics/height_error"] = jp.zeros(())
    metrics["metrics/orientation_error"] = jp.zeros(())
    return data, info, metrics

  def _apply_noise(
    self, info: dict[str, Any], value: jax.Array, scale: float
  ) -> jax.Array:
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noise = 2 * jax.random.uniform(noise_rng, shape=value.shape) - 1
    noisy_value = value + noise * self.cfg.noise_level * scale
    return noisy_value

  def get_observation(self, data: mjx.Data, state: mjx_env.State):
    cfg: NoiseConfig = self.cfg.noise_config

    # Ground-truth observations.
    gyro = self.go1.gyro(self.mjx_model, data)
    projected_gravity = self.go1.projected_gravity(self.mjx_model, data)
    joint_angles = self.go1.joint_angles(self.mjx_model, data)
    joint_velocities = self.go1.joint_velocities(self.mjx_model, data)

    # Noisy observations.
    noisy_gyro = self._apply_noise(state.info, gyro, cfg.gyro)
    noisy_projected_gravity = self._apply_noise(
      state.info, projected_gravity, cfg.projected_gravity
    )
    noisy_joint_angles = self._apply_noise(state.info, joint_angles, cfg.joint_pos)
    noisy_joint_velocities = self._apply_noise(
      state.info, joint_velocities, cfg.joint_vel
    )

    # Privileged observations.
    local_linvel = self.go1.local_linvel(self.mjx_model, data)
    torso_height = data.site_xpos[self._imu_site_id][2]
    torques = self.go1.joint_torques(self.mjx_model, data)

    obs = jp.hstack(
      [
        noisy_gyro,
        noisy_projected_gravity,
        noisy_joint_angles,
        noisy_joint_velocities,
        state.info["last_act"],
      ]
    )
    privileged_obs = jp.hstack(
      [
        obs,
        gyro,
        projected_gravity,
        joint_angles,
        joint_velocities,
        local_linvel,
        torso_height,
        torques,
      ]
    )

    return {
      "state": obs,
      "privileged_state": privileged_obs,
    }

  def get_reward(self, data, state, action, done):
    del done  # Unused.
    torso_height = data.site_xpos[self._imu_site_id][2]
    projected_gravity = self.go1.projected_gravity(self.mjx_model, data)
    joint_torques = self.go1.joint_torques(self.mjx_model, data)
    joint_accelerations = self.go1.joint_accelerations(self.mjx_model, data)
    joint_angles = self.go1.joint_angles(self.mjx_model, data)
    joint_velocities = self.go1.joint_velocities(self.mjx_model, data)
    is_upright = self._is_upright(projected_gravity, state)
    is_at_desired_height = self._is_at_desired_height(torso_height, state)
    gate = is_upright * is_at_desired_height
    reward_terms = {
      "orientation": self._reward_orientation(projected_gravity),
      "torso_height": self._reward_height(torso_height),
      "posture": self._reward_posture(joint_angles, is_upright),
      "stand_still": self._reward_stand_still(action, gate),
      "action_rate": self._cost_action_rate(action, state.info["last_act"]),
      "torques": self._cost_torques(joint_torques),
      "dof_pos_limits": self._cost_joint_pos_limits(joint_angles),
      "dof_acc": self._cost_dof_acc(joint_accelerations),
      "dof_vel": self._cost_dof_vel(joint_velocities),
    }
    rewards = {k: v * self._reward_scales[k] for k, v in reward_terms.items()}
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v / self.dt
    reward = sum(rewards.values()) * self.dt
    return reward

  def after_step(
    self,
    data: mjx.Data,
    state: mjx_env.State,
    action: jax.Array,
    done: jax.Array,
  ) -> mjx.Data:
    del done  # Unused.
    state.info["last_act"] = action
    return data

  def should_terminate_episode(self, data: mjx.Data, state: mjx_env.State) -> jax.Array:
    del state  # Unused.
    return jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

  # Reward functions.

  def _is_upright(
    self, gravity: jax.Array, state: mjx_env.State, ori_tol: float = 0.01
  ) -> jax.Array:
    ori_error = jp.sum(jp.square(self._up_vec - gravity))
    state.metrics["metrics/orientation_error"] = jp.sqrt(ori_error)
    return ori_error < ori_tol

  def _is_at_desired_height(
    self, torso_height: jax.Array, state: mjx_env.State, pos_tol: float = 0.005
  ) -> jax.Array:
    height = jp.min(jp.array([torso_height, self._z_des]))
    height_error = self._z_des - height
    state.metrics["metrics/height_error"] = height_error
    return height_error < pos_tol

  def _reward_orientation(self, up_vec: jax.Array) -> jax.Array:
    error = jp.sum(jp.square(self._up_vec - up_vec))
    return jp.exp(-2.0 * error)

  def _reward_height(self, torso_height: jax.Array) -> jax.Array:
    height = jp.min(jp.array([torso_height, self._z_des]))
    return jp.exp(height) - 1.0

  def _reward_posture(self, joint_angles: jax.Array, gate: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(joint_angles - self._default_pose))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _reward_stand_still(self, act: jax.Array, gate: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(act))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
    max_velocity = 2.0 * jp.pi  # rad/s
    cost = jp.maximum(jp.abs(qvel) - max_velocity, 0.0)
    return jp.sum(jp.square(cost))


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: Go1GetupConfig):
    root, _ = Go1GetupEnv.build_scene(config=cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(Go1GetupConfig))
