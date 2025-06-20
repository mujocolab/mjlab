from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mjlab.utils import reset as reset_utils
from mjlab.core import entity, mjx_env, mjx_task, step
from mjlab.entities.arenas import FlatTerrainArena
from mjlab.entities.t1 import BoosterT1, get_assets, T1_XML
from mjlab.entities.t1 import t1_constants as consts
from mjlab.entities import robot


class T1(BoosterT1):
  def post_init(self):
    super().post_init()

    for key in consts.KEYFRAMES:
      self.add_keyframe(key, ctrl=key.joint_angles)

    for collision_pair in consts.SELF_COLLISIONS:
      self.add_collision_pair(collision_pair)

    for collision_pair in consts.FLOOR_COLLISIONS:
      self.add_collision_pair(collision_pair)

    self._joint_stiffness = tuple([a.gainprm[0] for a in self._actuators])
    self._joint_damping = tuple([-a.biasprm[2] for a in self._actuators])
    self._default_joint_pos_nominal = self.spec.key("home").ctrl

  @property
  def joint_stiffness(self) -> Tuple[float, ...]:
    return self._joint_stiffness

  @property
  def joint_damping(self) -> Tuple[float, ...]:
    return self._joint_damping

  @property
  def default_joint_pos_nominal(self) -> Tuple[float, ...]:
    return tuple(self._default_joint_pos_nominal.tolist())


@dataclass(frozen=True)
class RewardScales:
  torso_height: float = 1.0
  posture: float = 1.0
  action_rate: float = -0.01
  torques: float = 0.0
  dof_acc: float = 0.0
  dof_vel: float = 0.0
  dof_pos_limits: float = -1.0


@dataclass(frozen=True)
class RewardConfig:
  scales: RewardScales = RewardScales()
  max_velocity: float = 2.0 * jp.pi


@dataclass(frozen=True)
class NoiseConfig:
  joint_pos: float = 0.03
  joint_vel: float = 1.5
  gyro: float = 0.2
  projected_gravity: float = 0.05


@dataclass(frozen=True)
class T1GetupConfig(mjx_task.TaskConfig):
  """T1 configuration."""

  sim_dt: float = 0.004
  ctrl_dt: float = 0.02
  solver_iterations: int = 5
  solver_ls_iterations: int = 8
  integrator: str = "implicitfast"
  euler_damping: bool = False
  max_episode_length: int = 500
  friction_cone: str = "pyramidal"
  noise_level: float = 1.0
  action_scale: float = 0.5
  drop_from_height_prob: float = 0.6
  soft_joint_pos_limit_factor: float = 0.95
  reward_config: RewardConfig = RewardConfig()
  noise_config: NoiseConfig = NoiseConfig()


class T1GetupEnv(mjx_task.MjxTask[T1GetupConfig]):
  """T1 fall recovery task."""

  def __init__(self, config: T1GetupConfig = T1GetupConfig()):
    root, entities = T1GetupEnv.build_scene(config)
    super().__init__(config, root.spec, entities=entities)
    self.t1: T1 = cast(T1, entities["t1"])
    self._reward_scales = asdict(self.cfg.reward_config.scales)

    self._waist_body_id = self.model.body("Waist").id
    self._torso_body_id = self.model.body(consts.TORSO_BODY).id
    self._floor_geom_id = self.model.geom("floor").id
    self._imu_site_id = self.model.site(consts.IMU_SITE).id
    self._init_q = jp.array(self.model.keyframe("home").qpos)
    self._default_pose = jp.array(self.t1.default_joint_pos_nominal)
    jnt_ids = [j.id for j in self.t1.joints]
    lowers, uppers = self.model.jnt_range[jnt_ids].T
    c = (lowers + uppers) / 2
    r = uppers - lowers
    self._soft_lowers = c - 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._settle_steps = int(0.5 / self.sim_dt)
    self._torso_height_des = 0.67
    self._waist_height_des = 0.55

  @property
  def observation_names(self) -> Tuple[str, ...]:
    return (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    )

  @staticmethod
  def build_scene(
    config: T1GetupConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    del config  # Unused.
    assets = get_assets()
    t1_entity = T1.from_file(T1_XML, assets=assets)

    arena = FlatTerrainArena()
    arena.add_skybox()
    arena.floor_geom.priority = 1
    arena.floor_geom.condim = 3
    arena.floor_geom.friction[0] = 0.6

    arena.spec.stat.meansize = 0.03

    frame = arena.spec.worldbody.add_frame()
    arena.spec.attach(t1_entity.spec, prefix="", frame=frame)

    return arena, {"t1": t1_entity, "arena": arena}

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    joint_qpos_ids = jp.array([model.bind(j).qposadr for j in self.t1.joints])

    @jax.vmap
    def _randomize(rng):
      # Floor friction: *U(0.4, 1.0).
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
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

  @property
  def robot(self) -> robot.Robot:
    return self.t1

  def before_step(self, action: jax.Array, state: mjx_env.State) -> mjx.Data:
    motor_targets = self._default_pose + action * self.cfg.action_scale
    return super().before_step(motor_targets, state)

  def initialize_episode(self, data: mjx.Data, rng: jax.Array):
    qpos = jp.zeros(self.mjx_model.nq)

    rng, key = jax.random.split(rng)
    joint_angles = jax.random.uniform(
      key, (self.model.nu,), minval=self._soft_lowers, maxval=self._soft_uppers
    )
    qpos = qpos.at[7:].set(joint_angles)

    # Initialize height and orientation of the root body.
    rng, key = jax.random.split(rng)
    height = 1.0
    qpos = qpos.at[2].set(height)
    quat = jax.random.normal(key, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)

    rng, key = jax.random.split(rng)
    drop_from_height = jax.random.bernoulli(key, self.cfg.drop_from_height_prob)
    qpos = jp.where(drop_from_height, qpos, self._init_q)

    data = data.replace(qpos=qpos, ctrl=qpos[7:])

    # Initialize root body velocity.
    rng, key = jax.random.split(rng)
    data = reset_utils.reset_root_state(
      rng=rng,
      robot=self.t1,
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

    # Let the robot fall and settle.
    data = step(self.mjx_model, data, self._settle_steps)
    data = data.replace(time=0.0)

    info = {
      "rng": rng,
      "last_act": jp.zeros(self.mjx_model.nu),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_scales.keys()}
    metrics["metrics/torso_height_error"] = jp.zeros(())
    metrics["metrics/waist_height_error"] = jp.zeros(())
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
    gyro = self.t1.gyro(self.mjx_model, data)
    projected_gravity = self.t1.projected_gravity(self.mjx_model, data)
    joint_angles = self.t1.joint_angles(self.mjx_model, data)
    joint_velocities = self.t1.joint_velocities(self.mjx_model, data)

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
    local_linvel = self.t1.local_linvel(self.mjx_model, data)
    torso_height = data.site_xpos[self._imu_site_id][2]
    torques = self.t1.joint_torques(self.mjx_model, data)
    root_pos = data.qpos[:3]
    root_quat = data.qpos[3:7]
    waist_height = data.xpos[self._waist_body_id][2]

    obs = jp.hstack(
      [
        noisy_gyro,
        noisy_projected_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_velocities,
        state.info["last_act"],
      ]
    )
    privileged_obs = jp.hstack(
      [
        gyro,
        projected_gravity,
        joint_angles,
        joint_velocities,
        local_linvel,
        torso_height,
        torques,
        root_pos,
        root_quat,
        waist_height,
      ]
    )

    return {
      "state": obs,
      "privileged_state": privileged_obs,
    }

  def get_reward(self, data, state, action, done):
    del done  # Unused.
    torso_height = data.site_xpos[self._imu_site_id][2]
    waist_height = data.xpos[self._waist_body_id][2]
    gravity = data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, 1])
    joint_torques = self.t1.joint_torques(self.mjx_model, data)
    joint_accelerations = self.t1.joint_accelerations(self.mjx_model, data)
    joint_angles = self.t1.joint_angles(self.mjx_model, data)
    joint_velocities = self.t1.joint_velocities(self.mjx_model, data)
    reward_terms = {
      "torso_height": self._reward_height(
        torso_height, waist_height, gravity, state.metrics
      ),
      "posture": self._reward_posture(joint_angles),
      "action_rate": self._cost_action_rate(action, state.info["last_act"]),
      "torques": self._cost_torques(joint_torques),
      "dof_acc": self._cost_dof_acc(joint_accelerations),
      "dof_vel": self._cost_dof_vel(joint_velocities),
      "dof_pos_limits": self._cost_joint_pos_limits(joint_angles),
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

  def _reward_height(
    self,
    torso_height: jax.Array,
    waist_height: jax.Array,
    up_vec: jax.Array,
    metrics: dict[str, Any],
  ) -> jax.Array:
    # Torso height.
    height = jp.min(jp.array([torso_height, self._torso_height_des]))
    error = self._torso_height_des - height
    error_normalized = jp.clip(error / self._torso_height_des, 0.0, 1.0)
    r_torso = 1.0 - error_normalized
    metrics["metrics/torso_height_error"] = error_normalized
    # Waist height.
    waist_height = jp.min(jp.array([waist_height, self._waist_height_des]))
    error = self._waist_height_des - waist_height
    error_normalized = jp.clip(error / self._waist_height_des, 0.0, 1.0)
    r_waist = 1.0 - error_normalized
    metrics["metrics/waist_height_error"] = error_normalized
    r_height = r_torso + r_waist
    # Orientation.
    r_ori = (0.5 * up_vec[2] + 0.5) ** 2
    return r_ori * (r_height + 1.0) / 2.0

  def _reward_posture(self, joint_angles: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(joint_angles - self._default_pose))
    return jp.exp(-0.5 * cost)

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
    return jp.sum(jp.maximum(jp.abs(qvel) - self.cfg.reward_config.max_velocity, 0.0))


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: T1GetupConfig):
    root, _ = T1GetupEnv.build_scene(config=cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(T1GetupConfig))
