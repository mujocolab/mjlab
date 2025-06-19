from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Union, cast

import jax
import jax.numpy as jp
import mujoco
import enum
from mujoco import mjx
import numpy as np

from mjlab.utils.collision import geoms_colliding
from mjlab.core import entity, mjx_env, mjx_task
from mjlab.entities.arenas import FlatTerrainArena, PlaygroundTerrainArena, Arena
from mjlab.entities.go1 import UnitreeGo1, get_assets, GO1_XML
from mjlab.entities.go1 import go1_constants as consts
from mjlab.entities import robot
from mjlab.utils import reset as reset_utils


class Go1(UnitreeGo1):
  """Go1 with custom collision pairs."""

  def post_init(self):
    for geom in self.spec.geoms:
      if geom.name not in consts.FEET_GEOMS:
        continue
      geom.contype = 1
      geom.conaffinity = 1
      geom.solimp[:3] = (0.9, 0.95, 0.023)

    super().post_init()

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


def get_rz(
  phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jp.pi) / (2 * jp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jp.where(x <= 0.5, stance, swing)


class Terrain(enum.Enum):
  FLAT = enum.auto()
  PLAYGROUND = enum.auto()


@dataclass(frozen=True)
class CommandConfig:
  min: Tuple[float, float, float] = (-1.5, -0.8, -1.2)
  max: Tuple[float, float, float] = (1.5, 0.8, 1.2)


@dataclass(frozen=True)
class RewardScales:
  tracking_lin_vel: float = 1.0
  tracking_ang_vel: float = 0.5
  lin_vel_z: float = -0.5
  ang_vel_xy: float = -0.05
  flat_orientation: float = -5.0
  dof_pos_limits: float = -1.0
  pose: float = -0.5
  feet_phase: float = 1.0
  torques: float = -0.0002
  dof_acc: float = -2.5e-7
  action_rate: float = -0.01
  energy: float = -0.001
  collision: float = -1.0


@dataclass(frozen=True)
class RewardConfig:
  scales: RewardScales = RewardScales()
  tracking_sigma: float = 0.25
  max_foot_height: float = 0.1


@dataclass(frozen=True)
class NoiseConfig:
  joint_pos: float = 0.03
  joint_vel: float = 1.5
  gyro: float = 0.2
  projected_gravity: float = 0.05
  linvel: float = 0.1


@dataclass(frozen=True)
class Go1JoystickConfig(mjx_task.TaskConfig):
  """Go1 configuration."""

  sim_dt: float = 0.004
  ctrl_dt: float = 0.02
  solver_iterations: int = 5
  solver_ls_iterations: int = 8
  integrator: str = "implicitfast"
  euler_damping: bool = False
  friction_cone: str = "elliptic"
  max_episode_length: int = 1_000
  noise_level: float = 1.0
  action_scale: float = 0.5
  terrain: Terrain = Terrain.FLAT
  soft_joint_pos_limit_factor: float = 0.9
  command_config: CommandConfig = CommandConfig()
  reward_config: RewardConfig = RewardConfig()
  noise_config: NoiseConfig = NoiseConfig()


class Go1JoystickEnv(mjx_task.MjxTask[Go1JoystickConfig]):
  """Go1 joystick task."""

  def __init__(self, config: Go1JoystickConfig = Go1JoystickConfig()):
    root, entities = Go1JoystickEnv.build_scene(config)
    super().__init__(config, root.spec, entities=entities)
    self.go1: Go1 = cast(Go1, entities["go1"])
    self._reward_scales = asdict(self.cfg.reward_config.scales)

    torso_geoms = ["trunk_collision1", "trunk_collision2"]
    self._torso_body_id = self.model.body("trunk").id
    self._floor_geom_id = self.model.geom("floor").id
    self._feet_geom_id = np.array([self.model.geom(n).id for n in consts.FEET_GEOMS])
    self._feet_site_id = np.array([self.model.site(n).id for n in consts.FEET_SITES])
    self._torso_geom_ids = np.array([self.model.geom(n).id for n in torso_geoms])
    self._imu_site_id = self.model.site("imu").id
    self._init_q = jp.array(self.model.keyframe("home").qpos)
    self._default_pose = jp.array(self.go1.default_joint_pos_nominal)
    jnt_ids = [j.id for j in self.go1.joints]
    lowers, uppers = self.model.jnt_range[jnt_ids].T
    c = (lowers + uppers) / 2
    r = uppers - lowers
    self._soft_lowers = c - 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._collision_pairs = []
    for pair in (("FR", "FL"), ("RR", "RL")):
      geom1_id = self.model.geom(pair[0]).id
      geom2_id = self.model.geom(pair[1]).id
      self._collision_pairs.append((geom1_id, geom2_id))

  @property
  def robot(self) -> robot.Robot:
    return self.go1

  @property
  def observation_names(self) -> Tuple[str, ...]:
    return (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "base_lin_vel",
      "actions",
      "command",
      "phase",
    )

  @property
  def command_names(self) -> Tuple[str, ...]:
    return ("twist",)

  @staticmethod
  def build_scene(
    config: Go1JoystickConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    assets = get_assets()
    go1_entity = Go1.from_file(GO1_XML, assets=assets)

    arena: Arena
    if config.terrain == Terrain.PLAYGROUND:
      arena = PlaygroundTerrainArena()
    else:
      arena = FlatTerrainArena()
      arena.add_skybox()

    arena.floor_geom.contype = 1
    arena.floor_geom.conaffinity = 1
    arena.floor_geom.priority = 1
    arena.floor_geom.condim = 3
    arena.floor_geom.friction[0] = 0.6

    arena.spec.stat.meansize = 0.03

    frame = arena.spec.worldbody.add_frame()
    arena.spec.attach(go1_entity.spec, prefix="", frame=frame)

    return arena, {"go1": go1_entity, "arena": arena}

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    joint_dof_ids = jp.array([model.bind(j).dofadr for j in self.go1.joints])
    joint_qpos_ids = jp.array([model.bind(j).qposadr for j in self.go1.joints])

    @jax.vmap
    def _randomize(rng):
      # Floor friction: *U(0.2, 1.0).
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=0.2, maxval=1.0)
      geom_friction = model.geom_friction.at[self._floor_geom_id, 0].set(friction)

      # Joint stiction: *U(0.9, 1.1).
      rng, key = jax.random.split(rng)
      dof_frictionloss = model.dof_frictionloss.at[joint_dof_ids].set(
        model.dof_frictionloss[joint_dof_ids]
        * jax.random.uniform(key, (len(joint_dof_ids),), minval=0.9, maxval=1.1)
      )

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

      return geom_friction, dof_frictionloss, body_mass, body_ipos, qpos0

    geom_friction, dof_frictionloss, body_mass, body_ipos, qpos0 = _randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
      {
        "geom_friction": 0,
        "dof_frictionloss": 0,
        "body_mass": 0,
        "body_ipos": 0,
        "qpos0": 0,
      }
    )
    model = model.tree_replace(
      {
        "geom_friction": geom_friction,
        "dof_frictionloss": dof_frictionloss,
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
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    data = data.replace(qpos=qpos, qvel=qvel)
    rng, key = jax.random.split(rng)
    data = reset_utils.reset_joints_by_scale(
      rng=rng,
      robot=self.go1,
      model=self.mjx_model,
      data=data,
      position_range=(0.5, 1.5),
    )
    rng, key = jax.random.split(rng)
    pose_range = {"x": (-2.5, 2.5), "y": (-2.5, 2.5), "yaw": (-3.14, 3.14)}
    if self.cfg.terrain == Terrain.PLAYGROUND:
      pose_range["z"] = (0.35, 0.35)
    data = reset_utils.reset_root_state(
      rng=rng,
      robot=self.go1,
      model=self.mjx_model,
      data=data,
      pose_range=pose_range,
      velocity_range={
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
      },
    )

    rng, cmd_rng = jax.random.split(rng)
    command = self._sample_command(cmd_rng)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi, jp.pi, 0])
    info = {
      "rng": rng,
      "step": 0,
      "last_act": jp.zeros(self.mjx_model.nu),
      "command": command,
      "phase": phase,
      "phase_dt": phase_dt,
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_scales.keys()}
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
    local_linvel = self.go1.local_linvel(self.mjx_model, data)

    # Noisy observations.
    noisy_gyro = self._apply_noise(state.info, gyro, cfg.gyro)
    noisy_projected_gravity = self._apply_noise(
      state.info, projected_gravity, cfg.projected_gravity
    )
    noisy_joint_angles = self._apply_noise(state.info, joint_angles, cfg.joint_pos)
    noisy_joint_velocities = self._apply_noise(
      state.info, joint_velocities, cfg.joint_vel
    )
    noisy_local_linvel = self._apply_noise(state.info, local_linvel, cfg.linvel)

    obs = jp.hstack(
      [
        noisy_gyro,
        noisy_projected_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_velocities,
        noisy_local_linvel,
        state.info["last_act"],
        state.info["command"],
        jp.concatenate([jp.cos(state.info["phase"]), jp.sin(state.info["phase"])]),
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
      ]
    )

    return {
      "state": obs,
      "privileged_state": privileged_obs,
    }

  def get_reward(self, data, state, action, done):
    del done  # Unused.
    local_linvel = self.go1.local_linvel(self.mjx_model, data)
    gyro = self.go1.gyro(self.mjx_model, data)
    projected_gravity = self.go1.projected_gravity(self.mjx_model, data)
    joint_torques = self.go1.joint_torques(self.mjx_model, data)
    joint_accelerations = self.go1.joint_accelerations(self.mjx_model, data)
    joint_angles = self.go1.joint_angles(self.mjx_model, data)
    joint_velocities = self.go1.joint_velocities(self.mjx_model, data)
    reward_terms = {
      "tracking_lin_vel": self._reward_tracking_lin_vel(
        state.info["command"], local_linvel
      ),
      "tracking_ang_vel": self._reward_tracking_ang_vel(state.info["command"], gyro),
      "lin_vel_z": self._cost_lin_vel_z(local_linvel),
      "ang_vel_xy": self._cost_ang_vel_xy(gyro),
      "flat_orientation": self._cost_flat_orientation(projected_gravity),
      "torques": self._cost_torques(joint_torques),
      "action_rate": self._cost_action_rate(action, state.info["last_act"]),
      "dof_acc": self._cost_dof_acc(joint_accelerations),
      "pose": self._cost_pose(joint_angles),
      "dof_pos_limits": self._cost_joint_pos_limits(joint_angles),
      "energy": self._cost_energy(joint_velocities, joint_torques),
      "feet_phase": self._reward_feet_phase(data, state.info["phase"]),
      "collision": self._cost_collision(data),
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
    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
      state.info["step"] > 500, self._sample_command(cmd_rng), state.info["command"]
    )
    state.info["step"] = jp.where(
      done | (state.info["step"] > 500), 0, state.info["step"]
    )
    return data

  def should_terminate_episode(self, data: mjx.Data, state: mjx_env.State) -> jax.Array:
    del state  # Unused.
    fall_termination = self.go1.upvector(self.mjx_model, data)[-1] < 0.0
    return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

  def _sample_command(self, rng: jax.Array) -> jax.Array:
    cfg: CommandConfig = self.cfg.command_config
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    lin_vel_x = jax.random.uniform(rng1, minval=cfg.min[0], maxval=cfg.max[0])
    lin_vel_y = jax.random.uniform(rng2, minval=cfg.min[1], maxval=cfg.max[1])
    ang_vel_yaw = jax.random.uniform(rng3, minval=cfg.min[2], maxval=cfg.max[2])
    return jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])

  # Reward functions.

  def _reward_tracking_lin_vel(
    self, commands: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
    self,
    commands: jax.Array,
    ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

  def _cost_lin_vel_z(self, local_vel) -> jax.Array:
    return jp.square(local_vel[2])

  def _cost_ang_vel_xy(self, ang_vel) -> jax.Array:
    return jp.sum(jp.square(ang_vel[:2]))

  def _cost_flat_orientation(self, gravity: jax.Array) -> jax.Array:
    return jp.sum(jp.square(gravity[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(self, qvel: jax.Array, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(torques))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.sum(jp.square(qpos - self._default_pose) * weight)

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    coll = []
    for geom1_id, geom2_id in self._collision_pairs:
      coll.append(geoms_colliding(data, geom1_id, geom2_id))
    return jp.any(jp.stack(coll))

  def _reward_feet_phase(self, data: mjx.Data, phase: jax.Array) -> jax.Array:
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    rz = get_rz(phase, swing_height=self.cfg.reward_config.max_foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    return jp.exp(-error / 0.005)

  # Visualization.

  def visualize(self, state: mjx_env.State, scn):
    torso_pos = np.asarray(state.data.xpos[self._torso_body_id])
    torso_rot = np.asarray(state.data.xmat[self._torso_body_id])

    def local_to_world(vec: np.ndarray) -> np.ndarray:
      return torso_pos + torso_rot @ vec

    def make_arrow(
      from_local: np.ndarray, to_local: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
      return local_to_world(from_local), local_to_world(to_local)

    def add_arrow(from_w, to_w, rgba, width=0.015, size=(0.005, 0.02, 0.02)):
      scn.ngeom += 1
      geom = scn.geoms[scn.ngeom - 1]
      geom.category = mujoco.mjtCatBit.mjCAT_DECOR

      mujoco.mjv_initGeom(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
        size=np.array(size, dtype=np.float32),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=np.asarray(rgba, dtype=np.float32),
      )

      mujoco.mjv_connector(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
        width=width,
        from_=from_w,
        to=to_w,
      )

    scale = 0.75  # Scale the arrows to 75% of their original size.
    z_offset = 0.2  # Leave a 20cm gap between the base and the arrows.

    # Commanded velocities.
    cmd = state.info["command"]
    cmd_lin_from = np.array([0, 0, z_offset]) * scale
    cmd_lin_to = cmd_lin_from + np.array([cmd[0], cmd[1], 0]) * scale
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = cmd_ang_from + np.array([0, 0, cmd[2]]) * scale
    add_arrow(*make_arrow(cmd_lin_from, cmd_lin_to), rgba=[0.2, 0.2, 0.6, 0.6])
    add_arrow(*make_arrow(cmd_ang_from, cmd_ang_to), rgba=[0.2, 0.6, 0.2, 0.6])

    # Actual velocities.
    linvel = self.go1.local_linvel(self.mjx_model, state.data)
    gyro = self.go1.gyro(self.mjx_model, state.data)
    act_lin_from = np.array([0, 0, z_offset]) * scale
    act_lin_to = act_lin_from + np.array([linvel[0], linvel[1], 0]) * scale
    act_ang_from = act_lin_from
    act_ang_to = act_ang_from + np.array([0, 0, gyro[2]]) * scale
    add_arrow(*make_arrow(act_lin_from, act_lin_to), rgba=[0.0, 0.6, 1.0, 0.7])
    add_arrow(*make_arrow(act_ang_from, act_ang_to), rgba=[0.0, 1.0, 0.4, 0.7])


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: Go1JoystickConfig):
    root, _ = Go1JoystickEnv.build_scene(config=cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(Go1JoystickConfig))
