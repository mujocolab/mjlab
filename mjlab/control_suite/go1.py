from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mjlab._src.collision import geoms_colliding
from mjlab._src import MENAGERIE_PATH
from mjlab._src import entity, mjx_env, mjx_task
from mjlab._src.arenas import FlatTerrainArena

_HERE = Path(__file__).parent
_GO1_XML = _HERE / "go1.xml"


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = MENAGERIE_PATH / "unitree_go1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class Go1(entity.Entity):
  """Go1 entity."""

  def post_init(self) -> None:
    self._torso_body = self.spec.body("trunk")
    self._imu_site = self.spec.site("imu")
    self._joints = self.get_non_root_joints()
    self._actuators = self.spec.actuators
    self._gyro_sensor = self.spec.sensor("gyro")
    self._local_linvel_sensor = self.spec.sensor("local_linvel")
    self._upvector_sensor = self.spec.sensor("upvector")

  def change_pd_gains(self, Kp: float, Kv: float) -> None:
    for act in self._actuators:
      act.gainprm[0] = Kp
      act.biasprm[1] = -Kp
      act.biasprm[2] = 0.0
    for jnt in self._joints:
      jnt.damping = Kv

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._joints

  @property
  def actuators(self) -> Tuple[mujoco.MjsActuator]:
    return self._actuators

  @property
  def nq(self) -> int:
    return self.mjx_model.nq - 7

  @property
  def nv(self) -> int:
    return self.mjx_model.nv - 6

  def joint_angles(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._joints).qpos

  def joint_velocities(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._joints).qvel

  def joint_accelerations(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._joints).qacc

  def joint_torques(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._actuators).force

  def gyro(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._gyro_sensor).sensordata

  def local_linvel(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._local_linvel_sensor).sensordata

  def projected_gravity(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._imu_site).xmat.T @ jp.array([0, 0, -1.0])

  def upvector(self, data: mjx.Data) -> jax.Array:
    return data.bind(self.mjx_model, self._upvector_sensor).sensordata


@dataclass(frozen=True)
class CommandConfig:
  min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
  max: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class RewardScales:
  tracking_lin_vel: float = 1.5
  tracking_ang_vel: float = 0.75
  lin_vel_z: float = -2.0
  ang_vel_xy: float = -0.05
  torques: float = -0.0002
  dof_acc: float = -2.5e-7
  action_rate: float = -0.01
  feet_air_time: float = 0.25
  flat_orientation: float = -2.5
  pose: float = -1.0


@dataclass(frozen=True)
class RewardConfig:
  scales: RewardScales = RewardScales()
  tracking_sigma: float = 0.25


@dataclass
class Go1Config(mjx_task.TaskConfig):
  """Go1 configuration."""

  sim_dt: float = 0.004
  ctrl_dt: float = 0.02
  solver_iterations: int = 1
  solver_ls_iterations: int = 5
  euler_damping: bool = False
  max_episode_length: int = 1_000
  Kp: float = 35
  Kv: float = 0.5
  action_scale: float = 0.25
  command_config: CommandConfig = CommandConfig()
  reward_config: RewardConfig = RewardConfig()


class Go1Env(mjx_task.MjxTask[Go1Config]):
  def __init__(self, config: Go1Config = Go1Config()):
    root, entities = Go1Env.build_scene(config)
    super().__init__(config, root, entities=entities)

    self._reward_scales = asdict(self.cfg.reward_config.scales)
    self._floor_geom_id = self._mj_model.geom("floor").id
    feet_geoms = ["FR", "FL", "RR", "RL"]
    self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in feet_geoms])
    self._feet_site_id = np.array([self._mj_model.site(name).id for name in feet_geoms])
    self._torso_geom_ids = np.array(
      [
        self._mj_model.geom(name).id
        for name in ["trunk_collision1", "trunk_collision2"]
      ]
    )

  @staticmethod
  def build_scene(config: Go1Config) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    assets = get_assets()
    go1_entity = Go1.from_file(_GO1_XML, assets=assets)
    go1_entity.change_pd_gains(config.Kp, config.Kv)

    arena = FlatTerrainArena()
    arena.add_skybox()
    arena.floor_geom.contype = 1
    arena.floor_geom.conaffinity = 0
    arena.floor_geom.priority = 1
    arena.floor_geom.condim = 3
    arena.floor_geom.friction[0] = 0.6

    arena.spec.stat.meansize = 0.04

    frame = arena.spec.worldbody.add_frame()
    arena.spec.attach(go1_entity.spec, prefix="", frame=frame)

    return arena, {"go1": go1_entity, "arena": arena}

  def after_compile(self) -> None:
    super().after_compile()
    self.go1: Go1 = self._entities["go1"]

    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self.mj_model.keyframe("home").qpos[7:])

  def before_step(self, action: jax.Array, state: mjx_env.State) -> mjx.Data:
    motor_targets = self._default_pose + action * self.cfg.action_scale
    return super().before_step(motor_targets, state)

  def initialize_episode(self, data: mjx.Data, rng: jax.Array):
    qpos = self._init_q
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)
    data = data.replace(qpos=qpos)
    rng, cmd_rng = jax.random.split(rng)
    command = self._sample_command(cmd_rng)
    info = {
      "rng": rng,
      "step": 0,
      "last_act": jp.zeros(self.mjx_model.nu),
      "command": command,
      "feet_air_time": jp.zeros(4),
      "last_contact": jp.zeros(4, dtype=bool),
      "first_contact": jp.zeros(4, dtype=bool),
      "contact": jp.zeros(4, dtype=bool),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_scales.keys()}
    return data, info, metrics

  def get_observation(self, data, state):
    return jp.concatenate(
      [
        self.go1.gyro(data),
        self.go1.projected_gravity(data),
        self.go1.joint_angles(data),
        self.go1.joint_velocities(data),
        self.go1.local_linvel(data),
        state.info["last_act"],
        state.info["command"],
      ]
    )

  def get_reward(self, data, state, action, done):
    del done  # Unused.

    contact = jp.array(
      [
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
      ]
    )
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    state.info["first_contact"] = first_contact

    joint_torques = self.go1.joint_torques(data)
    joint_accelerations = self.go1.joint_accelerations(data)
    reward_terms = {
      "tracking_lin_vel": self._reward_tracking_lin_vel(
        state.info["command"], self.go1.local_linvel(data)
      ),
      "tracking_ang_vel": self._reward_tracking_ang_vel(
        state.info["command"], self.go1.gyro(data)
      ),
      "lin_vel_z": self._cost_lin_vel_z(self.go1.local_linvel(data)),
      "ang_vel_xy": self._cost_ang_vel_xy(self.go1.gyro(data)),
      "flat_orientation": self._cost_flat_orientation(self.go1.projected_gravity(data)),
      "torques": self._cost_torques(joint_torques),
      "action_rate": self._cost_action_rate(action, state.info["last_act"]),
      "dof_acc": self._cost_dof_acc(joint_accelerations),
      "feet_air_time": self._reward_feet_air_time(
        state.info["feet_air_time"], first_contact, state.info["command"]
      ),
      "pose": self._reward_pose(self.go1.joint_angles(data)),
    }
    rewards = {k: v * self._reward_scales[k] for k, v in reward_terms.items()}
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v / self.dt

    reward = sum(rewards.values()) * self.dt
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact

    return reward

  def after_step(
    self,
    data: mjx.Data,
    state: mjx_env.State,
    action: jax.Array,
    done: jax.Array,
  ) -> mjx.Data:
    state.info["step"] += 1
    state.info["last_act"] = action
    # Update command.
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
    torso_ground_contact = jp.array(
      [
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._torso_geom_ids
      ]
    )
    return jp.any(torso_ground_contact)

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

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _reward_feet_air_time(
    self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.5) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    error = jp.sum(jp.square(qpos - self._default_pose))
    return jp.exp(-error / 10)


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: Go1Config):
    root, _ = Go1Env.build_scene(config=cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(Go1Config))
