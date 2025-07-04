import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, cast

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
import numpy as np

from mjlab.core import entity, mjx_env, mjx_task
from mjlab.entities.g1 import UnitreeG1, get_assets, G1_XML
from mjlab.entities.g1 import g1_constants as consts
from mjlab.entities.robot_config import CollisionPair
from mjlab.entities.arenas import FlatTerrainArena
from mjlab.envs.motion_imitation import ReferenceMotion
from mjlab import PROCESSED_DATA_DIR
from mjlab.utils import reset as reset_utils
from mjlab.utils.collision import geoms_colliding
from jax.scipy.spatial.transform import Rotation


def quaternion_to_rotation_6d(quat):
  """Convert a batch of quaternions to a 6D rotation representation."""
  mat = jax.vmap(mjx_math.quat_to_mat)(quat)
  return jp.concatenate([mat[..., 0], mat[..., 1]], axis=-1)


def mat_to_rotation_6d(mat):
  """Convert a batch of rotation matrices to a 6D rotation representation."""
  return jp.concatenate([mat[..., 0], mat[..., 1]], axis=-1)


def quaternion_to_angle(quat_actual, quat_desired):
  """Compute the angle of the relative rotation from `quat_actual` to `quat_desired`."""
  q = mjx_math.quat_mul(quat_desired, mjx_math.quat_inv(quat_actual))
  _, angle = mjx_math.quat_to_axis_angle(q)
  return angle


def so3_derivative(qfrom: jax.Array, qto: jax.Array, dt: float) -> jax.Array:
  """Compute the angular velocity that would rotate `qfrom` to `qto` in `dt` seconds,
  assuming a constant angular velocity."""
  q = mjx_math.quat_mul(qto, mjx_math.quat_inv(qfrom))
  axis, angle = mjx_math.quat_to_axis_angle(q)
  return axis * angle / dt


class G1(UnitreeG1):
  def post_init(self):
    for keyframe in consts.KEYFRAMES:
      self.add_keyframe(keyframe, ctrl=keyframe.joint_angles)

    for collision_pair in consts.SELF_COLLISIONS:
      self.add_collision_pair(collision_pair)

    for collision_pair in consts.FLOOR_COLLISIONS:
      self.add_collision_pair(collision_pair)

    foot_floor_pairs = []
    for side in ["left", "right"]:
      for i in range(1, 4):
        pair = CollisionPair(
          geom1=f"{side}_foot{i}_collision",
          geom2="floor",
          condim=3,
          friction=(0.6, 0.6),
        )
        self.add_collision_pair(pair)
        foot_floor_pairs.append(pair.full_name())
    self._foot_floor_pairs = tuple(foot_floor_pairs)

    super().post_init()

    self._joint_stiffness = tuple([a.gainprm[0] for a in self._actuators])
    self._joint_damping = tuple([-a.biasprm[2] for a in self._actuators])
    self._default_joint_pos_nominal = self.spec.key("knees_bent").ctrl

  @property
  def joint_stiffness(self) -> Tuple[float, ...]:
    return self._joint_stiffness

  @property
  def joint_damping(self) -> Tuple[float, ...]:
    return self._joint_damping

  @property
  def default_joint_pos_nominal(self) -> Tuple[float, ...]:
    return tuple(self._default_joint_pos_nominal.tolist())

  @property
  def foot_floor_pairs(self) -> Tuple[str, ...]:
    return self._foot_floor_pairs


@dataclass(frozen=True)
class RewardScales:
  # Pos and rot.
  track_root_pos: float = 1.0
  track_eef_pos: float = 1.0
  track_body_pos: float = 1.0
  track_root_rot: float = 1.0
  track_eef_rot: float = 1.0
  track_body_rot: float = 1.0
  track_joint_pos: float = 0.0
  # Linvels and angvels.
  track_root_vel: float = 0.0
  track_eef_vel: float = 1.0
  track_body_vel: float = 1.0
  track_root_angvel: float = 0.0
  track_eef_angvel: float = 1.0
  track_body_angvel: float = 1.0
  track_joint_vel: float = 0.0
  # COM.
  track_com_pos: float = 0.0
  # Other.
  energy: float = 0.0
  action_rate: float = -1e-2
  termination: float = 0.0
  alive: float = 0.0
  dof_pos_limits: float = -1.0
  collision: float = -1.0


@dataclass(frozen=True)
class RewardConfig:
  scales: RewardScales = RewardScales()
  # Pos.
  joint_pos_std: float = 0.5
  root_pos_std: float = 0.3
  eef_pos_std: float = 0.1
  body_pos_std: float = 0.3
  # Rot.
  root_rot_std: float = 0.4
  eef_rot_std: float = 0.1
  body_rot_std: float = 0.4
  # Linvels.
  root_vel_std: float = 1.0
  eef_vel_std: float = 0.5
  body_vel_std: float = 0.5
  # Angvels.
  joint_vel_std: float = 2.0
  root_angvel_std: float = np.pi
  eef_angvel_std: float = np.pi
  body_angvel_std: float = np.pi
  # COM.
  com_pos_std: float = 1.0


@dataclass(frozen=True)
class NoiseConfig:
  joint_pos: float = 0.03
  joint_vel: float = 1.5
  gyro: float = 0.2
  projected_gravity: float = 0.05
  linvel: float = 0.1
  root_pos: float = 0.05
  root_quat: float = 0.05


@dataclass(frozen=True)
class G1MotionTrackingConfig(mjx_task.TaskConfig):
  """G1 configuration."""

  motion_name: str = "dance1_subject2_50hz.npz"
  sim_dt: float = 0.004
  ctrl_dt: float = 0.02
  solver_iterations: int = 5
  solver_ls_iterations: int = 8
  integrator: str = "implicitfast"
  euler_damping: bool = False
  noise_level: float = 1.0
  action_scale: float = 0.5
  friction_cone: str = "pyramidal"
  max_episode_length: int = 2_000
  soft_joint_pos_limit_factor: float = 0.9
  reward_config: RewardConfig = RewardConfig()
  noise_config: NoiseConfig = NoiseConfig()
  root_pos_fail_threshold: float = 0.5
  root_rot_fail_threshold: float = 0.8
  random_start: bool = True
  target_horizon_steps: Tuple[int, ...] = (1,)


class MotionTrackingEnv(mjx_task.MjxTask[G1MotionTrackingConfig]):
  def __init__(self, config: G1MotionTrackingConfig = G1MotionTrackingConfig()):
    root, entities = MotionTrackingEnv.build_scene(config)
    super().__init__(config, root.spec, entities=entities)
    self.g1: G1 = cast(G1, entities["g1"])
    self._reward_scales = asdict(self.cfg.reward_config.scales)
    self._ref = ReferenceMotion.from_npz(PROCESSED_DATA_DIR / self.cfg.motion_name)

    self._body_rootid = jp.array(self.model.body_rootid)
    self._qpos_init = jp.array(self.model.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(self.model.keyframe("knees_bent").qpos[7:])
    self._inv_default_rot = mjx_math.quat_inv(self._default_pose[3:7])
    self._lowers, self._uppers = self.model.jnt_range[1:].T  # Skip root joint.
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._actuator_torques = self.model.jnt_actfrcrange[1:, 1]

    self._torso_body_id = self.model.body(consts.TORSO_BODY).id
    self._pelvis_body_id = self.model.body(consts.PELVIS_BODY).id
    self._floor_geom_id = self.model.geom("floor").id
    self._left_hand_geom_id = self.model.geom("left_hand_collision").id
    self._right_hand_geom_id = self.model.geom("right_hand_collision").id
    self._left_foot_geom_id = self.model.geom("left_foot_box_collision").id
    self._right_foot_geom_id = self.model.geom("right_foot_box_collision").id
    self._left_shin_geom_id = self.model.geom("left_shin_collision").id
    self._right_shin_geom_id = self.model.geom("right_shin_collision").id
    self._left_thigh_geom_id = self.model.geom("left_thigh_collision").id
    self._right_thigh_geom_id = self.model.geom("right_thigh_collision").id
    self._left_hip_geom_id = self.model.geom("left_hip_collision").id
    self._right_hip_geom_id = self.model.geom("right_hip_collision").id
    self._left_wrist_geom_id = self.model.geom("left_wrist_collision").id
    self._right_wrist_geom_id = self.model.geom("right_wrist_collision").id
    self._torso_geom_id = self.model.geom("torso_collision").id
    self._left_shoulder_geom_id = self.model.geom("left_shoulder_yaw_collision").id
    self._right_shoulder_geom_id = self.model.geom("right_shoulder_yaw_collision").id
    self._left_elbow_geom_id = self.model.geom("left_elbow_yaw_collision").id
    self._right_elbow_geom_id = self.model.geom("right_elbow_yaw_collision").id
    self._left_linkage_brace_geom_id = self.model.geom(
      "left_linkage_brace_collision"
    ).id
    self._right_linkage_brace_geom_id = self.model.geom(
      "right_linkage_brace_collision"
    ).id
    self._end_effector_ids = np.array(
      [self.model.body(name).id for name in consts.END_EFFECTOR_NAMES]
    )
    self._all_body_ids = np.array(
      [self.model.body(name).id for name in consts.BODY_NAMES]
    )
    self._body_ids_minus_root = np.array(
      [self.model.body(name).id for name in consts.BODY_NAMES[1:]]
    )

    # Define unwanted collision pairs.
    # fmt: off
    self._collision_pairs = [
        # Hand - hip.
        (self._left_hand_geom_id, self._left_hip_geom_id),
        (self._right_hand_geom_id, self._right_hip_geom_id),
        # Hand - thigh.
        (self._left_hand_geom_id, self._left_thigh_geom_id),
        (self._right_hand_geom_id, self._right_thigh_geom_id),
        # Foot - foot.
        (self._left_foot_geom_id, self._right_foot_geom_id),
        # Foot - shin.
        (self._left_foot_geom_id, self._right_shin_geom_id),
        (self._right_foot_geom_id, self._left_shin_geom_id),
        # Foot - linkage brace.
        (self._left_foot_geom_id, self._right_linkage_brace_geom_id),
        (self._right_foot_geom_id, self._left_linkage_brace_geom_id),
        # Shin - shin.
        (self._left_shin_geom_id, self._right_shin_geom_id),
        # Torso - shoulder.
        (self._torso_geom_id, self._left_shoulder_geom_id),
        (self._torso_geom_id, self._right_shoulder_geom_id),
        # Torso - elbow.
        (self._torso_geom_id, self._left_elbow_geom_id),
        (self._torso_geom_id, self._right_elbow_geom_id),
        # Torso - wrist.
        (self._torso_geom_id, self._left_wrist_geom_id),
        (self._torso_geom_id, self._right_wrist_geom_id),
        # Torso - hand.
        (self._torso_geom_id, self._left_hand_geom_id),
        (self._torso_geom_id, self._right_hand_geom_id),
        # Hand - hand.
        (self._left_hand_geom_id, self._right_hand_geom_id),
    ]
    # fmt: on

  @property
  def robot(self) -> entity.Entity:
    return self.g1

  @property
  def observation_names(self):
    return (
      "target_qpos",
      "target_qvel",
      "target_torso_pos",
      "projected_gravity",
      "base_linvel",
      "base_angvel",
      "joint_pos",
      "joint_vel",
      "last_act",
    )

  @staticmethod
  def build_scene(
    config: G1MotionTrackingConfig,
  ) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    del config  # Unused.
    g1_entity = G1.from_file(G1_XML, assets=get_assets())

    arena = FlatTerrainArena()
    arena.add_skybox()

    arena.spec.stat.meansize = 0.03

    frame = arena.spec.worldbody.add_frame()
    arena.spec.attach(g1_entity.spec, prefix="", frame=frame)

    return arena, {"g1": g1_entity, "arena": arena}

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    joint_qpos_ids = jp.array([model.bind(j).qposadr for j in self.g1.joints])
    collision_pair_ids = jp.array(
      [self.spec.pair(p).id for p in self.g1.foot_floor_pairs]
    )

    @jax.vmap
    def _randomize(rng):
      # Floor friction: *U(0.4, 1.0).
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=0.6, maxval=1.0)
      pair_friction = model.pair_friction.at[collision_pair_ids, 0:2].set(friction)

      # Joint calibration offsets: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      dqpos = jax.random.uniform(
        key, shape=(len(joint_qpos_ids),), minval=-0.05, maxval=0.05
      )
      qpos0 = model.qpos0.at[joint_qpos_ids].set(model.qpos0[joint_qpos_ids] + dqpos)

      return pair_friction, qpos0

    pair_friction, qpos0 = _randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
      {
        "pair_friction": 0,
        "qpos0": 0,
      }
    )
    model = model.tree_replace(
      {
        "pair_friction": pair_friction,
        "qpos0": qpos0,
      }
    )
    return model, in_axes

  def initialize_episode(self, data: mjx.Data, rng: jax.Array):
    rng, rng_init = jax.random.split(rng)
    random_step_index = jax.random.randint(rng_init, (), 0, len(self._ref))
    step_index = jp.where(self.cfg.random_start, random_step_index, 0)
    ref = self._ref.get(step_index)
    ref_qpos = ref.qpos
    ref_qvel = ref.qvel
    data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
    rng, key = jax.random.split(rng)
    data = reset_utils.reset_joints_by_noise_add(
      key,
      robot=self.g1,
      model=self.mjx_model,
      data=data,
      position_range=(-0.1, 0.1),
    )
    rng, key = jax.random.split(rng)
    data = reset_utils.reset_root_state(
      key,
      robot=self.g1,
      model=self.mjx_model,
      data=data,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range={
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (-0.05, 0.05),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.1, 0.1),
      },
    )
    info = {
      "rng": rng,
      "step_index": step_index,
      "last_act": jp.zeros(self.mjx_model.nu),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_scales.keys()}
    metrics["metrics/root_pos_error"] = jp.zeros(())
    metrics["metrics/root_rot_error"] = jp.zeros(())
    metrics["metrics/eef_pos_error"] = jp.zeros(())
    metrics["metrics/eef_rot_error"] = jp.zeros(())
    metrics["metrics/body_pos_error"] = jp.zeros(())
    metrics["metrics/body_rot_error"] = jp.zeros(())
    metrics["metrics/joint_pos_error"] = jp.zeros(())
    metrics["metrics/root_vel_error"] = jp.zeros(())
    metrics["metrics/root_angvel_error"] = jp.zeros(())
    metrics["metrics/eef_vel_error"] = jp.zeros(())
    metrics["metrics/eef_angvel_error"] = jp.zeros(())
    metrics["metrics/joint_vel_error"] = jp.zeros(())
    metrics["metrics/body_vel_error"] = jp.zeros(())
    metrics["metrics/body_angvel_error"] = jp.zeros(())
    metrics["metrics/com_pos_error"] = jp.zeros(())
    return data, info, metrics

  def before_step(self, action: jax.Array, state: mjx_env.State) -> mjx.Data:
    motor_targets = self._default_pose + action * self.cfg.action_scale
    state.info["step_index"] = state.info["step_index"] + 1
    return super().before_step(motor_targets, state)

  def _apply_noise(
    self, info: dict[str, Any], value: jax.Array, scale: float
  ) -> jax.Array:
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noise = 2 * jax.random.uniform(noise_rng, shape=value.shape) - 1
    noisy_value = value + noise * self.cfg.noise_level * scale
    return noisy_value

  def _get_goal_obs(
    self,
    state: mjx_env.State,
    torso_pos: jax.Array,
    torso_quat: jax.Array,
  ) -> dict[str, jax.Array]:
    """Get goal observations for target horizon steps."""
    cur_torso_pos = torso_pos
    cur_torso_quat = torso_quat
    cur_torso_quat_inv = mjx_math.quat_inv(cur_torso_quat)
    cur_torso_mat_inv = mjx_math.quat_to_mat(cur_torso_quat).T

    def get_single_goal(horizon_step: int) -> dict[str, jax.Array]:
      next_step = jp.minimum(
        state.info["step_index"] + horizon_step, len(self._ref) - 1
      )
      next_ref = self._ref.get(next_step)

      # Target torso position and orientation in the egocentric frame.
      next_torso_pos = cur_torso_mat_inv @ (
        next_ref.pos[self._torso_body_id] - cur_torso_pos
      )
      next_torso_quat = mjx_math.quat_mul(
        cur_torso_quat_inv, next_ref.rot[self._torso_body_id]
      )

      # Target body positions in the egocentric frame.
      next_body_pos = next_ref.pos[self._body_ids_minus_root]  # (N, 3)
      next_body_quat = next_ref.rot[self._body_ids_minus_root]  # (N, 4)
      rel_pos = next_body_pos - cur_torso_pos
      next_body_pos_ego = jp.einsum("ij,nj->ni", cur_torso_mat_inv, rel_pos)
      quat_mul_batched = jax.vmap(mjx_math.quat_mul, in_axes=(None, 0))
      next_body_quat_ego = quat_mul_batched(cur_torso_quat_inv, next_body_quat)

      # Target joint positions and velocities.
      next_joint_angles = next_ref.joint_pos
      next_joint_velocities = next_ref.joint_vel

      # Target end-effector positions.
      next_eef_pos = next_ref.pos[self._end_effector_ids]

      return {
        "joint_angles": next_joint_angles,
        "joint_velocities": next_joint_velocities,
        "eef_pos": next_eef_pos,
        "torso_pos": next_torso_pos,
        "torso_quat": next_torso_quat,
        "body_pos": next_body_pos_ego,
        "body_quat": next_body_quat_ego,
      }

    goals = jax.vmap(get_single_goal)(jp.array(self.cfg.target_horizon_steps))

    return {
      "target_joint_angles": goals["joint_angles"].reshape(-1),
      "target_joint_velocities": goals["joint_velocities"].reshape(-1),
      "target_torso_pos": goals["torso_pos"].reshape(-1),
      "target_torso_quat": goals["torso_quat"].reshape(-1),
      "target_eef_pos": goals["eef_pos"].reshape(-1),
      "target_body_pos": goals["body_pos"].reshape(-1),
      "target_body_quat": goals["body_quat"].reshape(-1),
    }

  def get_observation(self, data: mjx.Data, state: mjx_env.State):
    cfg: NoiseConfig = self.cfg.noise_config

    # Ground-truth observations.
    joint_pos = self.g1.joint_angles(self.mjx_model, data)
    joint_vel = self.g1.joint_velocities(self.mjx_model, data)
    joint_torques = self.g1.joint_torques(self.mjx_model, data)
    linvel = self.g1.local_linvel(self.mjx_model, data)
    gyro = self.g1.gyro(self.mjx_model, data)
    projected_gravity = data.xmat[self._pelvis_body_id].T @ jp.array([0, 0, -1])
    torso_pos = data.xpos[self._torso_body_id]
    torso_xmat = data.xmat[self._torso_body_id]
    torso_quat = Rotation.from_matrix(torso_xmat).as_quat(scalar_first=True)

    # Goal observations.
    goals = self._get_goal_obs(state, torso_pos, torso_quat)

    # Noisy observations.
    noisy_joint_pos = self._apply_noise(state.info, joint_pos, cfg.joint_pos)
    noisy_joint_vel = self._apply_noise(state.info, joint_vel, cfg.joint_vel)
    noisy_gyro = self._apply_noise(state.info, gyro, cfg.gyro)
    noisy_linvel = self._apply_noise(state.info, linvel, cfg.linvel)
    noisy_projected_gravity = self._apply_noise(
      state.info, projected_gravity, cfg.projected_gravity
    )
    noisy_target_torso_pos = self._apply_noise(
      state.info, goals["target_torso_pos"], cfg.root_pos
    )
    noisy_target_torso_quat = self._apply_noise(
      state.info, goals["target_torso_quat"], cfg.root_quat
    )

    obs = jp.hstack(
      [
        goals["target_joint_angles"],
        goals["target_joint_velocities"],
        noisy_target_torso_pos,
        noisy_target_torso_quat,
        noisy_projected_gravity,
        noisy_linvel,
        noisy_gyro,
        noisy_joint_pos - self._default_pose,
        noisy_joint_vel,
        state.info["last_act"],
      ]
    )
    privileged_obs = jp.hstack(
      [
        goals["target_joint_angles"],
        goals["target_joint_velocities"],
        goals["target_torso_pos"],
        goals["target_torso_quat"],
        projected_gravity,
        linvel,
        gyro,
        joint_pos - self._default_pose,
        joint_vel,
        state.info["last_act"],
        # Extra.
        joint_torques,
        goals["target_body_pos"],
        goals["target_body_quat"],
      ]
    )

    return {
      "state": obs,
      "privileged_state": privileged_obs,
    }

  def get_reward(self, data, state, action, done):
    ref = self._ref.get(state.info["step_index"])
    cfg: RewardConfig = self.cfg.reward_config
    reward_terms = {
      # Positions and rotations.
      "track_root_pos": self._reward_tracking_root_pos(
        data, ref, state.metrics, cfg.root_pos_std
      ),
      "track_root_rot": self._reward_tracking_root_rot(
        data, ref, state.metrics, cfg.root_rot_std
      ),
      "track_eef_pos": self._reward_tracking_eef_pos(
        data, ref, state.metrics, cfg.eef_pos_std
      ),
      "track_eef_rot": self._reward_tracking_eef_rot(
        data, ref, state.metrics, cfg.eef_rot_std
      ),
      "track_body_pos": self._reward_tracking_body_pos(
        data, ref, state.metrics, cfg.body_pos_std
      ),
      "track_body_rot": self._reward_tracking_body_rot(
        data, ref, state.metrics, cfg.body_rot_std
      ),
      "track_joint_pos": self._reward_tracking_joint_pos(
        data, ref, state.metrics, cfg.joint_pos_std
      ),
      # Linear and angular velocities.
      "track_root_vel": self._reward_tracking_root_vel(
        data, ref, state.metrics, cfg.root_vel_std
      ),
      "track_root_angvel": self._reward_tracking_root_angvel(
        data, ref, state.metrics, cfg.root_angvel_std
      ),
      "track_eef_vel": self._reward_tracking_eef_vel(
        data, ref, state.metrics, cfg.eef_vel_std
      ),
      "track_eef_angvel": self._reward_tracking_eef_angvel(
        data, ref, state.metrics, cfg.eef_angvel_std
      ),
      "track_joint_vel": self._reward_tracking_joint_vel(
        data, ref, state.metrics, cfg.joint_vel_std
      ),
      "track_body_vel": self._reward_tracking_body_vel(
        data, ref, state.metrics, cfg.body_vel_std
      ),
      "track_body_angvel": self._reward_tracking_body_angvel(
        data, ref, state.metrics, cfg.body_angvel_std
      ),
      # COM.
      "track_com_pos": self._reward_tracking_com_pos(
        data, ref, state.metrics, cfg.com_pos_std
      ),
      # Other.
      "termination": done,
      "alive": jp.array(1.0) - done,
      "energy": self._cost_energy(data.qvel, data.actuator_force),
      "action_rate": self._cost_action_rate(action, state.info["last_act"]),
      "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
      "collision": self._cost_collision(data),
    }
    rewards = {k: v * self._reward_scales[k] for k, v in reward_terms.items()}
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v / self.dt
    reward = sum(rewards.values()) * self.dt
    return reward

  def _exp_rew(self, error: jax.Array, std: jax.Array) -> jax.Array:
    return jp.exp(-error / (std**2))

  def _reward_tracking_root_pos(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the root position in the world frame."""
    root_pos_diff = ref.root_pos - data.qpos[:3]
    root_pos_err = root_pos_diff @ root_pos_diff
    metrics["metrics/root_pos_error"] = jp.sqrt(root_pos_err)
    return self._exp_rew(root_pos_err, std)

  def _reward_tracking_root_rot(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the root rotation in the world frame."""
    root_rot_diff_angle = quaternion_to_angle(data.qpos[3:7], ref.root_quat)
    root_rot_err = root_rot_diff_angle * root_rot_diff_angle
    metrics["metrics/root_rot_error"] = root_rot_err
    return self._exp_rew(root_rot_err, std)

  def _reward_tracking_eef_pos(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the end-effector positions in the world frame."""
    eef_pos = data.xpos[self._end_effector_ids]
    ref_eef_pos = ref.pos[self._end_effector_ids]
    eef_pos_diff = ref_eef_pos - eef_pos
    eef_pos_err = jp.sum(jp.square(eef_pos_diff), axis=-1).mean()
    metrics["metrics/eef_pos_error"] = jp.sqrt(eef_pos_err)
    return self._exp_rew(eef_pos_err, std)

  def _reward_tracking_eef_rot(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the end-effector rotations in the world frame."""
    eef_rot_6d = mat_to_rotation_6d(data.xmat[self._end_effector_ids])
    ref_eef_rot_6d = quaternion_to_rotation_6d(ref.rot[self._end_effector_ids])
    eef_rot_diff = eef_rot_6d - ref_eef_rot_6d
    eef_rot_err = jp.sum(jp.square(eef_rot_diff), axis=-1).mean()
    metrics["metrics/eef_rot_error"] = eef_rot_err
    return self._exp_rew(eef_rot_err, std)

  def _reward_tracking_body_pos(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the body positions in the world frame."""
    body_pos = data.xpos[self._body_ids_minus_root]
    ref_body_pos = ref.pos[self._body_ids_minus_root]
    body_pos_diff = ref_body_pos - body_pos
    body_pos_err = jp.sum(jp.square(body_pos_diff), axis=-1).mean()
    metrics["metrics/body_pos_error"] = jp.sqrt(body_pos_err)
    return self._exp_rew(body_pos_err, std)

  def _reward_tracking_body_rot(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the body rotations in the world frame."""
    body_rot_6d = mat_to_rotation_6d(data.xmat[self._body_ids_minus_root])
    ref_body_rot_6d = quaternion_to_rotation_6d(ref.rot[self._body_ids_minus_root])
    body_rot_diff = body_rot_6d - ref_body_rot_6d
    body_rot_err = jp.sum(jp.square(body_rot_diff), axis=-1).mean()
    metrics["metrics/body_rot_error"] = body_rot_err
    return self._exp_rew(body_rot_err, std)

  def _reward_tracking_joint_pos(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the joint positions."""
    diff = ref.joint_pos - data.qpos[7:]
    error = diff @ diff
    metrics["metrics/joint_pos_error"] = jp.sqrt(error)
    return self._exp_rew(error, std)

  def _reward_tracking_root_vel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the root linear velocity in the world frame."""
    root_vel_diff = ref.root_vel - data.qvel[:3]
    root_vel_err = root_vel_diff @ root_vel_diff
    metrics["metrics/root_vel_error"] = jp.sqrt(root_vel_err)
    return self._exp_rew(root_vel_err, std)

  def _reward_tracking_root_angvel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the root angular velocity in the world frame."""
    root_ang_vel_diff = ref.root_angvel - data.qvel[3:6]
    root_ang_vel_err = root_ang_vel_diff @ root_ang_vel_diff
    metrics["metrics/root_angvel_error"] = jp.sqrt(root_ang_vel_err)
    return self._exp_rew(root_ang_vel_err, std)

  def _reward_tracking_joint_vel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the joint velocities."""
    diff = ref.joint_vel - data.qvel[6:]
    error = diff @ diff
    metrics["metrics/joint_vel_error"] = jp.sqrt(error)
    return self._exp_rew(error, std)

  def _reward_tracking_eef_vel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the end-effector linear velocities in the world frame."""
    # Convert COM-frame velocities to world-frame velocities.
    com_linvel = data.cvel[:, 3:6]
    offset = data.xpos - data.subtree_com[self._body_rootid]
    all_body_vel = com_linvel - jp.cross(offset, com_linvel)
    eef_vel = all_body_vel[self._end_effector_ids]
    ref_eef_vel = ref.linvel[self._end_effector_ids]
    eef_vel_diff = ref_eef_vel - eef_vel
    eef_vel_err = jp.sum(jp.square(eef_vel_diff), axis=-1).mean()
    metrics["metrics/eef_vel_error"] = jp.sqrt(eef_vel_err)
    return self._exp_rew(eef_vel_err, std)

  def _reward_tracking_eef_angvel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the end-effector angular velocities in the world frame."""
    # Com-frame angular velocities are also world-frame angular velocities.
    eef_angvel = data.cvel[self._end_effector_ids, 0:3]
    ref_eef_angvel = ref.angvel[self._end_effector_ids]
    eef_angvel_diff = ref_eef_angvel - eef_angvel
    eef_angvel_err = jp.sum(jp.square(eef_angvel_diff), axis=-1).mean()
    metrics["metrics/eef_angvel_error"] = jp.sqrt(eef_angvel_err)
    return self._exp_rew(eef_angvel_err, std)

  def _reward_tracking_body_vel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the body linear velocities in the world frame."""
    # Convert COM-frame velocities to world-frame velocities.
    com_linvel = data.cvel[:, 3:6]
    offset = data.xpos - data.subtree_com[self._body_rootid]
    all_body_vel = com_linvel - jp.cross(offset, com_linvel)
    body_vel = all_body_vel[self._body_ids_minus_root]
    ref_body_vel = ref.linvel[self._body_ids_minus_root]
    body_vel_diff = ref_body_vel - body_vel
    body_vel_err = jp.sum(jp.square(body_vel_diff), axis=-1).mean()
    metrics["metrics/body_vel_error"] = jp.sqrt(body_vel_err)
    return self._exp_rew(body_vel_err, std)

  def _reward_tracking_body_angvel(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the body angular velocities in the world frame."""
    # Com-frame angular velocities are also world-frame angular velocities.
    body_angvel = data.cvel[self._body_ids_minus_root, 0:3]
    ref_body_angvel = ref.angvel[self._body_ids_minus_root]
    body_angvel_diff = ref_body_angvel - body_angvel
    body_angvel_err = jp.sum(jp.square(body_angvel_diff), axis=-1).mean()
    metrics["metrics/body_angvel_error"] = jp.sqrt(body_angvel_err)
    return self._exp_rew(body_angvel_err, std)

  def _reward_tracking_com_pos(
    self,
    data: mjx.Data,
    ref: ReferenceMotion,
    metrics: dict[str, Any],
    std: jax.Array,
  ) -> jax.Array:
    """Reward for tracking the center of mass position."""
    com_pos_diff = ref.com - data.subtree_com[0]
    com_pos_err = com_pos_diff @ com_pos_diff
    metrics["metrics/com_pos_error"] = jp.sqrt(com_pos_err)
    return self._exp_rew(com_pos_err, std)

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    """Cost for the action rate."""
    return jp.sum(jp.square(act - last_act))

  def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    """Cost for the energy."""
    # NOTE(kevin): Scale torques proportionately to their limits.
    torques = qfrc_actuator / self._actuator_torques
    return jp.sum(jp.abs(qvel[6:] * torques))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    """Cost for crossing joint limits."""
    lower_violation = jp.maximum(0.0, self._soft_lowers - qpos)
    upper_violation = jp.maximum(0.0, qpos - self._soft_uppers)
    return jp.sum(lower_violation + upper_violation)

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    """Cost for unwanted collisions."""
    # Incur a cost if any of the collision pairs are colliding. Cost is 0 only if
    # all unwanted collision pairs are not colliding.
    collision_results = []
    for geom1, geom2 in self._collision_pairs:
      collision_results.append(geoms_colliding(data, geom1, geom2))
    return jp.any(jp.stack(collision_results))

  def after_step(
    self,
    data: mjx.Data,
    state: mjx_env.State,
    action: jax.Array,
    done: jax.Array,
  ) -> mjx.Data:
    del action, done  # Unused.
    # If we've reached the end of the motion, reset to a randomly sampled state within
    # the reference motion. This is called "reference state initialization" in the
    # literature.
    step_index = state.info["step_index"]
    end_of_motion = step_index >= len(self._ref)
    state.info["rng"], rng = jax.random.split(state.info["rng"])
    new_step = jax.random.randint(rng, (), 0, len(self._ref))
    new_ref = self._ref.get(new_step)
    qpos = jp.where(end_of_motion, new_ref.qpos, data.qpos)
    qvel = jp.where(end_of_motion, new_ref.qvel, data.qvel)
    data = data.replace(qpos=qpos, qvel=qvel)
    state.info["step_index"] = jp.where(end_of_motion, new_step, step_index)
    return data

  def should_terminate_episode(self, data: mjx.Data, state: mjx_env.State) -> jax.Array:
    ref = self._ref.get(state.info["step_index"])

    # Root position termination.
    root_pos_diff = ref.root_pos - data.qpos[:3]
    root_pos_error = root_pos_diff @ root_pos_diff
    root_pos_fail = root_pos_error > self.cfg.root_pos_fail_threshold**2

    # Root rotation termination.
    proj_gravity = mjx_math.quat_to_mat(data.qpos[3:7]).T @ jp.array([0, 0, -1])
    proj_gravity_ref = mjx_math.quat_to_mat(ref.root_quat).T @ jp.array([0, 0, -1])
    root_rot_diff_angle = proj_gravity[2] - proj_gravity_ref[2]
    root_rot_fail = jp.abs(root_rot_diff_angle) > self.cfg.root_rot_fail_threshold

    # NaN termination.
    nan_fail = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    state.metrics["termination/root_pos"] = root_pos_fail.astype(jp.float32)
    state.metrics["termination/root_rot"] = root_rot_fail.astype(jp.float32)
    return root_pos_fail | root_rot_fail | nan_fail

  # Visualization.

  def visualize(self, state: mjx_env.State, scn):
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    model: mujoco.MjModel = copy.deepcopy(self.model)
    model.geom_rgba[:, 1] = np.clip(model.geom_rgba[:, 1] * 1.5, 0.0, 1.0)

    data = mujoco.MjData(self.model)
    data.qpos, data.qvel = state.data.qpos, state.data.qvel
    data.mocap_pos, data.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
    data.xfrc_applied = state.data.xfrc_applied
    mujoco.mj_forward(self.model, data)
    ref = self._ref.get(state.info["step_index"])
    data.qpos = np.array(ref.qpos)
    data.qpos[1] += 0.6
    mujoco.mj_forward(self.model, data)

    mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, scn)


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: G1MotionTrackingConfig):
    root, _ = MotionTrackingEnv.build_scene(config=cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(G1MotionTrackingConfig))
