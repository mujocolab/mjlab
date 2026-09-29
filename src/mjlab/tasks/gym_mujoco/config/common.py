"""Reusable observation and reset terms for the benchmark configurations."""

from copy import deepcopy

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.gym_mujoco import mdp


def actor_critic_observations(
  actor_terms: dict[str, ObservationTermCfg],
  critic_terms: dict[str, ObservationTermCfg] | None = None,
) -> dict[str, ObservationGroupCfg]:
  """Build independent groups, optionally with privileged critic terms."""
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=False
    ),
    "critic": ObservationGroupCfg(
      terms=deepcopy(actor_terms if critic_terms is None else critic_terms),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def joint_observation_terms() -> dict[str, ObservationTermCfg]:
  """Observe all articulation joints in position-then-velocity order."""
  return {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
  }


def floating_observation_terms() -> dict[str, ObservationTermCfg]:
  """Observe a free root and its joints in benchmark order."""
  return {
    "root_height": ObservationTermCfg(func=mdp.root_height),
    "root_orientation": ObservationTermCfg(func=mdp.root_orientation),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "root_lin_vel": ObservationTermCfg(func=mdp.root_lin_vel_w),
    "root_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
  }


def uniform_joint_reset(position_noise: float, velocity_noise: float) -> EventTermCfg:
  return EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-position_noise, position_noise),
      "velocity_range": (-velocity_noise, velocity_noise),
    },
  )


def normal_velocity_joint_reset(
  position_noise: float, velocity_noise: float
) -> EventTermCfg:
  return EventTermCfg(
    func=mdp.reset_joints_with_normal_velocity,
    mode="reset",
    params={"position_noise": position_noise, "velocity_noise": velocity_noise},
  )


def floating_root_reset(position_noise: float, velocity_noise: float) -> EventTermCfg:
  return EventTermCfg(
    func=envs_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        key: (-position_noise, position_noise)
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      },
      "velocity_range": {
        key: (-velocity_noise, velocity_noise)
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      },
    },
  )
