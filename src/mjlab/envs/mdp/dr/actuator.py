"""Domain randomization functions for actuators."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import (
  BuiltinDcMotorActuator,
  BuiltinPdActuator,
  BuiltinPositionActuator,
  IdealPdActuator,
)
from mjlab.actuator.actuator import Actuator, TransmissionType
from mjlab.actuator.builtin_actuator import DcMotorInputMode
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG
from ._types import Operation, resolve_distribution, resolve_operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _selected_actuators(
  asset: Entity, asset_cfg: SceneEntityCfg
) -> Iterator[tuple[Actuator, torch.Tensor]]:
  """Yield controller groups and their selected group-local control rows."""
  selection = asset_cfg.actuator_ids
  selected_ids = None
  if selection != slice(None):
    # SceneEntityCfg IDs index physical actuators, not controller groups.
    selected_ids = torch.arange(
      asset.num_actuators, device=asset.indexing.ctrl_ids.device
    )[selection]
  for actuator in asset.actuators:
    rows = torch.arange(len(actuator.ctrl_ids), device=actuator.ctrl_ids.device)
    if selected_ids is not None:
      rows = rows[torch.isin(actuator.ctrl_ids, selected_ids)]
    if rows.numel():
      yield actuator, rows


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Operation | str = "scale",
) -> None:
  """Randomize PD stiffness and damping gains.

  For BuiltinPdActuator, selected position rows receive kp and selected
  velocity rows receive kd. Unselected rows keep their gains.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    kp_range: (min, max) for proportional gain randomization.
    kd_range: (min, max) for derivative gain randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default gains by sampled values, "abs" sets
      absolute values.
  """
  op = resolve_operation(operation)
  if op.name not in ("scale", "abs"):
    raise ValueError(
      f"pd_gains only supports 'scale' and 'abs' operations, got {op.name!r}"
    )
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  for actuator, rows in _selected_actuators(asset, asset_cfg):
    ctrl_ids = actuator.global_ctrl_ids[rows]
    sample_ids = None
    if isinstance(actuator, BuiltinPdActuator):
      # Paired rows share a target, even when sorting interleaves their roles.
      targets, sample_ids = torch.unique(
        actuator.ctrl_target_ids[rows], return_inverse=True
      )
      n_gains = len(targets)
    else:
      n_gains = len(rows)

    dist = resolve_distribution(distribution)
    kp_samples = dist.sample(
      torch.tensor(kp_range[0], device=env.device),
      torch.tensor(kp_range[1], device=env.device),
      (len(env_ids), n_gains),
      env.device,
    )
    kd_samples = dist.sample(
      torch.tensor(kd_range[0], device=env.device),
      torch.tensor(kd_range[1], device=env.device),
      (len(env_ids), n_gains),
      env.device,
    )

    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if op.name == "scale":
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * kd_samples
        )
      else:
        assert op.name == "abs"
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples

    elif isinstance(actuator, BuiltinDcMotorActuator):
      if actuator.cfg.mode == DcMotorInputMode.VOLTAGE:
        raise ValueError(
          "dr.pd_gains does not apply to BuiltinDcMotorActuator in VOLTAGE "
          "mode (no internal PID gains to scale)."
        )
      # DC motor stores kp at gainprm[4] and kd at gainprm[6] (set via
      # set_to_dcmotor). The bias slots carry back-EMF / cogging, not the PD,
      # so we only touch gainprm.
      if op.name == "scale":
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 4] = (
          default_gainprm[ctrl_ids, 4] * kp_samples
        )
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 6] = (
          default_gainprm[ctrl_ids, 6] * kd_samples
        )
      else:
        assert op.name == "abs"
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 4] = kp_samples
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 6] = kd_samples

    elif isinstance(actuator, BuiltinPdActuator):
      assert sample_ids is not None
      position_mask = actuator.position_mask[rows]
      pos_ids = ctrl_ids[position_mask]
      vel_ids = ctrl_ids[~position_mask]
      kp_samples = kp_samples[:, sample_ids[position_mask]]
      kd_samples = kd_samples[:, sample_ids[~position_mask]]
      if op.name == "scale":
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], pos_ids, 0] = (
          default_gainprm[pos_ids, 0] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], pos_ids, 1] = (
          default_biasprm[pos_ids, 1] * kp_samples
        )
        env.sim.model.actuator_gainprm[env_ids[:, None], vel_ids, 0] = (
          default_gainprm[vel_ids, 0] * kd_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], vel_ids, 2] = (
          default_biasprm[vel_ids, 2] * kd_samples
        )
      else:
        assert op.name == "abs"
        env.sim.model.actuator_gainprm[env_ids[:, None], pos_ids, 0] = kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], pos_ids, 1] = -kp_samples
        env.sim.model.actuator_gainprm[env_ids[:, None], vel_ids, 0] = kd_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], vel_ids, 2] = -kd_samples
      # biasprm[2] on position rows stays zero by construction. Writing
      # anything else here would inject damping into the position element on
      # top of the velocity element, silently double-counting kd.

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.stiffness is not None
      assert actuator.damping is not None
      if op.name == "scale":
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.set_gains(
          env_ids,
          kp=actuator.default_stiffness[env_ids[:, None], rows] * kp_samples,
          kd=actuator.default_damping[env_ids[:, None], rows] * kd_samples,
          target_ids=rows,
        )
      else:
        assert op.name == "abs"
        actuator.set_gains(env_ids, kp=kp_samples, kd=kd_samples, target_ids=rows)

    else:
      raise TypeError(
        f"pd_gains only supports BuiltinPositionActuator, BuiltinPdActuator, "
        f"BuiltinDcMotorActuator (position/velocity mode), XmlActuator (position), "
        f"and IdealPdActuator, got {type(actuator).__name__}"
      )


@requires_model_fields("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
def effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Operation | str = "scale",
) -> None:
  """Randomize actuator effort limits.

  BuiltinPdActuator limits belong to the joint or tendon. Selecting either
  control row changes that shared limit, including other rows on the same
  target. Each shared limit receives one sample per call.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    effort_limit_range: (min, max) for effort limit randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default limits, "abs" sets absolute values.
  """
  op = resolve_operation(operation)
  if op.name not in ("scale", "abs"):
    raise ValueError(
      f"effort_limits only supports 'scale' and 'abs' operations, got {op.name!r}"
    )
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  seen_targets: dict[str, torch.Tensor] = {}
  for actuator, rows in _selected_actuators(asset, asset_cfg):
    ctrl_ids = actuator.global_ctrl_ids[rows]
    field = None
    target_global_ids = None
    if isinstance(actuator, BuiltinPdActuator):
      targets = torch.unique(actuator.ctrl_target_ids[rows])
      if actuator.transmission_type == TransmissionType.JOINT:
        field = "jnt_actfrcrange"
        target_global_ids = asset.indexing.joint_ids[actuator.target_ids[targets]]
      else:
        field = "tendon_actfrcrange"
        target_global_ids = asset.indexing.tendon_ids[actuator.target_ids[targets]]
      if field in seen_targets:
        target_global_ids = target_global_ids[
          ~torch.isin(target_global_ids, seen_targets[field])
        ]
        seen_targets[field] = torch.cat((seen_targets[field], target_global_ids))
      else:
        seen_targets[field] = target_global_ids
      n_samples = len(target_global_ids)
      if not n_samples:
        continue
    else:
      n_samples = len(rows)

    dist = resolve_distribution(distribution)
    effort_samples = dist.sample(
      torch.tensor(effort_limit_range[0], device=env.device),
      torch.tensor(effort_limit_range[1], device=env.device),
      (len(env_ids), n_samples),
      env.device,
    )

    if isinstance(actuator, (BuiltinPositionActuator, BuiltinDcMotorActuator)) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if op.name == "scale":
        default_forcerange = env.sim.get_default_field("actuator_forcerange")
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_samples
        )
      else:
        assert op.name == "abs"
        env.sim.model.actuator_forcerange[
          env_ids[:, None], ctrl_ids, 0
        ] = -effort_samples
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          effort_samples
        )

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.force_limit is not None
      if op.name == "scale":
        assert actuator.default_force_limit is not None
        actuator.set_effort_limit(
          env_ids,
          effort_limit=actuator.default_force_limit[env_ids[:, None], rows]
          * effort_samples,
          target_ids=rows,
        )
      else:
        assert op.name == "abs"
        actuator.set_effort_limit(env_ids, effort_limit=effort_samples, target_ids=rows)

    elif isinstance(actuator, BuiltinPdActuator):
      # BuiltinPd's effort_limit lives on the joint/tendon as a sum-clamp
      # (jnt_actfrcrange / tendon_actfrcrange), not on per-element forcerange.
      assert field is not None and target_global_ids is not None
      arr = getattr(env.sim.model, field)
      if op.name == "scale":
        default = env.sim.get_default_field(field)
        arr[env_ids[:, None], target_global_ids, 0] = (
          default[target_global_ids, 0] * effort_samples
        )
        arr[env_ids[:, None], target_global_ids, 1] = (
          default[target_global_ids, 1] * effort_samples
        )
      else:
        assert op.name == "abs"
        arr[env_ids[:, None], target_global_ids, 0] = -effort_samples
        arr[env_ids[:, None], target_global_ids, 1] = effort_samples

    else:
      raise TypeError(
        f"effort_limits only supports BuiltinPositionActuator, BuiltinPdActuator, "
        f"BuiltinDcMotorActuator, XmlActuator (position), and IdealPdActuator, "
        f"got {type(actuator).__name__}"
      )
