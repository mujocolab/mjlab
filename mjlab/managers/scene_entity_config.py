from dataclasses import dataclass, field
from typing import Sequence

import mujoco
from mjlab.utils.string import filter_exp


def dof_width(joint_type: mujoco.mjtJoint) -> int:
  """Get the dimensionality of the joint in qvel."""
  return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: mujoco.mjtJoint) -> int:
  """Get the dimensionality of the joint in qpos."""
  return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


@dataclass
class SceneEntityCfg:
  name: str
  joint_names: Sequence[str] = ()
  body_names: Sequence[str] = ()
  site_names: Sequence[str] = ()
  preserve_order: bool = False

  joint_ids: list[int] = field(default_factory=list)
  qpos_ids: list[int] = field(default_factory=list)
  dof_ids: list[int] = field(default_factory=list)
  body_ids: list[int] = field(default_factory=list)
  actuator_ids: list[int] = field(default_factory=list)
  site_ids: list[int] = field(default_factory=list)

  def resolve(self, model: mujoco.MjModel) -> None:
    self._resolve_joint_names(model)
    self._resolve_body_names(model)
    self._resolve_site_names(model)

  def _resolve_joint_names(self, model: mujoco.MjModel) -> None:
    # Extract joint names scoped to this entity.
    all_joint_names = [model.joint(i).name for i in range(model.njnt)]
    scoped_joint_names = [
      name for name in all_joint_names if name.startswith(f"{self.name}/")
    ]

    # Resolve joint names based on user input: default to all scoped joints,
    # apply filter patterns (e.g., regex), or map unscoped exact names to full names.
    if not self.joint_names:
      resolved_joint_names = scoped_joint_names
    elif any(name.startswith(".") for name in self.joint_names):
      resolved_joint_names = filter_exp(self.joint_names, scoped_joint_names)
    else:
      full_names = [
        name if name.startswith(f"{self.name}/") else f"{self.name}/{name}"
        for name in self.joint_names
      ]
      resolved_joint_names = [name for name in full_names if name in scoped_joint_names]

    self.joint_names = tuple(resolved_joint_names)

    # Resolve IDs for each joint.
    for name in self.joint_names:
      joint = model.joint(name)
      qpos_start = joint.qposadr[0]
      dof_start = joint.dofadr[0]
      self.qpos_ids.extend(range(qpos_start, qpos_start + qpos_width(joint.type[0])))
      self.dof_ids.extend(range(dof_start, dof_start + dof_width(joint.type[0])))
      self.joint_ids.append(joint.id)

    # Resolve actuators acting on each joint.
    for i in range(model.nu):
      act = model.actuator(i)
      if act.trntype[0] != mujoco.mjtTrn.mjTRN_JOINT:
        continue
      if act.trnid[0] in self.joint_ids:
        self.actuator_ids.append(i)

  def _resolve_body_names(self, model: mujoco.MjModel) -> None:
    # Extract body names scoped to this entity (skip world body).
    all_body_names = [model.body(i).name for i in range(1, model.nbody)]
    scoped_body_names = [
      name for name in all_body_names if name.startswith(f"{self.name}/")
    ]

    # Resolve body names based on user input: default to all scoped bodies,
    # apply filter patterns (e.g., regex), or map unscoped exact names to full names
    if not self.body_names:
      resolved_body_names = scoped_body_names
    elif any(name.startswith(".") for name in self.body_names):
      resolved_body_names = filter_exp(self.body_names, scoped_body_names)
    else:
      full_names = [
        name if name.startswith(f"{self.name}/") else f"{self.name}/{name}"
        for name in self.body_names
      ]
      resolved_body_names = [name for name in full_names if name in scoped_body_names]

    self.body_names = tuple(resolved_body_names)

    # Resolve IDs for each body.
    for name in self.body_names:
      self.body_ids.append(model.body(name).id)

  def _resolve_site_names(self, model: mujoco.MjModel) -> None:
    # Extract site names scoped to this entity.
    all_site_names = [model.site(i).name for i in range(model.nsite)]
    scoped_site_names = [
      name for name in all_site_names if name.startswith(f"{self.name}/")
    ]

    # Resolve site names based on user input: default to all scoped bodies,
    # apply filter patterns (e.g., regex), or map unscoped exact names to full names
    if not self.site_names:
      resolved_site_names = scoped_site_names
    elif any(name.startswith(".") for name in self.site_names):
      resolved_site_names = filter_exp(self.site_names, scoped_site_names)
    else:
      full_names = [
        name if name.startswith(f"{self.name}/") else f"{self.name}/{name}"
        for name in self.site_names
      ]
      resolved_site_names = [name for name in full_names if name in scoped_site_names]

    self.site_names = tuple(resolved_site_names)

    # Resolve IDs for each site.
    for name in self.site_names:
      self.site_ids.append(model.site(name).id)
