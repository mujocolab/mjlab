"""Shared helpers for syncing per-world model fields into a host MjModel.

The simulator owns one CPU MjModel as a template for MuJoCo host APIs, while
mjlab.sim.Simulation.model may hold per-world arrays after domain randomization
or per-world mesh compilation. Viewers and renderers must copy the target env's
fields into their host model before calling CPU MuJoCo rendering or kinematics
functions.

Used by the native / offscreen / viser viewers.
"""

from __future__ import annotations

from typing import Any

import mujoco

# Inertial fields that shift subtree_com (and thus the tracking camera).
VIEWER_INERTIAL_FIELDS = frozenset({"body_ipos", "body_mass"})

# Fields that affect viewer-side CPU rendering or kinematics. Physics-only
# fields (geom_aabb, geom_rbound, dof_*, jnt_*, actuator_*, tendon_*,
# body_invweight0) are skipped.
VIEWER_MODEL_FIELDS = frozenset(
  {
    "qpos0",  # Needed for correct mj_forward kinematics (qpos - qpos0).
    "geom_dataid",  # Per-world mesh variants.
    "geom_rgba",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "mat_rgba",
    "site_pos",
    "site_quat",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "body_mass",
    # body_subtreemass is the denominator mj_forward uses to normalize
    # subtree_com. Without it, the mjVIS_COM decor geoms (and any code that
    # reads data.subtree_com) land at scaled/arbitrary positions when
    # body_mass varies per-world.
    "body_subtreemass",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "cam_intrinsic",
    "light_pos",
    "light_dir",
  }
)


def disable_model_sameframe_shortcuts(model: mujoco.MjModel) -> None:
  """Force MuJoCo's host model down the full local-to-global transform path.

  mujoco_warp does not implement MuJoCo's *_sameframe shortcuts: its
  kinematics kernels always apply the full transform for inertial frames,
  geoms, and sites. Clearing these compile-time flags on the CPU viewer model
  keeps host-side mj_forward consistent with the GPU path when per-world
  mesh variants change local offsets and frame alignments.
  """
  none = mujoco.mjtSameFrame.mjSAMEFRAME_NONE.value
  model.body_sameframe[:] = none
  model.geom_sameframe[:] = none
  model.site_sameframe[:] = none
  # body_simple is another compile-time shortcut derived from body_sameframe.
  # Clear it too so host-side inertia code does not
  # assume the original compiled alignment.
  model.body_simple[:] = 0


def sync_model_fields(
  dst_model: mujoco.MjModel,
  sim_model: Any,
  fields: set[str] | frozenset[str],
  env_idx: int,
) -> None:
  """Copy per-world model fields from sim_model[env_idx] into dst_model."""
  for field_name in fields:
    src = getattr(sim_model, field_name)[env_idx].cpu().numpy()
    dst = getattr(dst_model, field_name)
    dst[:] = src.reshape(dst.shape)
