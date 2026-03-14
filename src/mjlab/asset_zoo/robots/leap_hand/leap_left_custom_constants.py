"""LEAP Hand (Left Custom) constants and EntityCfg for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.1),
  joint_pos={
    "if_mcp": 0.131,
    "if_rot": 0.0,
    "if_pip": 0.65,
    "if_dip": 1.0,
    "mf_mcp": 0.131,
    "mf_rot": 0.0,
    "mf_pip": 0.65,
    "mf_dip": 1.0,
    "rf_mcp": 0.131,
    "rf_rot": 0.0,
    "rf_pip": 0.65,
    "rf_dip": 1.0,
    "th_cmc": 0.8,
    "th_axl": -0.78,
    "th_mcp": 0.5,
    "th_ipl": 0.367,
  },
  joint_vel={".*": 0.0},
)

LEAP_LEFT_CUSTOM_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "leap_hand"
  / "xmls"
  / "left_hand_custom.xml"
)
assert LEAP_LEFT_CUSTOM_XML.exists(), f"Missing MJCF: {LEAP_LEFT_CUSTOM_XML}"

LEAP_LEFT_CUSTOM_ACTUATION = EntityArticulationInfoCfg(
  actuators=(
    XmlPositionActuatorCfg(target_names_expr=("if_mcp",)),
    XmlPositionActuatorCfg(target_names_expr=("if_rot",)),
    XmlPositionActuatorCfg(target_names_expr=("if_pip",)),
    XmlPositionActuatorCfg(target_names_expr=("if_dip",)),
    XmlPositionActuatorCfg(target_names_expr=("mf_mcp",)),
    XmlPositionActuatorCfg(target_names_expr=("mf_rot",)),
    XmlPositionActuatorCfg(target_names_expr=("mf_pip",)),
    XmlPositionActuatorCfg(target_names_expr=("mf_dip",)),
    XmlPositionActuatorCfg(target_names_expr=("rf_mcp",)),
    XmlPositionActuatorCfg(target_names_expr=("rf_rot",)),
    XmlPositionActuatorCfg(target_names_expr=("rf_pip",)),
    XmlPositionActuatorCfg(target_names_expr=("rf_dip",)),
    XmlPositionActuatorCfg(target_names_expr=("th_cmc",)),
    XmlPositionActuatorCfg(target_names_expr=("th_axl",)),
    XmlPositionActuatorCfg(target_names_expr=("th_mcp",)),
    XmlPositionActuatorCfg(target_names_expr=("th_ipl",)),
  ),
  soft_joint_pos_limit_factor=1.0,
)


def _get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(
    assets,
    LEAP_LEFT_CUSTOM_XML.parent / "assets",
    meshdir,
  )
  return assets


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LEAP_LEFT_CUSTOM_XML))
  # Lift-style tasks consume a named end-effector site.
  if spec.site("grasp_site") is None:
    spec.body("palm").add_site(
      name="grasp_site",
      pos=(-0.125, -0.055, 0.02),
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(0.005,),
      rgba=(0.0, 1.0, 0.0, 0.0),
      group=4,
    )
  spec.assets = _get_assets(spec.meshdir)
  return spec


def get_leap_left_custom_hand_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=_get_spec,
    articulation=LEAP_LEFT_CUSTOM_ACTUATION,
  )
