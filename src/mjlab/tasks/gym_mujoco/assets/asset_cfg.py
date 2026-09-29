"""MJCF loading helpers; environment and initial-state choices live in task cfgs."""

from functools import partial
from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec import non_default_option_fields

_ASSET_DIR = Path(__file__).parent


def get_entity_cfg(xml_name: str, init_state: EntityCfg.InitialStateCfg) -> EntityCfg:
  return EntityCfg(
    spec_fn=partial(asset_spec, xml_name),
    init_state=init_state,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
      soft_joint_pos_limit_factor=1.0,
    ),
  )


def asset_spec(xml_name: str) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_ASSET_DIR / xml_name))
  for i, body in enumerate(spec.bodies[1:], start=1):
    if not body.name:
      body.name = f"body_{i}"
  for actuator in spec.actuators:
    if not actuator.name:
      actuator.name = actuator.target
  # A native force sensor requests post-constraint wrench computation.
  if Path(xml_name).stem in ("ant", "humanoid", "humanoidstandup"):
    spec.bodies[1].add_site(name="contact_force_origin", size=[0.001] * 3)
    spec.add_sensor(
      name="contact_force",
      type=mujoco.mjtSensor.mjSENS_FORCE,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="contact_force_origin",
    )
  defaults = mujoco.MjSpec().option
  for key in non_default_option_fields(spec.option):
    setattr(spec.option, key, getattr(defaults, key))
  return spec


def scene_options(spec: mujoco.MjSpec, xml_name: str) -> None:
  source = mujoco.MjSpec.from_file(str(_ASSET_DIR / xml_name))
  # Scene attachment does not propagate global options or mass normalization.
  spec.compiler.settotalmass = source.compiler.settotalmass
  spec.nuser_geom = source.nuser_geom
  spec.nkey = source.nkey
  spec.visual.map.znear = source.visual.map.znear
  for key in non_default_option_fields(source.option):
    setattr(spec.option, key, getattr(source.option, key))
