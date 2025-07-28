from typing import Any, Sequence

import mujoco
import torch

from mjlab.entities import entity
from mjlab.scene.scene_config import SceneCfg
from mjlab.entities.robots.robot import Robot
from mjlab.entities.terrains.terrain import Terrain
from mjlab.utils.spec_editor.spec_editor_config import OptionCfg
from mjlab.utils.spec_editor import spec_editor as common_editors
from mjlab.entities.indexing import EntityIndexing, SceneIndexing
from mjlab.utils.mujoco import dof_width, qpos_width
from mjlab.sensors import SensorBase

_BASE_XML = r"""
<mujoco model="mjlab scene">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba force="1 0 0 1" haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080"/>
    <map force="0.005"/>
    <scale forcewidth="0.25" contactwidth="0.4" contactheight="0.15"/>
    <quality shadowsize="8192"/>
  </visual>
  <statistic meansize="0.02"/>
</mujoco>
"""


class Scene:
  def __init__(self, scene_cfg: SceneCfg):
    self._cfg = scene_cfg

    self._entities: dict[str, entity.Entity] = {}
    self._sensors: dict[str, SensorBase] = {}
    self._indexing: SceneIndexing = SceneIndexing()

    self._spec = mujoco.MjSpec.from_string(_BASE_XML)
    self._attach_terrains()
    self._attach_robots()
    self._attach_sensors()

  def compile(self):
    return self._spec.compile()

  def configure_sim_options(self, cfg: OptionCfg) -> None:
    common_editors.OptionEditor(cfg).edit_spec(self._spec)

  # Attributes.

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def entities(self) -> dict[str, entity.Entity]:
    return self._entities

  @property
  def sensors(self) -> dict[str, SensorBase]:
    return self._sensors

  @property
  def indexing(self) -> SceneIndexing:
    return self._indexing

  def __getitem__(self, key: str) -> Any:
    all_keys = []
    for asset_family in [
      self._entities,
      self._sensors,
    ]:
      out = asset_family.get(key)
      if out is not None:
        return out
      all_keys += list(asset_family.keys())
    raise KeyError(
      f"Scene entity with key '{key}' not found. Available Entities: '{all_keys}'"
    )

  # Methods.

  def initialize(self, model: mujoco.MjModel, data, device, wp_model):
    self._compute_indexing(model, device)
    for ent_name, ent in self._entities.items():
      ent.initialize(self.indexing.entities[ent_name], data, device)
    for sens in self._sensors.values():
      sens.initialize(
        self.indexing.entities[sens.cfg.entity_name], model, data, device, wp_model
      )

  def reset(self, env_ids: Sequence[int] | None = None) -> None:
    for ent in self._entities.values():
      ent.reset(env_ids)
    for sns in self._sensors.values():
      sns.reset(env_ids)

  def update(self, dt: float) -> None:
    for ent in self._entities.values():
      ent.update(dt)
    for sns in self._sensors.values():
      sns.update(dt=dt, force_recompute=not self._cfg.lazy_sensor_update)

  def write_data_to_sim(self) -> None:
    for ent in self._entities.values():
      ent.write_data_to_sim()

  # Private methods.

  def _attach_terrains(self) -> None:
    for ter_name, ter_cfg in self._cfg.terrains.items():
      ter = Terrain(ter_cfg)
      self._entities[ter_name] = ter
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(ter.spec, prefix=f"{ter_name}/", frame=frame)

  def _attach_robots(self) -> None:
    for rob_name, rob_cfg in self._cfg.robots.items():
      rob = Robot(rob_cfg)
      self._entities[rob_name] = rob
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(rob.spec, prefix=f"{rob_name}/", frame=frame)

  def _attach_sensors(self) -> None:
    for sns_name, sns_cfg in self._cfg.sensors.items():
      sns = sns_cfg.class_type(sns_cfg)
      self._sensors[sns_name] = sns

  def _compute_indexing(self, model: mujoco.MjModel, device: str) -> None:
    for ent_name, ent in self._entities.items():
      body_ids = []
      body_root_ids = []
      body_iquats = []
      for body in ent.spec.bodies:
        body_name = body.name
        if body_name == "world":
          continue
        body = model.body(body_name)
        body_ids.append(body.id)
        body_root_ids.extend(body.rootid)
        body_iquats.append(body.iquat)
      body_ids = torch.tensor(body_ids, dtype=torch.int, device=device)
      body_root_ids = torch.tensor(body_root_ids, dtype=torch.int, device=device)
      body_iquats = torch.tensor(body_iquats, dtype=torch.float, device=device)

      geom_ids = []
      for geom in ent.spec.geoms:
        geom_name = geom.name
        geom_id = model.geom(geom_name).id
        geom_ids.append(geom_id)
      geom_ids = torch.tensor(geom_ids, dtype=torch.int, device=device)

      site_ids = []
      for site in ent.spec.sites:
        site_name = site.name
        site_id = model.site(site_name).id
        site_ids.append(site_id)
      site_ids = torch.tensor(site_ids, dtype=torch.int, device=device)

      ctrl_ids = []
      for actuator in ent.spec.actuators:
        act = model.actuator(actuator.name)
        ctrl_ids.append(act.id)

      root_body_id = None
      for joint in ent.spec.joints:
        jnt = model.joint(joint.name)
        if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE:
          # TODO: Why is jnt.bodyid an array?
          root_body_id = model.jnt_bodyid[jnt.id]

      root_body_iquat = None
      if root_body_id is not None:
        root_body_iquat = torch.tensor(
          model.body_iquat[root_body_id], dtype=torch.float, device=device
        )

      sensor_adr = {}
      for sensor in ent.spec.sensors:
        sensor_name = sensor.name
        sns = model.sensor(sensor_name)
        dim = sns.dim[0]
        start_adr = sns.adr[0]
        sensor_adr[sensor_name.split("/")[1]] = torch.arange(
          start_adr, start_adr + dim, dtype=torch.int, device=device
        )

      joint_q_adr = []
      joint_v_adr = []
      free_joint_q_adr = []
      free_joint_v_adr = []
      for joint in ent.spec.joints:
        jnt = model.joint(joint.name)
        jnt_type = jnt.type[0]
        vadr = jnt.dofadr[0]
        qadr = jnt.qposadr[0]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
          free_joint_v_adr.extend(range(vadr, vadr + 6))
          free_joint_q_adr.extend(range(qadr, qadr + 7))
        else:
          vdim = dof_width(jnt_type)
          joint_v_adr.extend(range(vadr, vadr + vdim))
          qdim = qpos_width(jnt_type)
          joint_q_adr.extend(range(qadr, qadr + qdim))

      indexing = EntityIndexing(
        root_body_id=root_body_id,
        body_ids=body_ids,
        body_root_ids=body_root_ids,
        geom_ids=geom_ids,
        site_ids=site_ids,
        ctrl_ids=torch.tensor(ctrl_ids, dtype=torch.int, device=device),
        root_body_iquat=root_body_iquat,
        body_iquats=body_iquats,
        sensor_adr=sensor_adr,
        joint_q_adr=torch.tensor(joint_q_adr, dtype=torch.int, device=device),
        joint_v_adr=torch.tensor(joint_v_adr, dtype=torch.int, device=device),
        free_joint_v_adr=torch.tensor(free_joint_v_adr, dtype=torch.int, device=device),
        free_joint_q_adr=torch.tensor(free_joint_q_adr, dtype=torch.int, device=device),
      )
      self._indexing.entities[ent_name] = indexing
