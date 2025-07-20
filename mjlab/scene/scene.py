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

# _HERE = Path(__file__).parent
# _XML = _HERE / "scene.xml"

_XML = r"""
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
    self._indexing: SceneIndexing = SceneIndexing()

    # spec = mujoco.MjSpec.from_file(str(_XML))
    self._spec = mujoco.MjSpec.from_string(_XML)
    # super().__init__(spec)

    self._configure_terrain()
    self._configure_robots()
    self._configure_lights()
    self._configure_cameras()
    self._configure_skybox()

  # Attributes.

  @property
  def entities(self) -> dict[str, entity.Entity]:
    return self._entities

  @property
  def indexing(self) -> SceneIndexing:
    return self._indexing

  # Methods.

  def initialize(self, model: mujoco.MjModel, data, device):
    self._compute_indexing(model, device)
    for ent_name, ent in self._entities.items():
      ent.initialize(self.indexing.entities[ent_name], data, device)

  def reset(self):
    for ent in self._entities.values():
      ent.reset()

  def update(self, dt: float) -> None:
    for ent in self._entities.values():
      ent.update(dt)

  # Private methods.

  def _configure_terrain(self) -> None:
    for ter_name, ter_cfg in self._cfg.terrains.items():
      ter = Terrain(ter_cfg)
      self._entities[ter_name] = ter
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(ter.spec, prefix=f"{ter_name}/", frame=frame)

  def _configure_robots(self) -> None:
    for rob_name, rob_cfg in self._cfg.robots.items():
      rob = Robot(rob_cfg)
      self._entities[rob_name] = rob
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(rob.spec, prefix=f"{rob_name}/", frame=frame)

  def _configure_lights(self) -> None:
    for lig in self._cfg.lights:
      editors.LightEditor(lig).edit_spec(self._spec)

  def _configure_cameras(self) -> None:
    for cam in self._cfg.cameras:
      editors.CameraEditor(cam).edit_spec(self._spec)

  def _configure_skybox(self) -> None:
    if self._cfg.skybox is not None:
      common_editors.TextureEditor(self._cfg.skybox).edit_spec(self._spec)

  def configure_sim_options(self, cfg: OptionCfg) -> None:
    common_editors.OptionEditor(cfg).edit_spec(self._spec)

  def _compute_indexing(self, model: mujoco.MjModel, device: str) -> None:
    for ent_name, ent in self._entities.items():
      body_ids = []
      body_root_ids = []
      for body in ent.spec.bodies:
        body_name = body.name
        if body_name == "world":
          continue
        body = model.body(body_name)
        body_ids.append(body.id)
        body_root_ids.extend(body.rootid)
      body_ids = torch.tensor(body_ids, dtype=torch.int, device=device)
      body_root_ids = torch.tensor(body_root_ids, dtype=torch.int, device=device)

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

      sensor_adr = {}
      for sensor in ent.spec.sensors:
        sensor_name = sensor.name
        sns = model.sensor(sensor_name)
        dim = sns.dim[0]
        start_adr = sns.adr[0]
        sensor_adr[sensor_name] = torch.arange(
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

      root_body_id = None
      for joint in ent.spec.joints:
        jnt = model.joint(joint.name)
        if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE:
          # TODO: Why is jnt.bodyid an array?
          root_body_id = model.jnt_bodyid[jnt.id]

      indexing = EntityIndexing(
        root_body_id=root_body_id,
        body_ids=body_ids,
        body_root_ids=body_root_ids,
        geom_ids=geom_ids,
        site_ids=site_ids,
        sensor_adr=sensor_adr,
        joint_q_adr=torch.tensor(joint_q_adr, dtype=torch.int, device=device),
        joint_v_adr=torch.tensor(joint_v_adr, dtype=torch.int, device=device),
        free_joint_v_adr=torch.tensor(free_joint_v_adr, dtype=torch.int, device=device),
        free_joint_q_adr=torch.tensor(free_joint_q_adr, dtype=torch.int, device=device),
      )
      self._indexing.entities[ent_name] = indexing
