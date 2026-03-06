"""Tests for mjlab.utils.xml."""

import zipfile

import mujoco
import numpy as np

from mjlab.utils.xml import fix_spec_xml, strip_buffer_textures


def test_strip_buffer_textures():
  """Stripping buffer textures removes them and enables XML roundtrip."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="terrain")

  tex = spec.add_texture(
    name="hf_tex",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    width=2,
    height=2,
  )
  tex.data = (np.ones((2, 2, 3), dtype=np.uint8) * 128).tobytes()

  mat = spec.add_material(name="hf_mat")
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hf_tex"

  hf = spec.add_hfield(name="hf", nrow=4, ncol=4, size=[1, 1, 0.1, 0.01])
  hf.userdata = np.zeros(16).tolist()
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_HFIELD,
    hfieldname="hf",
    material="hf_mat",
  )

  strip_buffer_textures(spec)

  assert len(spec.textures) == 0
  assert len(spec.materials) == 0
  assert spec.geoms[0].material == ""

  # XML roundtrip should succeed now.
  model = mujoco.MjModel.from_xml_string(spec.to_xml())
  assert model.nhfield == 1


def test_fix_spec_xml():
  """Removes empty defaults and optionally injects meshdir."""
  xml = (
    "<mujoco>\n"
    "  <compiler/>\n"
    "  <default/>\n"
    '  <default class="foo/bar"/>\n'
    "  <worldbody/>\n"
    "</mujoco>\n"
  )
  result = fix_spec_xml(xml, meshdir="assets")

  assert "<default/>" not in result
  assert 'class="foo/bar"' not in result
  assert 'meshdir="assets"' in result
  assert "meshdir" not in fix_spec_xml(xml)


def test_rough_terrain_write_xml_roundtrip(tmp_path):
  """Export terrain with hfield via Entity.write_xml, reload as MjModel."""
  from mjlab.terrains.heightfield_terrains import (
    HfRandomUniformTerrainCfg,
  )
  from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

  terrain = TerrainEntity(
    TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=TerrainGeneratorCfg(
        size=(2.0, 2.0),
        num_rows=1,
        num_cols=1,
        sub_terrains={
          "rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
          ),
        },
      ),
    ),
    device="cpu",
  )
  xml_path = tmp_path / "terrain.xml"
  terrain.write_xml(xml_path)

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  assert model.nhfield == 1


def test_scene_write_zip(tmp_path):
  """Scene.write produces a directory; with zip=True, a zip archive."""
  from mjlab.scene.scene import Scene, SceneCfg

  scene = Scene(SceneCfg(), device="cpu")

  # Directory export.
  dir_out = tmp_path / "scene_dir"
  scene.write(dir_out)
  assert (dir_out / "scene.xml").exists()
  mujoco.MjModel.from_xml_path(str(dir_out / "scene.xml"))

  # Zip export.
  zip_out = tmp_path / "scene_pkg"
  scene.write(zip_out, zip=True)
  zip_path = zip_out.with_suffix(".zip")
  assert zip_path.exists()
  assert not zip_out.exists()
  with zipfile.ZipFile(zip_path) as zf:
    assert "scene.xml" in zf.namelist()
