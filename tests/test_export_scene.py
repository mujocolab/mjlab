"""Tests for the export-scene script."""

import mujoco
import pytest

from mjlab.entity import EntityCfg
from mjlab.scripts.export_scene import ExportSceneCfg, export_scene

XML = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


def tiny_entity_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(XML))


TARGET = f"{__name__}:tiny_entity_cfg"


def test_refuses_non_empty_output_dir(tmp_path):
  """Existing files are never deleted without --overwrite."""
  keep = tmp_path / "keep.txt"
  keep.write_text("user data")

  with pytest.raises(SystemExit):
    export_scene(ExportSceneCfg(target=TARGET, output_dir=str(tmp_path)))
  with pytest.raises(SystemExit):
    export_scene(ExportSceneCfg(target=TARGET, output_dir=str(tmp_path), zip=True))

  assert keep.read_text() == "user data"


def test_overwrite_replaces_output_dir(tmp_path):
  out = tmp_path / "export"
  out.mkdir()
  (out / "stale.txt").write_text("old")

  export_scene(ExportSceneCfg(target=TARGET, output_dir=str(out), overwrite=True))

  assert (out / "scene.xml").is_file()
  assert not (out / "stale.txt").exists()
