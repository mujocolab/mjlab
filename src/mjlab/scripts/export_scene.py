"""Export a task scene or asset_zoo entity to a directory."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro
import tyro.conf

import mjlab
import mjlab.tasks  # noqa: F401
from mjlab.scene import Scene, SceneCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.string import string_to_callable

ENTITY_ALIASES: dict[str, str] = {
  "g1": "mjlab.asset_zoo.robots:get_g1_robot_cfg",
  "go1": "mjlab.asset_zoo.robots:get_go1_robot_cfg",
  "yam": "mjlab.asset_zoo.robots:get_yam_robot_cfg",
}


@dataclass
class ExportSceneCfg:
  target: tyro.conf.Positional[str]
  """Task ID, entity alias (g1, go1, yam), or import path (pkg.module:get_cfg)."""

  output_dir: str = "export"
  """Output directory."""

  zip: bool = False
  """Compress into a zip archive."""


def export_scene(cfg: ExportSceneCfg) -> None:
  """Export a task scene or entity to a directory."""
  task_ids = list_tasks()
  target = ENTITY_ALIASES.get(cfg.target, cfg.target)

  if target in task_ids:
    env_cfg = load_env_cfg(target)
    scene = Scene(env_cfg.scene, device="cpu")
  elif ":" in target:
    try:
      factory = string_to_callable(target)
    except (ValueError, ImportError) as e:
      print(f"Error: {e}")
      return
    entity_cfg = factory()
    scene_cfg = SceneCfg(entities={"robot": entity_cfg})
    scene = Scene(scene_cfg, device="cpu")
  else:
    print(f"Unknown target: {cfg.target}\n")
    print("Available task IDs:")
    for tid in task_ids:
      print(f"  {tid}")
    print("\nAvailable entity aliases:")
    for name in sorted(ENTITY_ALIASES):
      print(f"  {name}")
    print("\nYou can also pass an import path, e.g. my_pkg.robots:get_my_robot_cfg")
    return

  output_dir = Path(cfg.output_dir)
  if output_dir.exists():
    shutil.rmtree(output_dir)
  scene.write(output_dir, zip=cfg.zip)
  if cfg.zip:
    print(f"Exported to {output_dir.with_suffix('.zip')}")
  else:
    print(f"Exported to {output_dir}")


def main():
  cfg = tyro.cli(ExportSceneCfg, config=mjlab.TYRO_FLAGS)
  export_scene(cfg)


if __name__ == "__main__":
  main()
