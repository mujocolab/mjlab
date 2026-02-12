"""Interactive terrain visualizer using Viser.

Displays a 10-row grid of terrains with increasing difficulty.
Configurations and parameters are dynamically loaded from mjlab.terrains.config.

Run with:
  uv run src/mjlab/scripts/visualize_terrain.py
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any, List, TypedDict

import mujoco
import numpy as np
import viser

from mjlab.asset_zoo.robots import (
  get_g1_robot_cfg,
  get_go1_robot_cfg,
  get_yam_robot_cfg,
)
from mjlab.terrains.config import ALL_TERRAINS_CFG
from mjlab.terrains.terrain_generator import (
  TerrainGenerator,
  TerrainGeneratorCfg,
)
from mjlab.viewer.viser.conversions import (
  merge_geoms,
  merge_geoms_global,
)

# Supported robots for visualization.
ROBOT_CFG_GETTERS = {
  "None": None,
  "Unitree Go1": get_go1_robot_cfg,
  "Unitree G1": get_g1_robot_cfg,
  "Yam": get_yam_robot_cfg,
}

# Parameter range hints for sliders.
PARAM_HINTS = {
  "octaves": (1, 10, 1),
  "persistence": (0.0, 1.0, 0.05),
  "lacunarity": (1.0, 5.0, 0.1),
  "scale": (0.01, 0.5, 0.01),
  "horizontal_scale": (0.001, 0.1, 0.001),
  "resolution": (0.1, 2.0, 0.1),
  "base_thickness_ratio": (0.1, 2.0, 0.1),
  "border_width": (0.0, 2.0, 0.05),
  "amplitude_range": (0.0, 1.0, 0.05),
  "height_range": (0.0, 2.0, 0.05),
  "num_waves": (1, 20, 1),
  "num_obstacles": (1, 100, 1),
  "obstacle_height_range": (0.0, 1.0, 0.05),
  "obstacle_width_range": (0.1, 2.0, 0.05),
  "box_width_range": (0.1, 2.0, 0.05),
  "box_length_range": (0.1, 2.0, 0.05),
  "slope_range": (0.0, 1.0, 0.05),
  "platform_width": (0.1, 5.0, 0.1),
  "step_height_range": (0.0, 0.5, 0.01),
  "step_width": (0.1, 1.0, 0.05),
  "grid_width": (0.1, 1.0, 0.05),
  "grid_height_range": (0.0, 1.0, 0.05),
  "height_merge_threshold": (0.01, 0.2, 0.01),
  "max_merge_distance": (1, 10, 1),
  "num_beams": (1, 64, 1),
  "num_rings": (1, 32, 1),
  "displacement_range": (0.0, 1.0, 0.005),
  "stone_size_variation": (0.0, 1.0, 0.005),
  "stone_height_variation": (0.0, 1.0, 0.005),
}


class _AppState(TypedDict):
  preset_name: str
  robot_name: str
  seed: int
  size: tuple[float, float]
  params: dict[str, Any]
  rows: int
  cols: int
  difficulty_range: tuple[float, float]
  robot_handles: list[Any]
  terrain_origins: np.ndarray | None


def main():
  server = viser.ViserServer()

  # Load available terrains from config.
  available_presets = ALL_TERRAINS_CFG.sub_terrains
  preset_names = ["All Terrains"] + list(available_presets.keys())

  # State management.
  state: _AppState = {
    "preset_name": preset_names[0],
    "robot_name": "None",
    "seed": 42,
    "size": ALL_TERRAINS_CFG.size,
    "params": {},
    "rows": 10,
    "cols": 1,
    "difficulty_range": (0.0, 1.0),
    "robot_handles": [],
    "terrain_origins": None,
  }

  # Handle for the terrain mesh in the scene.
  terrain_handle: viser.SceneNodeHandle | None = None

  # GUI for statistics.
  gui_stats_folder = server.gui.add_folder("Statistics")
  with gui_stats_folder:
    status_label = server.gui.add_markdown("**Status:** Ready")
    polygon_count_label = server.gui.add_markdown("**Number of Polygons:** -")

  def update_robots():
    # Clear old robot handles.
    for h in state["robot_handles"]:
      h.remove()
    state["robot_handles"] = []

    if state["robot_name"] == "None" or state["terrain_origins"] is None:
      status_label.content = "**Status:** Ready"
      return

    status_label.content = f"**Status:** Spawning {state['robot_name']}..."

    # Get robot config.
    robot_cfg_getter = ROBOT_CFG_GETTERS[state["robot_name"]]
    if robot_cfg_getter is None:
      return
    robot_cfg = robot_cfg_getter()

    # Merge robot template mesh.
    # We compile the robot spec standalone to get a clean local-space mesh template.
    robot_spec = robot_cfg.spec_fn()
    if robot_spec.worldbody.bodies:
      robot_spec.worldbody.bodies[0].pos = (0, 0, 0)

    if robot_spec.keys:
      robot_spec.delete(robot_spec.keys[0])

    robot_model = robot_spec.compile()
    robot_data = mujoco.MjData(robot_model)

    # Apply joint poses.
    joint_pos = robot_cfg.init_state.joint_pos
    if joint_pos is None:
      joint_pos = {}
    for pattern, val in joint_pos.items():
      if not pattern.startswith("^"):
        pattern = ".*" + pattern
      for j_id in range(robot_model.njnt):
        name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if re.match(pattern, name):
          adr = robot_model.jnt_qposadr[j_id]
          robot_data.qpos[adr] = val

    mujoco.mj_forward(robot_model, robot_data)

    # Filter for visual geoms only (group < 3).
    visual_geom_ids = [
      i for i in range(robot_model.ngeom) if robot_model.geom_group[i] < 3
    ]
    if not visual_geom_ids:
      status_label.content = "**Status:** Error: No visual geoms for robot"
      print(f"Error: No visual geoms found for {state['robot_name']}")
      return

    robot_mesh = merge_geoms_global(robot_model, robot_data, visual_geom_ids)
    n_verts = len(robot_mesh.vertices)
    n_faces = len(robot_mesh.faces)
    print(f"Robot mesh: {n_verts} vertices, {n_faces} faces.")

    # Prepare batched positions.
    num_rows, num_cols = state["terrain_origins"].shape[:2]
    batched_positions = []
    for row in range(num_rows):
      for col in range(num_cols):
        origin = state["terrain_origins"][row, col]
        pos = origin + np.array(robot_cfg.init_state.pos)
        batched_positions.append(pos)

    batched_positions = np.array(batched_positions)
    n = len(batched_positions)
    print(f"Instancing {n} robots at positions like {batched_positions[0]}")
    batched_wxyzs = np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
      len(batched_positions), axis=0
    )

    with server.atomic():
      handle = server.scene.add_batched_meshes_trimesh(
        "/robots_batched",
        robot_mesh,
        batched_positions=batched_positions,
        batched_wxyzs=batched_wxyzs,
      )
      state["robot_handles"].append(handle)
    status_label.content = "**Status:** Ready"

  def update_terrain():
    nonlocal terrain_handle
    status_label.content = "**Status:** Generating terrain..."

    if state["preset_name"] == "All Terrains":
      # Create a copy with equal proportions to ensure all are shown once.
      sub_terrains = {}
      for name, cfg in available_presets.items():
        new_cfg = dataclasses.replace(cfg, proportion=1.0)
        sub_terrains[name] = new_cfg
      num_cols = len(sub_terrains)
      num_rows = state["rows"]
    else:
      selected_instance = available_presets[state["preset_name"]]
      terrain_type = type(selected_instance)

      # Instantiate sub-terrain config with current GUI state.
      sub_cfg_params = {}
      for field in dataclasses.fields(terrain_type):
        if field.name in ["proportion", "size", "flat_patch_sampling"]:
          sub_cfg_params[field.name] = getattr(selected_instance, field.name)
          continue

        if "range" in field.name and isinstance(
          getattr(selected_instance, field.name), (tuple, list)
        ):
          if field.name + "_min" in state["params"]:
            sub_cfg_params[field.name] = (
              state["params"][field.name + "_min"],
              state["params"][field.name + "_max"],
            )
          else:
            sub_cfg_params[field.name] = getattr(selected_instance, field.name)
        elif field.name in state["params"]:
          sub_cfg_params[field.name] = state["params"][field.name]
        else:
          sub_cfg_params[field.name] = getattr(selected_instance, field.name)

      try:
        sub_cfg = terrain_type(**sub_cfg_params)
      except Exception as e:
        print(f"Error creating config: {e}")
        return

      num_rows = state["rows"]
      num_cols = state["cols"]
      sub_terrains = {state["preset_name"]: sub_cfg}

    generator_cfg = TerrainGeneratorCfg(
      seed=state["seed"],
      size=state["size"],
      num_rows=num_rows,
      num_cols=num_cols,
      curriculum=True,
      difficulty_range=state["difficulty_range"],
      sub_terrains=sub_terrains,
      add_lights=True,
    )

    generator = TerrainGenerator(generator_cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)

    # Save terrain metadata for robot spawning.
    state["terrain_origins"] = generator.terrain_origins

    status_label.content = "**Status:** Compiling MuJoCo model..."
    model = spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # Find terrain geoms.
    terrain_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    terrain_geom_ids = [
      i for i in range(model.ngeom) if model.geom_bodyid[i] == terrain_body_id
    ]

    if not terrain_geom_ids:
      status_label.content = "**Status:** Error: No terrain geoms found"
      print("No terrain geoms found.")
      return

    status_label.content = (
      f"**Status:** Merging {len(terrain_geom_ids)} terrain geoms..."
    )
    terrain_mesh = merge_geoms(model, terrain_geom_ids)
    num_faces = len(terrain_mesh.faces)
    polygon_count_label.content = f"**Number of Polygons:** {num_faces:,}"

    # Remove old terrain mesh if exists.
    if terrain_handle is not None:
      terrain_handle.remove()

    status_label.content = "**Status:** Uploading terrain mesh..."
    terrain_handle = server.scene.add_mesh_trimesh(
      "/terrain",
      terrain_mesh,
    )

    # Trigger robot update.
    update_robots()

  # GUI Setup.
  gui_params_folder = server.gui.add_folder("Terrain Parameters")
  param_controls: List[Any] = []

  def rebuild_gui():
    nonlocal param_controls
    for control in param_controls:
      control.remove()
    param_controls.clear()

    if state["preset_name"] == "All Terrains":
      with gui_params_folder:
        md = server.gui.add_markdown(
          "_Parameters not available for 'All Terrains' mode._"
        )
        param_controls.append(md)
      return

    selected_instance = available_presets[state["preset_name"]]
    terrain_type = type(selected_instance)
    fields = dataclasses.fields(terrain_type)

    with gui_params_folder:
      for field in fields:
        if field.name in ["proportion", "size", "flat_patch_sampling"]:
          continue

        # Get type as string for comparison (handles future annotations).
        type_str = str(field.type)

        # Check for range tuples first.
        if "range" in field.name and isinstance(
          getattr(selected_instance, field.name), (tuple, list)
        ):
          hint = PARAM_HINTS.get(field.name, (0.0, 1.0, 0.01))

          val_min, val_max = getattr(selected_instance, field.name)

          # Store in state if not present.
          if field.name + "_min" not in state["params"]:
            state["params"][field.name + "_min"] = val_min
            state["params"][field.name + "_max"] = val_max

          cur_min = state["params"][field.name + "_min"]
          cur_max = state["params"][field.name + "_max"]

          v_min, v_max, v_step = hint
          # Ensure range is valid for sliders.
          if cur_min is not None:
            v_min = min(v_min, cur_min)
          else:
            cur_min = v_min

          if cur_max is not None:
            v_max = max(v_max, cur_max)
          else:
            cur_max = v_max

          s_min = server.gui.add_slider(
            f"{field.name} min",
            min=v_min,
            max=v_max,
            step=v_step,
            initial_value=cur_min,
          )
          s_max = server.gui.add_slider(
            f"{field.name} max",
            min=v_min,
            max=v_max,
            step=v_step,
            initial_value=cur_max,
          )

          @s_min.on_update
          def _(event, name=field.name):
            state["params"][name + "_min"] = event.target.value
            update_terrain()

          @s_max.on_update
          def _(event, name=field.name):
            state["params"][name + "_max"] = event.target.value
            update_terrain()

          param_controls.extend([s_min, s_max])

        elif "float" in type_str or "int" in type_str or field.type in [float, int]:
          hint = PARAM_HINTS.get(field.name, (0.0, 10.0, 0.1))
          val = getattr(selected_instance, field.name)

          if field.name not in state["params"]:
            state["params"][field.name] = val

          cur_val = state["params"][field.name]
          v_min, v_max, v_step = hint
          if cur_val is not None:
            v_min = min(v_min, cur_val)
            v_max = max(v_max, cur_val)
          else:
            cur_val = (v_min + v_max) / 2.0

          slider = server.gui.add_slider(
            field.name, min=v_min, max=v_max, step=v_step, initial_value=cur_val
          )

          @slider.on_update
          def _(
            event, name=field.name, is_int=("int" in type_str) or field.type is int
          ):
            val = event.target.value
            if is_int:
              val = int(val)
            state["params"][name] = val
            update_terrain()

          param_controls.append(slider)

        elif "bool" in type_str or field.type is bool:
          val = getattr(selected_instance, field.name)
          if field.name not in state["params"]:
            state["params"][field.name] = val

          checkbox = server.gui.add_checkbox(
            field.name, initial_value=state["params"][field.name]
          )

          @checkbox.on_update
          def _(event, name=field.name):
            state["params"][name] = event.target.value
            update_terrain()

          param_controls.append(checkbox)
        else:
          # Fallback for other simple types if they have a default value.
          try:
            val = getattr(selected_instance, field.name)
            if isinstance(val, (int, float)):
              if field.name not in state["params"]:
                state["params"][field.name] = val
              slider = server.gui.add_slider(
                field.name,
                min=min(0.0, val),
                max=max(10.0, val),
                step=0.1,
                initial_value=val,
              )

              @slider.on_update
              def _(event, name=field.name):
                state["params"][name] = event.target.value
                update_terrain()

              param_controls.append(slider)
          except Exception:
            pass

  # Global Controls.
  with server.gui.add_folder("Global Settings"):
    preset_select = server.gui.add_dropdown(
      "Preset", options=preset_names, initial_value=state["preset_name"]
    )

    @preset_select.on_update
    def _(event):
      state["preset_name"] = event.target.value
      state["params"] = {}  # Clear local overrides for new preset.
      rebuild_gui()
      update_terrain()

    seed_input = server.gui.add_number("Seed", initial_value=int(state["seed"]))

    @seed_input.on_update
    def _(event):
      state["seed"] = int(event.target.value)
      update_terrain()

    robot_select = server.gui.add_dropdown(
      "Robot", options=list(ROBOT_CFG_GETTERS.keys()), initial_value=state["robot_name"]
    )

    @robot_select.on_update
    def _(event):
      state["robot_name"] = event.target.value
      update_robots()

    btn_randomize = server.gui.add_button("Randomize Seed")

    @btn_randomize.on_click
    def _(_):
      new_seed = np.random.randint(0, 10000)
      seed_input.value = new_seed
      state["seed"] = int(new_seed)
      update_terrain()

    btn_reset_camera = server.gui.add_button("Reset Camera")

    @btn_reset_camera.on_click
    def _(_):
      for client in server.get_clients().values():
        client.camera.position = (10, 10, 10)
        client.camera.look_at = (0, 0, 0)

  # Initialize.
  rebuild_gui()
  update_terrain()

  print("Viser Terrain Visualizer running...")
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
