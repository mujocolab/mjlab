"""Interactive terrain visualizer using Viser.

Displays a 10-row grid of terrains with increasing difficulty.
Configurations and parameters are dynamically loaded from mjlab.terrains.config.

Run with:
  uv run python src/mjlab/scripts/visualize_terrain.py
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Type

import mujoco
import numpy as np
import trimesh
import viser
import viser.transforms as vtf

import mjlab.terrains as terrain_gen
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainGenerator
from mjlab.viewer.viser.conversions import merge_geoms

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

def main():
    server = viser.ViserServer()
    
    # Load available terrains from config.
    available_presets = ROUGH_TERRAINS_CFG.sub_terrains
    preset_names = ["All Terrains"] + list(available_presets.keys())
    
    # State management.
    state = {
        "preset_name": preset_names[0],
        "seed": 42,
        "size": ROUGH_TERRAINS_CFG.size,
        "params": {},
        "rows": 10,
        "cols": 1,
        "difficulty_range": (0.0, 1.0),
    }
    
    # Handle for the terrain mesh in the scene.
    terrain_handle: viser.SceneNodeHandle | None = None

    # GUI for statistics.
    gui_stats_folder = server.gui.add_folder("Statistics")
    with gui_stats_folder:
        polygon_count_label = server.gui.add_markdown("**Number of Polygons:** -")

    def update_terrain():
        nonlocal terrain_handle
        
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
                
                if "range" in field.name and isinstance(getattr(selected_instance, field.name), (tuple, list)):
                    if field.name + "_min" in state["params"]:
                        sub_cfg_params[field.name] = (
                            state["params"][field.name + "_min"],
                            state["params"][field.name + "_max"]
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
            
            sub_terrains = {"main": sub_cfg}
            num_cols = state["cols"]
            num_rows = state["rows"]

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
        model = spec.compile()
        
        # The terrain body is named "terrain".
        terrain_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
        geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == terrain_body_id]
        
        if not geom_ids:
            print("No terrain geoms found.")
            return

        mesh = merge_geoms(model, geom_ids)
        polygon_count_label.content = f"**Number of Polygons:** {len(mesh.faces):,}"
        
        # Remove old mesh if exists.
        if terrain_handle is not None:
            terrain_handle.remove()
            
        terrain_handle = server.scene.add_mesh_trimesh(
            "/terrain",
            mesh,
            position=(0, 0, 0),
        )

    # GUI Setup.
    gui_params_folder = server.gui.add_folder("Terrain Parameters")
    param_controls: List[viser.GuiControlHandle] = []

    def rebuild_gui():
        nonlocal param_controls
        for control in param_controls:
            control.remove()
        param_controls.clear()
        
        if state["preset_name"] == "All Terrains":
            with gui_params_folder:
                server.gui.add_markdown("_Parameters not available for 'All Terrains' mode._")
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
                if "range" in field.name and isinstance(getattr(selected_instance, field.name), (tuple, list)):
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
                    v_min = min(v_min, cur_min)
                    v_max = max(v_max, cur_max)

                    s_min = server.gui.add_slider(
                        f"{field.name} min",
                        min=v_min, max=v_max, step=v_step,
                        initial_value=cur_min
                    )
                    s_max = server.gui.add_slider(
                        f"{field.name} max",
                        min=v_min, max=v_max, step=v_step,
                        initial_value=cur_max
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
                    v_min = min(v_min, cur_val)
                    v_max = max(v_max, cur_val)

                    slider = server.gui.add_slider(
                        field.name,
                        min=v_min, max=v_max, step=v_step,
                        initial_value=cur_val
                    )
                    
                    @slider.on_update
                    def _(event, name=field.name, is_int=("int" in type_str) or field.type is int):
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
                        field.name,
                        initial_value=state["params"][field.name]
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
                                    min=min(0.0, val), max=max(10.0, val), step=0.1,
                                    initial_value=val
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
            "Preset",
            options=preset_names,
            initial_value=state["preset_name"]
        )
        
        @preset_select.on_update
        def _(event):
            state["preset_name"] = event.target.value
            state["params"] = {} # Clear local overrides for new preset.
            rebuild_gui()
            update_terrain()

        seed_input = server.gui.add_number("Seed", initial_value=state["seed"])
        @seed_input.on_update
        def _(event):
            state["seed"] = event.target.value
            update_terrain()

        btn_randomize = server.gui.add_button("Randomize Seed")
        @btn_randomize.on_click
        def _(_):
            new_seed = np.random.randint(0, 10000)
            seed_input.value = new_seed
            state["seed"] = new_seed
            update_terrain()

    # Initialize.
    rebuild_gui()
    update_terrain()

    print("Viser Terrain Visualizer running...")
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
