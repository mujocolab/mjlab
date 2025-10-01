# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mujoco

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.4),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.3,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.3,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    # NOTE: Heightfield terrains are currently disabled due to compilation issues
    # in mujoco-warp.
    # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
    #   proportion=0.1,
    #   slope_range=(0.0, 1.0),
    #   platform_width=2.0,
    #   border_width=0.25,
    # ),
    # "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
    #   proportion=0.1,
    #   slope_range=(0.0, 1.0),
    #   platform_width=2.0,
    #   border_width=0.25,
    #   inverted=True,
    # ),
    # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
    #   proportion=0.2,
    #   noise_range=(0.02, 0.10),
    #   noise_step=0.02,
    #   border_width=0.25,
    # ),
    # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
    #   proportion=0.2,
    #   amplitude_range=(0.0, 0.2),
    #   num_waves=4,
    #   border_width=0.25,
    # ),
  },
  add_lights=False,
)


if __name__ == "__main__":
  import mujoco.viewer

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainImporter(terrain_cfg, device="cuda:0")
  mujoco.viewer.launch(terrain.spec.compile())
