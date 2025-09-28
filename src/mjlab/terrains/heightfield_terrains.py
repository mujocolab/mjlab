import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
import scipy.interpolate as interpolate
from scipy import ndimage

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)


def color_by_height(
  spec: mujoco.MjSpec,
  noise: np.ndarray,
  unique_id: str,
  normalized_elevation: np.ndarray,
  texture_size: int = 128,
) -> str:
  texture_name = f"hf_texture_{unique_id}"
  texture = spec.add_texture(
    name=texture_name,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    width=texture_size,
    height=texture_size,
  )

  texture_elevation = ndimage.zoom(
    normalized_elevation,
    (texture_size / noise.shape[0], texture_size / noise.shape[1]),
    order=1,
  )
  texture_elevation = np.asarray(texture_elevation)

  hue = 0.5 - texture_elevation * 0.45
  saturation = 0.6 - texture_elevation * 0.2
  value = 0.4 + texture_elevation * 0.3

  c = value * saturation
  x = c * (1 - np.abs((hue * 6) % 2 - 1))
  m = value - c

  hue_sector = (hue * 6).astype(int) % 6

  r = np.zeros_like(hue)
  g = np.zeros_like(hue)
  b = np.zeros_like(hue)

  # Assign RGB values based on hue sector.
  # Sector 0: R=C, G=X, B=0.
  mask = hue_sector == 0
  r[mask] = c[mask]
  g[mask] = x[mask]

  # Sector 1: R=X, G=C, B=0.
  mask = hue_sector == 1
  r[mask] = x[mask]
  g[mask] = c[mask]

  # Sector 2: R=0, G=C, B=X.
  mask = hue_sector == 2
  g[mask] = c[mask]
  b[mask] = x[mask]

  # Sector 3: R=0, G=X, B=C.
  mask = hue_sector == 3
  g[mask] = x[mask]
  b[mask] = c[mask]

  # Sector 4: R=X, G=0, B=C.
  mask = hue_sector == 4
  r[mask] = x[mask]
  b[mask] = c[mask]

  # Sector 5: R=C, G=0, B=X.
  mask = hue_sector == 5
  r[mask] = c[mask]
  b[mask] = x[mask]

  # Add the minimum value.
  r += m
  g += m
  b += m

  rgb_data = np.stack([r, g, b], axis=-1)
  rgb_data = (rgb_data * 255).astype(np.uint8)

  rgb_data = np.flipud(rgb_data)
  texture.data = rgb_data.tobytes()

  material_name = f"hf_material_{unique_id}"
  material = spec.add_material(name=material_name)
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

  return material_name


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(SubTerrainCfg):
  slope_range: tuple[float, float]
  platform_width: float = 1.0
  inverted: bool = False
  border_width: float = 0.0
  horizontal_scale: float = 0.1
  vertical_scale: float = 0.005
  base_thickness_ratio: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    if self.inverted:
      slope = -self.slope_range[0] - difficulty * (
        self.slope_range[1] - self.slope_range[0]
      )
    else:
      slope = self.slope_range[0] + difficulty * (
        self.slope_range[1] - self.slope_range[0]
      )

    # Switch parameters to discrete units.
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)

    # Height is 1/2 of the width since terrain is a pyramid.
    height_max = int(slope * self.size[0] / 2 / self.vertical_scale)

    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)

    # Offset meshgrid to center of terrain.
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y

    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # Create sloped surface.
    hf_raw = height_max * xx * yy

    # Create flat platform at center.
    platform_width = int(self.platform_width / self.horizontal_scale / 2)

    # Get platform height at corner.
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    # Round heights to nearest vertical step.
    noise = np.rint(hf_raw).astype(np.int16)

    # Calculate elevation range for scaling.
    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )

    # Convert to physical height units.
    max_physical_height = elevation_range * self.vertical_scale

    # Calculate base thickness.
    base_thickness = max_physical_height * self.base_thickness_ratio

    # Normalize elevation data to [0, 1] range.
    if elevation_range > 0:
      normalized_elevation = (noise - elevation_min) / elevation_range
    else:
      normalized_elevation = np.zeros_like(noise)

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{unique_id}",
      size=[
        self.size[0] / 2,  # radius_x.
        self.size[1] / 2,  # radius_y.
        max_physical_height,  # elevation_z.
        base_thickness,  # base_z.
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized_elevation.flatten().astype(np.float32),
    )

    # Calculate z offset for proper alignment.
    if self.inverted:
      # For inverted pyramid, edges (max in data) should be at z=0
      # Accounting for the base thickness that extends below.
      hfield_z_offset = -max_physical_height
    else:
      # For normal pyramid, terrain surface starts at z=0 at the edges.
      hfield_z_offset = 0

    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[
        self.size[0] / 2,
        self.size[1] / 2,
        hfield_z_offset,
      ],
      material=material_name,
    )

    if self.inverted:
      # Platform is at the bottom (min in data = -base_thickness).
      spawn_height = hfield_z_offset - base_thickness
    else:
      # Platform is at the top (max in data = elevation - base).
      spawn_height = max_physical_height - base_thickness

    origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

    geom = TerrainGeometry(geom=hfield_geom, hfield=field)
    return TerrainOutput(origin=origin, geometries=[geom])


@dataclass(kw_only=True)
class HfRandomUniformTerrainCfg(SubTerrainCfg):
  noise_range: tuple[float, float]
  noise_step: float = 0.005
  downsampled_scale: float | None = None
  horizontal_scale: float = 0.1
  vertical_scale: float = 0.005
  base_thickness_ratio: float = 1.0
  border_width: float = 0.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty  # Unused.

    body = spec.body("terrain")

    if self.downsampled_scale is None:
      downsampled_scale = self.horizontal_scale
    elif self.downsampled_scale < self.horizontal_scale:
      raise ValueError(
        f"Downsampled scale must be >= horizontal scale: "
        f"{self.downsampled_scale} < {self.horizontal_scale}"
      )
    else:
      downsampled_scale = self.downsampled_scale

    # Switch parameters to discrete units.
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)
    width_downsampled = int(self.size[0] / downsampled_scale)
    length_downsampled = int(self.size[1] / downsampled_scale)
    height_min = int(self.noise_range[0] / self.vertical_scale)
    height_max = int(self.noise_range[1] / self.vertical_scale)
    height_step = int(self.noise_step / self.vertical_scale)

    # Create range of heights possible.
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # Sample heights randomly from the range along a grid.
    height_field_downsampled = rng.choice(
      height_range, size=(width_downsampled, length_downsampled)
    )

    # Create interpolation function for the sampled heights.
    x = np.linspace(0, self.size[0], width_downsampled)
    y = np.linspace(0, self.size[1], length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # Interpolate the sampled heights to obtain the height field.
    x_upsampled = np.linspace(0, self.size[0], width_pixels)
    y_upsampled = np.linspace(0, self.size[1], length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # Round off the interpolated heights to the nearest vertical step.
    noise = np.rint(z_upsampled).astype(np.int16)

    # Calculate elevation range for scaling.
    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )

    # Convert to physical height units.
    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio

    # Normalize elevation data to [0, 1] range.
    if elevation_range > 0:
      normalized_elevation = (noise - elevation_min) / elevation_range
    else:
      normalized_elevation = np.zeros_like(noise)

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{unique_id}",
      size=[
        self.size[0] / 2,  # radius_x
        self.size[1] / 2,  # radius_y
        max_physical_height,  # elevation_z
        base_thickness,  # base_z
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized_elevation.flatten().astype(np.float32),
    )

    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[self.size[0] / 2, self.size[1] / 2, 0],
      material=material_name,
    )

    spawn_height = (self.noise_range[0] + self.noise_range[1]) / 2
    origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

    geom = TerrainGeometry(geom=hfield_geom, hfield=field)
    return TerrainOutput(origin=origin, geometries=[geom])


@dataclass(kw_only=True)
class HfWaveTerrainCfg(SubTerrainCfg):
  amplitude_range: tuple[float, float]
  num_waves: float = 1.0
  horizontal_scale: float = 0.1
  vertical_scale: float = 0.005
  base_thickness_ratio: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    # Check number of waves
    if self.num_waves <= 0:
      raise ValueError(f"Number of waves must be positive. Got: {self.num_waves}")

    amplitude = self.amplitude_range[0] + difficulty * (
      self.amplitude_range[1] - self.amplitude_range[0]
    )

    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / self.vertical_scale)

    wave_length = length_pixels / self.num_waves
    wave_number = 2 * np.pi / wave_length

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
    noise = np.rint(hf_raw).astype(np.int16)

    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )

    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio

    if elevation_range > 0:
      normalized_elevation = (noise - elevation_min) / elevation_range
    else:
      normalized_elevation = np.zeros_like(noise)

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{unique_id}",
      size=[
        self.size[0] / 2,
        self.size[1] / 2,
        max_physical_height,
        base_thickness,
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized_elevation.flatten().astype(np.float32),
    )

    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[self.size[0] / 2, self.size[1] / 2, -max_physical_height / 2],
      material=material_name,
    )

    spawn_height = 0.0
    origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

    geom = TerrainGeometry(geom=hfield_geom, hfield=field)
    return TerrainOutput(origin=origin, geometries=[geom])
