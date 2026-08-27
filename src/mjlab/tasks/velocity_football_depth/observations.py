"""Image observations for the depth-image football task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
import torch.nn.functional as F

from mjlab.sensor import CameraSensor
from mjlab.tasks.velocity_football.mdp.observations import ball_pos_b

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_depth_noise_generators: dict[tuple[int, str], torch.Generator] = {}


def _depth_randomization_state(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  device: torch.device,
  *,
  depth_scale_range: tuple[float, float],
  depth_bias_range: tuple[float, float],
  crop_shift_x_pixels: int,
  crop_shift_y_pixels: int,
) -> dict[str, torch.Tensor]:
  """Return per-episode depth calibration/crop perturbations."""
  cache_key = f"_depth_camera_randomization_{sensor_name}"
  cache = vars(env).get(cache_key)
  valid = (
    isinstance(cache, dict)
    and isinstance(cache.get("last_episode_step"), torch.Tensor)
    and cache["last_episode_step"].shape == env.episode_length_buf.shape
  )
  if not valid:
    cache = {
      "last_episode_step": torch.full_like(env.episode_length_buf, -1),
      "scale": torch.ones(env.num_envs, device=device),
      "bias": torch.zeros(env.num_envs, device=device),
      "crop_x": torch.zeros(env.num_envs, dtype=torch.long, device=device),
      "crop_y": torch.zeros(env.num_envs, dtype=torch.long, device=device),
      "last_frame": torch.empty(0, device=device),
      "initialized": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
    }
    vars(env)[cache_key] = cache
  assert isinstance(cache, dict)

  last_step = cache["last_episode_step"]
  initialized = cache["initialized"]
  reset = ~initialized | (
    (env.episode_length_buf == 0) & (last_step != env.episode_length_buf)
  )
  reset_ids = torch.where(reset)[0]
  if reset_ids.numel() > 0:
    generator = _depth_noise_generator(env, device)
    count = reset_ids.numel()
    cache["scale"][reset_ids] = torch.empty(count, device=device).uniform_(
      *depth_scale_range, generator=generator
    )
    cache["bias"][reset_ids] = torch.empty(count, device=device).uniform_(
      *depth_bias_range, generator=generator
    )
    cache["crop_x"][reset_ids] = torch.randint(
      -crop_shift_x_pixels,
      crop_shift_x_pixels + 1,
      (count,),
      device=device,
      generator=generator,
    )
    cache["crop_y"][reset_ids] = torch.randint(
      -crop_shift_y_pixels,
      crop_shift_y_pixels + 1,
      (count,),
      device=device,
      generator=generator,
    )
    initialized[reset_ids] = True
  last_step.copy_(env.episode_length_buf)
  cache["reset_mask"] = reset
  return cache


def _shift_depth_image(
  depth: torch.Tensor,
  shift_x: torch.Tensor,
  shift_y: torch.Tensor,
) -> torch.Tensor:
  """Translate each environment's image independently, filling exposed pixels."""
  batch, channels, height, width = depth.shape
  x = torch.arange(width, device=depth.device).view(1, width) + shift_x[:, None]
  y = torch.arange(height, device=depth.device).view(1, height) + shift_y[:, None]
  valid_x = (x >= 0) & (x < width)
  valid_y = (y >= 0) & (y < height)
  x = x.clamp(0, width - 1)
  y = y.clamp(0, height - 1)
  shifted = torch.gather(
    depth,
    3,
    x[:, None, None, :].expand(batch, channels, height, width),
  )
  shifted = torch.gather(
    shifted,
    2,
    y[:, None, :, None].expand(batch, channels, height, width),
  )
  valid = valid_y[:, :, None] & valid_x[:, None, :]
  return torch.where(valid[:, None], shifted, torch.full_like(shifted, float("inf")))


def _replace_hidden_ball_with_background(
  depth: torch.Tensor,
  ball_pixels: torch.Tensor,
  sensor_hidden: torch.Tensor,
  max_depth: float,
) -> torch.Tensor:
  """Erase the ball without leaving a privileged invalid-depth silhouette."""
  hidden_ids = torch.where(sensor_hidden)[0]
  if hidden_ids.numel() == 0:
    return depth
  hidden_depth = depth[hidden_ids]
  hidden_ball = ball_pixels[hidden_ids]
  hidden_ball = F.max_pool2d(
    hidden_ball[:, None].to(dtype=depth.dtype), kernel_size=3, stride=1, padding=1
  ).bool()
  valid_background = torch.isfinite(hidden_depth) & (hidden_depth <= max_depth)
  background = torch.where(
    valid_background & ~hidden_ball,
    hidden_depth,
    torch.full_like(hidden_depth, -float("inf")),
  )
  fill = F.max_pool2d(background, kernel_size=31, stride=1, padding=15)
  fill = torch.where(torch.isfinite(fill), fill, torch.full_like(fill, max_depth))
  inpainted = torch.where(hidden_ball, fill, hidden_depth)
  result = depth.clone()
  result[hidden_ids] = inpainted
  return result


def _depth_noise_generator(
  env: ManagerBasedRlEnv, device: torch.device
) -> torch.Generator:
  """Return a per-env RNG isolated from the env's own (seeded) RNG stream.

  Sampling depth noise from the default/shared RNG would consume draws from
  the same stream the environment uses for reset randomization (ball spawn,
  terrain, friction, curriculum, ...). Since that stream is strictly
  sequential, doing so reshuffles every downstream "random" event even under
  a fixed seed, silently turning on/off domain randomization into a
  different physical rollout rather than the same rollout with noisier
  depth. A dedicated generator keeps depth-noise sampling from perturbing
  anything else.
  """
  key = (id(env), str(device))
  generator = _depth_noise_generators.get(key)
  if generator is None:
    generator = torch.Generator(device=device)
    generator.manual_seed(0x0EF7 ^ id(env))
    _depth_noise_generators[key] = generator
  return generator


def normalized_camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_depth: float = 0.2,
  max_depth: float = 3.0,
  near_noise_std: float = 0.0,
  far_noise_std: float = 0.0,
  far_distance: float = 2.0,
  dropout_probability: float = 0.0,
  mask_ball_when_sensor_hidden: bool = False,
  inpaint_ball_when_sensor_hidden: bool = False,
  depth_scale_range: tuple[float, float] = (1.0, 1.0),
  depth_bias_range: tuple[float, float] = (0.0, 0.0),
  crop_shift_x_pixels: int = 0,
  crop_shift_y_pixels: int = 0,
) -> torch.Tensor:
  """Return perspective depth as ``(B, 1, H, W)`` in the range ``[0, 1]``.

  Invalid pixels are mapped to one, the same value as measurements at or beyond
  ``max_depth``. This prevents missing depth from looking like an obstacle at the
  camera origin.

  The remaining arguments add domain randomization toward real stereo depth
  cameras (e.g. RealSense D435), whose ideal MuJoCo rendering has none of
  this. All default to zero/disabled, so existing callers are unaffected
  unless they opt in. The range-noise shape is adapted from Project-
  Instinct's InstinctMJ (``instinct_mj.utils.noise``, ``DepthSteroNoiseCfg``),
  simplified to this task's single-camera, functional observation style:

  - ``near_noise_std`` / ``far_noise_std`` / ``far_distance``: real stereo
    range noise grows with distance, so readings closer than ``far_distance``
    and farther than it get different Gaussian noise magnitudes instead of
    one fixed sigma.
  - ``dropout_probability``: independent per-pixel invalid readings (holes
    from IR-absorbing or low-texture surfaces).

  All sampling draws from a generator private to this observation term (see
  ``_depth_noise_generator``), not the env's shared/seeded RNG stream, so
  enabling or tuning this randomization never changes reset randomization,
  curriculum sampling, or any other seeded event's outcome.
  """
  if min_depth < 0.0:
    raise ValueError(f"min_depth must be non-negative, got {min_depth}")
  if max_depth <= min_depth:
    raise ValueError(
      f"max_depth must be greater than min_depth, got {min_depth}, {max_depth}"
    )
  if not 0.0 <= dropout_probability <= 1.0:
    raise ValueError(
      f"dropout_probability must be in [0, 1], got {dropout_probability}"
    )
  if near_noise_std < 0.0 or far_noise_std < 0.0:
    raise ValueError("near_noise_std and far_noise_std must be non-negative")
  if depth_scale_range[0] <= 0.0 or depth_scale_range[0] > depth_scale_range[1]:
    raise ValueError("depth_scale_range must be positive and ordered")
  if depth_bias_range[0] > depth_bias_range[1]:
    raise ValueError("depth_bias_range must be ordered")
  if crop_shift_x_pixels < 0 or crop_shift_y_pixels < 0:
    raise ValueError("crop shifts must be non-negative")
  if mask_ball_when_sensor_hidden and inpaint_ball_when_sensor_hidden:
    raise ValueError("hidden ball can be masked or inpainted, not both")

  sensor: CameraSensor = env.scene[sensor_name]
  depth = sensor.data.depth
  assert depth is not None, f"Camera '{sensor_name}' has no depth data"

  depth = depth.permute(0, 3, 1, 2)

  if mask_ball_when_sensor_hidden or inpaint_ball_when_sensor_hidden:
    segmentation = sensor.data.segmentation
    if segmentation is None:
      raise ValueError(
        f"Camera '{sensor_name}' requires segmentation to synchronize ball dropout"
      )
    visual_cache = vars(env).get("_football_masked_ball_visual")
    if not isinstance(visual_cache, dict):
      raise RuntimeError(
        "Synchronized depth ball dropout requires the masked-ball observation "
        "cache to be evaluated before the depth observation"
      )
    episode_hidden = visual_cache.get("episode_hidden")
    synthetic_hidden = visual_cache.get("synthetic_hidden")
    if not isinstance(episode_hidden, torch.Tensor) or not isinstance(
      synthetic_hidden, torch.Tensor
    ):
      raise RuntimeError("Masked-ball cache does not contain valid hidden-state masks")

    ball = env.scene["ball"]
    geom_ids = ball.indexing.geom_ids.to(device=segmentation.device)
    object_ids = segmentation[..., 0]
    object_types = segmentation[..., 1]
    is_geom = object_types == int(mujoco.mjtObj.mjOBJ_GEOM)
    ball_pixels = (object_ids[..., None] == geom_ids).any(dim=-1) & is_geom
    sensor_hidden = episode_hidden | synthetic_hidden
    if inpaint_ball_when_sensor_hidden:
      depth = _replace_hidden_ball_with_background(
        depth, ball_pixels, sensor_hidden, max_depth
      )
    else:
      hidden_ball_pixels = ball_pixels & sensor_hidden[:, None, None]
      depth = torch.where(
        hidden_ball_pixels[:, None],
        torch.full_like(depth, float("inf")),
        depth,
      )

  randomize_calibration = (
    depth_scale_range != (1.0, 1.0)
    or depth_bias_range != (0.0, 0.0)
    or crop_shift_x_pixels > 0
    or crop_shift_y_pixels > 0
  )
  if randomize_calibration:
    state = _depth_randomization_state(
      env,
      sensor_name,
      depth.device,
      depth_scale_range=depth_scale_range,
      depth_bias_range=depth_bias_range,
      crop_shift_x_pixels=crop_shift_x_pixels,
      crop_shift_y_pixels=crop_shift_y_pixels,
    )
    if crop_shift_x_pixels > 0 or crop_shift_y_pixels > 0:
      depth = _shift_depth_image(depth, state["crop_x"], state["crop_y"])
    depth = depth * state["scale"][:, None, None, None]
    depth = depth + state["bias"][:, None, None, None]

  randomize_depth = (
    near_noise_std > 0.0 or far_noise_std > 0.0 or dropout_probability > 0.0
  )
  generator = _depth_noise_generator(env, depth.device) if randomize_depth else None

  if near_noise_std > 0.0 or far_noise_std > 0.0:
    assert generator is not None
    finite = torch.isfinite(depth)
    near_mask = finite & (depth >= min_depth) & (depth <= far_distance)
    far_mask = finite & (depth > far_distance)
    if near_noise_std > 0.0:
      noise = torch.randn(depth.shape, generator=generator, device=depth.device)
      depth = torch.where(near_mask, depth + near_noise_std * noise, depth)
    if far_noise_std > 0.0:
      noise = torch.randn(depth.shape, generator=generator, device=depth.device)
      depth = torch.where(far_mask, depth + far_noise_std * noise, depth)

  valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
  if dropout_probability > 0.0:
    assert generator is not None
    keep = torch.rand(depth.shape, generator=generator, device=depth.device)
    valid = valid & (keep >= dropout_probability)
  normalized = torch.clamp(depth, min=min_depth, max=max_depth) / max_depth
  return torch.where(valid, normalized, torch.ones_like(normalized))


def normalized_camera_depth_frame(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_depth: float = 0.2,
  max_depth: float = 3.0,
  output_size: tuple[int, int] = (30, 40),
  near_noise_std: float = 0.0,
  far_noise_std: float = 0.0,
  far_distance: float = 2.0,
  dropout_probability: float = 0.0,
  mask_ball_when_sensor_hidden: bool = False,
  inpaint_ball_when_sensor_hidden: bool = False,
  depth_scale_range: tuple[float, float] = (1.0, 1.0),
  depth_bias_range: tuple[float, float] = (0.0, 0.0),
  crop_shift_x_pixels: int = 0,
  crop_shift_y_pixels: int = 0,
  frame_repeat_probability: float = 0.0,
) -> torch.Tensor:
  """Return one normalized, downsampled depth frame as ``(B, H, W)``.

  Returning a channel-free frame lets the observation manager stack temporal
  history along the channel dimension, producing ``(B, T, H, W)`` for a CNN.
  Domain randomization (see ``normalized_camera_depth``) is applied at the raw
  camera resolution, before downsampling, so it degrades the low-resolution
  pixels it is averaged into the same way a real depth hole would after the
  same downsampling.
  """
  depth = normalized_camera_depth(
    env,
    sensor_name,
    min_depth,
    max_depth,
    near_noise_std,
    far_noise_std,
    far_distance,
    dropout_probability,
    mask_ball_when_sensor_hidden,
    inpaint_ball_when_sensor_hidden,
    depth_scale_range,
    depth_bias_range,
    crop_shift_x_pixels,
    crop_shift_y_pixels,
  )
  if output_size[0] <= 0 or output_size[1] <= 0:
    raise ValueError(f"output_size must be positive, got {output_size}")
  if depth.shape[-2:] != output_size:
    depth = F.interpolate(depth, size=output_size, mode="area")
  frame = depth[:, 0]
  if not 0.0 <= frame_repeat_probability <= 1.0:
    raise ValueError("frame_repeat_probability must be in [0, 1]")
  if frame_repeat_probability > 0.0:
    cache_key = f"_depth_camera_randomization_{sensor_name}"
    state = vars(env).get(cache_key)
    if not isinstance(state, dict):
      state = _depth_randomization_state(
        env,
        sensor_name,
        frame.device,
        depth_scale_range=depth_scale_range,
        depth_bias_range=depth_bias_range,
        crop_shift_x_pixels=crop_shift_x_pixels,
        crop_shift_y_pixels=crop_shift_y_pixels,
      )
    last_frame = state["last_frame"]
    if last_frame.shape != frame.shape:
      last_frame = frame.clone()
      state["last_frame"] = last_frame
    generator = _depth_noise_generator(env, frame.device)
    repeat = (
      torch.rand(frame.shape[0], device=frame.device, generator=generator)
      < frame_repeat_probability
    ) & ~state["reset_mask"]
    delivered = torch.where(repeat[:, None, None], last_frame, frame)
    last_frame.copy_(delivered)
    frame = delivered
  return frame


def ball_auxiliary_target(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_position: float = 3.0,
) -> torch.Tensor:
  """Return normalized ball XY, planar distance, and exact camera visibility.

  Segmentation is used only to construct a supervised training target; it is not
  part of the Actor observation set or the exported policy inputs.
  """
  if max_position <= 0.0:
    raise ValueError(f"max_position must be positive, got {max_position}")

  sensor: CameraSensor = env.scene[sensor_name]
  segmentation = sensor.data.segmentation
  assert segmentation is not None, f"Camera '{sensor_name}' has no segmentation data"

  ball = env.scene["ball"]
  geom_ids = ball.indexing.geom_ids.to(device=segmentation.device)
  object_ids = segmentation[..., 0]
  object_types = segmentation[..., 1]
  is_geom = object_types == int(mujoco.mjtObj.mjOBJ_GEOM)
  ball_pixels = (object_ids[..., None] == geom_ids).any(dim=-1) & is_geom
  visible = ball_pixels.flatten(1).any(dim=1, keepdim=True)

  position = ball_pos_b(env)[:, :2]
  distance = torch.linalg.vector_norm(position, dim=1, keepdim=True)
  geometry = torch.cat((position, distance), dim=1)
  geometry = torch.clamp(geometry / max_position, min=-1.0, max=1.0)
  return torch.cat((geometry, visible.to(dtype=geometry.dtype)), dim=1)
