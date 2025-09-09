"""Color manipulation utilities for RGB/HSV conversions and transformations."""

from typing import NamedTuple, Tuple


class RGB(NamedTuple):
  """RGB color with values in range [0, 1]."""

  r: float
  g: float
  b: float

  def to_tuple(self) -> Tuple[float, float, float]:
    """Convert to tuple representation."""
    return (self.r, self.g, self.b)


class RGBA(NamedTuple):
  """RGBA color with values in range [0, 1]."""

  r: float
  g: float
  b: float
  a: float

  @classmethod
  def from_rgb(cls, rgb: RGB, alpha: float = 1.0) -> "RGBA":
    """Create RGBA from RGB with specified alpha."""
    return cls(rgb.r, rgb.g, rgb.b, alpha)


class HSV(NamedTuple):
  """HSV color representation."""

  h: float  # Hue in range [0, 1].
  s: float  # Saturation in range [0, 1].
  v: float  # Value in range [0, 1].


def rgb_to_hsv(rgb: Tuple[float, float, float]) -> HSV:
  """Convert RGB to HSV color space.

  Args:
    rgb: RGB color tuple with values in range [0, 1].

  Returns:
    HSV color representation.
  """
  r, g, b = rgb
  max_val = max(r, g, b)
  min_val = min(r, g, b)
  delta = max_val - min_val

  # Calculate Value.
  v = max_val

  # Calculate Saturation.
  s = 0.0 if max_val == 0 else delta / max_val

  # Calculate Hue.
  if delta == 0:
    h = 0.0
  elif max_val == r:
    h = ((g - b) / delta) % 6
  elif max_val == g:
    h = (b - r) / delta + 2
  else:
    h = (r - g) / delta + 4

  h /= 6.0  # Normalize to [0, 1].

  return HSV(h, s, v)


def hsv_to_rgb(hsv: HSV) -> Tuple[float, float, float]:
  """Convert HSV to RGB color space.

  Args:
    hsv: HSV color representation.

  Returns:
    RGB color tuple with values in range [0, 1].
  """
  h, s, v = hsv.h, hsv.s, hsv.v

  i = int(h * 6)
  f = h * 6 - i
  p = v * (1 - s)
  q = v * (1 - f * s)
  t = v * (1 - (1 - f) * s)

  i %= 6

  if i == 0:
    return (v, t, p)
  elif i == 1:
    return (q, v, p)
  elif i == 2:
    return (p, v, t)
  elif i == 3:
    return (p, q, v)
  elif i == 4:
    return (t, p, v)
  else:
    return (v, p, q)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
  """Clamp a value between min and max."""
  return max(min_val, min(value, max_val))


def brand_ramp(
  base_rgb: Tuple[float, float, float], t: float, alpha: float = 1.0
) -> Tuple[float, float, float, float]:
  """Create a color ramp from a base RGB color using parameter t.

  This function creates a color variation by adjusting saturation and
  value based on the parameter t, useful for creating color gradients
  or variations of a brand color.

  Args:
    base_rgb: Base RGB color tuple with values in range [0, 1].
    t: Ramp parameter in range [0, 1], where 0 is darkest and 1 is brightest.
    alpha: Alpha transparency value in range [0, 1].

  Returns:
    RGBA color tuple representing the ramped color.

  Raises:
    ValueError: If t is not in range [0, 1].
  """
  if not 0 <= t <= 1:
    raise ValueError(f"Parameter t must be in range [0, 1], got {t}")

  hsv = rgb_to_hsv(base_rgb)

  # Ramp value: interpolate from 0.75 to 1.0 based on t.
  new_v = 0.75 + 0.25 * t

  # Ramp saturation: scale from 85% to 110% of original based on t.
  saturation_factor = 0.85 + 0.25 * t
  new_s = clamp(hsv.s * saturation_factor)

  new_hsv = HSV(hsv.h, new_s, new_v)
  r, g, b = hsv_to_rgb(new_hsv)

  return (r, g, b, alpha)


def darken_rgba(
  rgba: Tuple[float, float, float, float], factor: float = 0.85
) -> Tuple[float, float, float, float]:
  """Darken an RGBA color by a given factor.

  Args:
    rgba: RGBA color tuple with values in range [0, 1].
    factor: Darkening factor in range [0, 1], where 0 is black and 1 is original.

  Returns:
    RGBA color tuple with darkened color.

  Raises:
    ValueError: If factor is not in range [0, 1].
  """
  if not 0 <= factor <= 1:
    raise ValueError(f"Factor must be in range [0, 1], got {factor}")

  r, g, b, a = rgba
  return (r * factor, g * factor, b * factor, a)


def lighten_rgba(
  rgba: Tuple[float, float, float, float], factor: float = 0.15
) -> Tuple[float, float, float, float]:
  """Lighten an RGBA color by interpolating towards white.

  Args:
    rgba: RGBA color tuple with values in range [0, 1].
    factor: Lightening factor in range [0, 1], where 0 is original and 1 is white.

  Returns:
    RGBA color tuple with lightened color.

  Raises:
    ValueError: If factor is not in range [0, 1].
  """
  if not 0 <= factor <= 1:
    raise ValueError(f"Factor must be in range [0, 1], got {factor}")

  r, g, b, a = rgba
  return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor, a)


def adjust_saturation(
  rgb: Tuple[float, float, float], factor: float
) -> Tuple[float, float, float]:
  """Adjust the saturation of an RGB color.

  Args:
    rgb: RGB color tuple with values in range [0, 1].
    factor: Saturation factor where 0 is grayscale, 1 is original, >1 is more saturated.

  Returns:
    RGB color tuple with adjusted saturation.
  """
  hsv = rgb_to_hsv(rgb)
  new_s = clamp(hsv.s * factor)
  new_hsv = HSV(hsv.h, new_s, hsv.v)
  return hsv_to_rgb(new_hsv)
