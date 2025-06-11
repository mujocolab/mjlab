"""A port of dm_control.utils.rewards to JAX."""

import warnings

import jax.numpy as jp

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
  if sigmoid in ("cosine", "linear", "quadratic"):
    if not 0 <= value_at_1 < 1:
      raise ValueError(
        f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
      )
  else:
    if not 0 < value_at_1 < 1:
      raise ValueError(
        f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
      )

  if sigmoid == "gaussian":
    scale = jp.sqrt(-2 * jp.log(value_at_1))
    return jp.exp(-0.5 * (x * scale) ** 2)

  elif sigmoid == "hyperbolic":
    scale = jp.arccosh(1 / value_at_1)
    return 1 / jp.cosh(x * scale)

  elif sigmoid == "long_tail":
    scale = jp.sqrt(1 / value_at_1 - 1)
    return 1 / ((x * scale) ** 2 + 1)

  elif sigmoid == "reciprocal":
    scale = 1 / value_at_1 - 1
    return 1 / (abs(x) * scale + 1)

  elif sigmoid == "cosine":
    scale = jp.arccos(2 * value_at_1 - 1) / jp.pi
    scaled_x = x * scale
    with warnings.catch_warnings():
      warnings.filterwarnings(
        action="ignore", message="invalid value encountered in cos"
      )
      cos_pi_scaled_x = jp.cos(jp.pi * scaled_x)
    return jp.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

  elif sigmoid == "linear":
    scale = 1 - value_at_1
    scaled_x = x * scale
    return jp.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

  elif sigmoid == "quadratic":
    scale = jp.sqrt(1 - value_at_1)
    scaled_x = x * scale
    return jp.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

  elif sigmoid == "tanh_squared":
    scale = jp.arctanh(jp.sqrt(1 - value_at_1))
    return 1 - jp.tanh(x * scale) ** 2

  else:
    raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
  x: jp.ndarray,
  bounds: tuple[float, float] = (0.0, 0.0),
  margin: float = 0.0,
  sigmoid: str = "gaussian",
  value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> jp.ndarray:
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

  Args:
    x: A jax numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded at
      one or both ends, or they can be equal to one another if the target value
      is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds. * If `margin == 0` then the output will be 0 for
      all values of `x` outside of `bounds`. * If `margin > 0` then the output
      will decrease sigmoidally with increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
      'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.

  Returns:
    A jax numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError("Lower bound must be <= upper bound.")
  if margin < 0:
    raise ValueError("`margin` must be non-negative.")

  in_bounds = jp.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = jp.where(in_bounds, 1.0, 0.0)
  else:
    d = jp.where(x < lower, lower - x, x - upper) / margin
    value = jp.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  return value
