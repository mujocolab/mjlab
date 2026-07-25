"""A set of reward utilities written by the authors of dm_control."""
from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


SIGMOID_TYPE = Literal[
    "gaussian",
    "hyperbolic",
    "long_tail",
    "reciprocal",
    "cosine",
    "linear",
    "quadratic",
    "tanh_squared",
]

X = TypeVar("X", float, npt.NDArray, np.floating)


def _sigmoids(x: X, value_at_1: float, sigmoid: SIGMOID_TYPE) -> X:
    """Maps the input to values between 0 and 1 using a specified sigmoid function. Returns 1 when the input is 0, between 0 and 1 otherwise.

    Args:
        x: The input.
        value_at_1: The output value when `x` == 1. Must be between 0 and 1.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.

    Returns:
        The input mapped to values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
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
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x: X,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float | np.floating[Any] = 0.0,
    sigmoid: SIGMOID_TYPE = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> X:
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: The input.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError(f"`margin` must be non-negative. Current value: {margin}")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value.item() if np.isscalar(x) else value


def inverse_tolerance(
    x: X,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: SIGMOID_TYPE = "reciprocal",
) -> X:
    """Returns 0 when `x` falls inside the bounds, between 1 and 0 otherwise.

    Args:
        x: The input
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    bound = tolerance(
        x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=0
    )
    return 1 - bound


def rect_prism_tolerance(
    curr: npt.NDArray[np.float_],
    zero: npt.NDArray[np.float_],
    one: npt.NDArray[np.float_],
) -> float:
    """Computes a reward if curr is inside a rectangular prism region.

    All inputs are 3D points with shape (3,).

    Args:
        curr: The point that the prism reward region is being applied for.
        zero: The diagonal opposite corner of the prism with reward 0.
        one: The corner of the prism with reward 1.

    Returns:
        A reward if curr is inside the prism, 1.0 otherwise.
    """

    def in_range(a, b, c):
        return float(b <= a <= c) if c >= b else float(c <= a <= b)

    in_prism = (
        in_range(curr[0], zero[0], one[0])
        and in_range(curr[1], zero[1], one[1])
        and in_range(curr[2], zero[2], one[2])
    )
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
    else:
        return 1.0


def hamacher_product(a: float, b: float) -> float:
    """Returns the hamacher (t-norm) product of a and b.

    Computes (a * b) / ((a + b) - (a * b)).

    Args:
        a: 1st term of the hamacher product.
        b: 2nd term of the hamacher product.

    Returns:
        The hammacher product of a and b

    Raises:
        ValueError: a and b must range between 0 and 1
    """
    if not ((0.0 <= a <= 1.0) and (0.0 <= b <= 1.0)):
        raise ValueError(f"a ({b}) and b ({b}) must range between 0 and 1")

    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0.0 <= h_prod <= 1.0
    return h_prod
