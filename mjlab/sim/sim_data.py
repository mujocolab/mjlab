from typing import Any, Dict, Optional, Tuple
import torch
import warp as wp


class TorchArray:
  """Warp array that behaves like a torch.Tensor with shared memory.

  Enables seamless use of Warp arrays in PyTorch operations while
  maintaining zero-copy performance through memory sharing.
  """

  def __init__(self, wp_array: wp.array) -> None:
    """Initialize the tensor proxy with a Warp array."""
    self._wp_array = wp_array
    # A torch.Tensor that shares memory with the Warp array.
    self._tensor = wp.to_torch(wp_array)

  def __repr__(self) -> str:
    """Return string representation of the underlying tensor."""
    return repr(self._tensor)

  def __getitem__(self, idx: Any) -> Any:
    """Get item(s) from the tensor using standard indexing."""
    return self._tensor[idx]

  def __setitem__(self, idx: Any, value: Any) -> None:
    """Set item(s) in the tensor using standard indexing."""
    self._tensor[idx] = value

  def __getattr__(self, name: str) -> Any:
    """Delegate attribute access to the underlying tensor."""
    return getattr(self._tensor, name)

  @classmethod
  def __torch_function__(
    cls,
    func: Any,
    types: Tuple[type, ...],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Intercept torch.* function calls to unwrap TorchArray objects.

    This enables transparent use of TorchArray objects in torch functions
    by automatically unwrapping them to their underlying tensors.

    Args:
      func: The torch function being called
      types: Types of all arguments
      args: Positional arguments to the function
      kwargs: Keyword arguments to the function

    Returns:
      Result of the torch function call with unwrapped tensors
    """
    if kwargs is None:
      kwargs = {}

    # Only intercept when at least one argument is our proxy.
    if not any(issubclass(t, cls) for t in types):
      return NotImplemented

    def _unwrap(x: Any) -> Any:
      """Unwrap TorchArray objects to their underlying tensors."""
      return x._tensor if isinstance(x, cls) else x

    # Unwrap all TorchArray objects in args and kwargs.
    unwrapped_args = tuple(_unwrap(arg) for arg in args)
    unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

    return func(*unwrapped_args, **unwrapped_kwargs)

  def __add__(self, other: Any) -> Any:
    """Addition operation."""
    return self._tensor + other

  def __radd__(self, other: Any) -> Any:
    """Right addition operation."""
    return other + self._tensor

  def __sub__(self, other: Any) -> Any:
    """Subtraction operation."""
    return self._tensor - other

  def __rsub__(self, other: Any) -> Any:
    """Right subtraction operation."""
    return other - self._tensor

  def __mul__(self, other: Any) -> Any:
    """Multiplication operation."""
    return self._tensor * other

  def __rmul__(self, other: Any) -> Any:
    """Right multiplication operation."""
    return other * self._tensor

  def __truediv__(self, other: Any) -> Any:
    """Division operation."""
    return self._tensor / other

  def __rtruediv__(self, other: Any) -> Any:
    """Right division operation."""
    return other / self._tensor

  def __pow__(self, other: Any) -> Any:
    """Power operation."""
    return self._tensor**other

  def __rpow__(self, other: Any) -> Any:
    """Right power operation."""
    return other**self._tensor

  def __neg__(self) -> Any:
    """Unary negation operation."""
    return -self._tensor

  def __pos__(self) -> Any:
    """Unary positive operation."""
    return +self._tensor

  def __abs__(self) -> Any:
    """Absolute value operation."""
    return abs(self._tensor)

  def __eq__(self, other: Any) -> Any:
    """Equality comparison."""
    return self._tensor == other

  def __ne__(self, other: Any) -> Any:
    """Inequality comparison."""
    return self._tensor != other

  def __lt__(self, other: Any) -> Any:
    """Less than comparison."""
    return self._tensor < other

  def __le__(self, other: Any) -> Any:
    """Less than or equal comparison."""
    return self._tensor <= other

  def __gt__(self, other: Any) -> Any:
    """Greater than comparison."""
    return self._tensor > other

  def __ge__(self, other: Any) -> Any:
    """Greater than or equal comparison."""
    return self._tensor >= other


class WarpBridge:
  """Wraps mjwarp objects to expose Warp arrays as PyTorch tensors.

  Automatically converts Warp array attributes to TorchArray objects
  on access, enabling direct PyTorch operations on simulation data.
  """

  def __init__(self, struct: Any) -> None:
    super().__setattr__("struct", struct)

  def __getattr__(self, name: str) -> Any:
    """
    Get attribute from the wrapped data, wrapping Warp arrays as TorchArray.

    Args:
      name: Name of the attribute to access

    Returns:
      TorchArray if the attribute is a Warp array, otherwise the raw value
    """
    val = getattr(self.struct, name)
    if isinstance(val, wp.array):
      return TorchArray(val)
    if name in ["contact", "efc"]:
      return WarpBridge(val)
    return val

  def __setattr__(self, name: str, value: Any) -> None:
    """
    Set attribute on the wrapped data, handling tensor conversions.

    For existing Warp array fields, accepts TorchArray objects, PyTorch tensors,
    or other compatible values. Non-array fields are set normally.

    Args:
      name: Name of the attribute to set
      value: Value to set (TorchArray, torch.Tensor, or other)

    Raises:
      TypeError: If trying to set a Warp array field with an incompatible type
    """
    # Special case: setting 'data' attribute during initialization
    if name == "struct":
      super().__setattr__(name, value)
      return

    # Handle assignments to existing wp.array fields
    if hasattr(self.struct, name) and isinstance(getattr(self.struct, name), wp.array):
      if isinstance(value, TorchArray):
        new_wp_array = value._wp_array
      elif isinstance(value, torch.Tensor):
        new_wp_array = wp.from_torch(value)
      else:
        raise TypeError(
          f"Cannot set Warp array field '{name}' from {type(value)}. "
          f"Expected TorchArray or torch.Tensor."
        )
      setattr(self.struct, name, new_wp_array)
    else:
      # For non-array fields, set the attribute on the underlying struct object
      setattr(self.struct, name, value)

  def __repr__(self) -> str:
    """Return string representation of the wrapped struct."""
    return f"WarpTensor({repr(self.struct)})"
