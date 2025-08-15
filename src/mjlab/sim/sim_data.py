from typing import Any, Dict, Optional, Tuple, TypeVar, Generic
import torch
import warp as wp

T = TypeVar("T")


class TorchArray:
  """Warp array that behaves like a torch.Tensor with shared memory.

  Enables seamless use of Warp arrays in PyTorch operations while
  maintaining zero-copy performance through memory sharing.
  """

  def __init__(self, wp_array: wp.array) -> None:
    """Initialize the tensor proxy with a Warp array."""
    self._wp_array = wp_array
    self._tensor = wp.to_torch(wp_array)
    self._is_cuda = not self._wp_array.device.is_cpu
    self._torch_stream = self._setup_stream()

  def _setup_stream(self) -> Optional[torch.cuda.Stream]:
    """Setup appropriate stream for the device."""
    if not self._is_cuda:
      return None

    try:
      warp_stream = wp.get_stream(self._wp_array.device)
      return torch.cuda.ExternalStream(warp_stream.cuda_stream)
    except Exception as e:
      # Fallback to default stream if external stream creation fails.
      print(f"Warning: Could not create external stream: {e}")
      return torch.cuda.current_stream(self._tensor.device)

  @property
  def wp_array(self) -> wp.array:
    return self._wp_array

  def __repr__(self) -> str:
    """Return string representation of the underlying tensor."""
    return repr(self._tensor)

  def __getitem__(self, idx: Any) -> Any:
    """Get item(s) from the tensor using standard indexing."""
    return self._tensor[idx]

  def __setitem__(self, idx: Any, value: Any) -> None:
    """Set item(s) in the tensor using standard indexing."""
    if self._is_cuda and self._torch_stream is not None:
      with torch.cuda.stream(self._torch_stream):
        self._tensor[idx] = value
    else:
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
    """Intercept torch.* function calls to unwrap TorchArray objects."""
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

  # Arithmetic operators.

  def __add__(self, other: Any) -> Any:
    return self._tensor + other

  def __radd__(self, other: Any) -> Any:
    return other + self._tensor

  def __sub__(self, other: Any) -> Any:
    return self._tensor - other

  def __rsub__(self, other: Any) -> Any:
    return other - self._tensor

  def __mul__(self, other: Any) -> Any:
    return self._tensor * other

  def __rmul__(self, other: Any) -> Any:
    return other * self._tensor

  def __truediv__(self, other: Any) -> Any:
    return self._tensor / other

  def __rtruediv__(self, other: Any) -> Any:
    return other / self._tensor

  def __pow__(self, other: Any) -> Any:
    return self._tensor**other

  def __rpow__(self, other: Any) -> Any:
    return other**self._tensor

  def __neg__(self) -> Any:
    return -self._tensor

  def __pos__(self) -> Any:
    return +self._tensor

  def __abs__(self) -> Any:
    return abs(self._tensor)

  # Comparison operators.

  def __eq__(self, other: Any) -> Any:
    return self._tensor == other

  def __ne__(self, other: Any) -> Any:
    return self._tensor != other

  def __lt__(self, other: Any) -> Any:
    return self._tensor < other

  def __le__(self, other: Any) -> Any:
    return self._tensor <= other

  def __gt__(self, other: Any) -> Any:
    return self._tensor > other

  def __ge__(self, other: Any) -> Any:
    return self._tensor >= other


def _contains_warp_arrays(obj: Any) -> bool:
  """Check if an object or its attributes contain any Warp arrays."""
  if isinstance(obj, wp.array):
    return True

  # Check if it's a struct-like object with attributes
  if hasattr(obj, "__dict__"):
    return any(
      isinstance(getattr(obj, attr), wp.array)
      for attr in dir(obj)
      if not attr.startswith("_")
    )

  return False


class WarpBridge(Generic[T]):
  """Wraps mjwarp objects to expose Warp arrays as PyTorch tensors.

  Automatically converts Warp array attributes to TorchArray objects
  on access, enabling direct PyTorch operations on simulation data.
  Recursively wraps nested structures that contain Warp arrays.
  """

  def __init__(self, struct: T) -> None:
    super().__setattr__("_struct", struct)
    super().__setattr__("_wrapped_cache", {})

  def __getattr__(self, name: str) -> Any:
    """Get attribute from the wrapped data, wrapping Warp arrays as TorchArray."""
    # Check cache first to avoid recreating wrappers.
    if name in self._wrapped_cache:
      return self._wrapped_cache[name]

    val = getattr(self._struct, name)

    # Wrap Warp arrays.
    if isinstance(val, wp.array):
      wrapped = TorchArray(val)
      self._wrapped_cache[name] = wrapped
      return wrapped

    # Recursively wrap nested structures that contain Warp arrays.
    if _contains_warp_arrays(val):
      wrapped = WarpBridge(val)
      self._wrapped_cache[name] = wrapped
      return wrapped

    return val

  def __setattr__(self, name: str, value: Any) -> None:
    """Set attribute on the wrapped data, handling tensor conversions."""
    # Special case: setting internal attributes during initialization
    if name in ("_struct", "_wrapped_cache"):
      super().__setattr__(name, value)
      return

    # Clear cache for this attribute since we're modifying it
    if name in self._wrapped_cache:
      del self._wrapped_cache[name]

    # Handle assignments to existing wp.array fields
    if hasattr(self._struct, name) and isinstance(
      getattr(self._struct, name), wp.array
    ):
      if isinstance(value, TorchArray):
        new_wp_array = value.wp_array
      elif isinstance(value, torch.Tensor):
        new_wp_array = wp.from_torch(value)
      else:
        raise TypeError(
          f"Cannot set Warp array field '{name}' from {type(value)}. "
          f"Expected TorchArray or torch.Tensor."
        )
      setattr(self._struct, name, new_wp_array)
    else:
      # For non-array fields, set the attribute on the underlying struct object
      setattr(self._struct, name, value)

  def __repr__(self) -> str:
    """Return string representation of the wrapped struct."""
    return f"WarpBridge({repr(self._struct)})"

  @property
  def struct(self) -> T:
    """Access the underlying wrapped struct."""
    return self._struct
