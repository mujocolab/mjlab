from typing import Callable, TypeVar

T = TypeVar("T")


def retval(func: Callable[[], T]) -> T:
  """Invoke a function at module load time and use its return value as a constant.

  Useful as a decorator for factory functions that should be evaluated once
  at import time to create module-level constants.

  ```python
  # This:
  @retval
  def MY_CONFIG() -> SomeConfigType:
      return SomeConfigType()

  # is equivalent to:
  MY_CONFIG = SomeConfigType()
  ```
  """
  return func()
