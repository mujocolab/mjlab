from typing import Callable, TypeVar

T = TypeVar("T")


def immediately(func: Callable[[], T]) -> T:
  """Immediately invoke a function and return its result. Useful as a decorator
  for factory functions.

  ```python
  # This:
  @immediately
  def MY_CONFIG() -> SomeConfigType:
      return SomeConfigType()

  # is equivalent to:
  MY_CONFIG = SomeConfigType()
  ```
  """
  return func()
