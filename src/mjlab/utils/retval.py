from typing import Callable, TypeVar

T = TypeVar("T")


def retval(func: Callable[[], T]) -> T:
  """Invoke a function immediately and get its return value.

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
