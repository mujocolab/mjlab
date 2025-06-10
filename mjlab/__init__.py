from typing import Union

# Import tasks to register them.
from mjlab import control_suite

# Import configs and add them to ConfigUnion.
from mjlab.control_suite.cartpole import CartpoleConfig

TaskConfigUnion = Union[CartpoleConfig,]


__all__ = (
    "control_suite",
    "TaskConfigUnion",
)
