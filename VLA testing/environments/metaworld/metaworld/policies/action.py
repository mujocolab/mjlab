from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class Action:
    """Represents an action to be taken in an environment.

    Once initialized, fields can be assigned as if the action
    is a dictionary. Once filled, the corresponding array is
    available as an instance variable.
    """

    def __init__(self, structure: dict[str, npt.NDArray[Any] | int]) -> None:
        """Action.

        Args:
            structure: Map from field names to output array indices
        """
        self._structure = structure
        self.array = np.zeros(len(self), dtype=np.float32)

    def __len__(self) -> int:
        return sum(
            [1 if isinstance(idx, int) else len(idx) for idx in self._structure.items()]
        )

    def __getitem__(self, key) -> npt.NDArray[np.float32]:
        assert key in self._structure, (
            "This action's structure does not contain %s" % key
        )
        return self.array[self._structure[key]]

    def __setitem__(self, key: str, value) -> None:
        assert key in self._structure, f"This action's structure does not contain {key}"
        self.array[self._structure[key]] = value
