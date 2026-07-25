from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move


class SawyerButtonPressWallV3Policy(Policy):
    @staticmethod
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "hand_closed": obs[3],
            "button_pos": obs[4:7],
            "unused_info": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=15.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_button = o_d["button_pos"] + np.array([0.0, 0.0, 0.04])

        if abs(pos_curr[0] - pos_button[0]) > 0.02:
            return np.array([pos_button[0], pos_curr[1], 0.3])
        elif pos_button[1] - pos_curr[1] > 0.09:
            return np.array([pos_button[0], pos_button[1], 0.3])
        elif abs(pos_curr[2] - pos_button[2]) > 0.02:
            return pos_button + np.array([0.0, -0.05, 0.0])
        else:
            return pos_button + np.array([0.0, -0.02, 0.0])

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_button = o_d["button_pos"] + np.array([0.0, 0.0, 0.04])

        if (
            abs(pos_curr[0] - pos_button[0]) > 0.02
            or pos_button[1] - pos_curr[1] > 0.09
            or abs(pos_curr[2] - pos_button[2]) > 0.02
        ):
            return 1.0
        else:
            return -1.0
