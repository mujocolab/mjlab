from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPlateSlideBackV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "puck_pos": obs[4:7],
            "unused_2": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_effort"] = -1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"] + np.array([0.0, -0.065, 0.025])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.01:
            return pos_puck + np.array([0.0, 0.0, 0.1])
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck
        elif pos_curr[1] > 0.7:
            return pos_curr + np.array([0.0, -0.1, 0.0])
        elif pos_curr[1] > 0.6:
            return np.array([0.15, 0.55, pos_curr[2]])
        else:
            return np.array([pos_curr[0] - 0.1, 0.55, pos_curr[2]])
