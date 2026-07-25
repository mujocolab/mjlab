from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerCoffeePullV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "mug_pos": obs[4:7],
            "unused_info": obs[7:-3],
            "target_pos": obs[-3:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["mug_pos"] + np.array([-0.005, 0.0, 0.05])

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06:
            return pos_mug + np.array([0.0, 0.0, 0.15])
        elif abs(pos_curr[2] - pos_mug[2]) > 0.02:
            return pos_mug
        else:
            return o_d["target_pos"]

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["mug_pos"] + np.array([0.01, 0.0, 0.05])

        if (
            np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06
            or abs(pos_curr[2] - pos_mug[2]) > 0.1
        ):
            return -1.0
        else:
            return 0.7
