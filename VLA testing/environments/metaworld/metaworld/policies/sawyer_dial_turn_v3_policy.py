from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDialTurnV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_gripper_open": obs[3],
            "dial_pos": obs[4:7],
            "extra_info": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_pow": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_pow"] = 1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        hand_pos = o_d["hand_pos"]
        dial_pos = o_d["dial_pos"] + np.array([0.05, 0.02, 0.09])

        if np.linalg.norm(hand_pos[:2] - dial_pos[:2]) > 0.02:
            return np.array([*dial_pos[:2], 0.2])
        if abs(hand_pos[2] - dial_pos[2]) > 0.02:
            return dial_pos
        return dial_pos + np.array([-0.05, 0.005, 0.0])
