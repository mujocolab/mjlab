from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBasketballV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "ball_pos": obs[4:7],
            "hoop_x": obs[-3],
            "hoop_yz": obs[-2:],
            "unused_info": obs[7:-3],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = self._grab_effort(o_d)
        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"] + np.array([0.0, 0.0, 0.01])
        # X is given by hoop_pos
        # Y varies between .85 and .9, so we take avg
        # Z is constant at .35
        pos_hoop = np.array([o_d["hoop_x"], 0.875, 0.35])

        if np.linalg.norm(pos_curr[:2] - pos_ball[:2]) > 0.04:
            return pos_ball + np.array([0.0, 0.0, 0.3])
        elif abs(pos_curr[2] - pos_ball[2]) > 0.025:
            return pos_ball
        elif abs(pos_ball[2] - pos_hoop[2]) > 0.025:
            return np.array([pos_curr[0], pos_curr[1], pos_hoop[2]])
        else:
            return pos_hoop

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"]
        if (
            np.linalg.norm(pos_curr[:2] - pos_ball[:2]) > 0.04
            or abs(pos_curr[2] - pos_ball[2]) > 0.15
        ):
            return -1.0
        else:
            return 0.6
