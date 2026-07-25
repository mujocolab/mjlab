from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerSoccerV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "ball_pos": obs[4:7],
            "unused_2": obs[7:-3],
            "goal_pos": obs[-3:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = 1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"] + np.array([0.0, 0.0, 0.03])
        pos_goal = o_d["goal_pos"]

        desired_z = 0.1 if np.linalg.norm(pos_curr[:2] - pos_ball[:2]) < 0.02 else 0.03

        to_left_of_goal = pos_ball[0] - pos_goal[0] < -0.05
        to_right_of_goal = pos_ball[0] - pos_goal[0] > 0.05

        offset = 0.03
        push_location = pos_ball + np.array([0.0, -offset, 0.0])
        if to_left_of_goal:
            push_location = pos_ball + np.array([-offset, 0.0, 0.0])
        elif to_right_of_goal:
            push_location = pos_ball + np.array([+offset, 0.0, 0.0])

        push_location[2] = desired_z

        if np.linalg.norm(pos_curr - push_location) > 0.01:
            return push_location
        return pos_ball
