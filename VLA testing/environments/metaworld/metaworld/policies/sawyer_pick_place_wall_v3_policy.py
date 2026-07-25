from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPickPlaceWallV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "puck_pos": obs[4:7],
            "unused_2": obs[7:-3],
            "goal_pos": obs[-3:],
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
        pos_puck = o_d["puck_pos"] + np.array([-0.005, 0, 0])
        pos_goal = o_d["goal_pos"]

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.015:
            return pos_puck + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04 and pos_puck[-1] < 0.03:
            return pos_puck + np.array([0.0, 0.0, 0.03])
        # Move to the goal
        else:
            # if wall is in the way of arm, straight up above the wall
            if (
                -0.15 <= pos_curr[0] <= 0.35
                and 0.60 <= pos_curr[1] <= 0.80
                and pos_curr[2] < 0.25
            ):
                return pos_curr + [0, 0, 1]
            # move towards the goal while staying above the wall
            elif (
                -0.15 <= pos_curr[0] <= 0.35
                and 0.60 <= pos_curr[1] <= 0.80
                and pos_curr[2] < 0.35
            ):
                return np.array([pos_goal[0], pos_goal[1], pos_curr[2]])
            # If not at the same Z height as the goal, move up to that plane
            elif abs(pos_curr[2] - pos_goal[2]) > 0.01:
                return np.array([pos_curr[0], pos_curr[1], pos_goal[2]])
            return pos_goal

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"]
        if (
            np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.015
            or abs(pos_curr[2] - pos_puck[2]) > 0.1
        ):
            return 0.0

        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 0.9
