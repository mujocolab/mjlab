from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerHandInsertV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "obj_pos": obs[4:7],
            "goal_pos": obs[-3:],
            "unused_info": obs[7:-3],
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
        hand_pos = o_d["hand_pos"]
        obj_pos = o_d["obj_pos"]
        goal_pos = o_d["goal_pos"]

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(hand_pos[:2] - obj_pos[:2]) > 0.02:
            return obj_pos + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(hand_pos[2] - obj_pos[2]) > 0.05:
            return obj_pos + np.array([0.0, 0.0, 0.03])
        # If not above goal, move to be directly above goal
        elif np.linalg.norm(hand_pos[:2] - goal_pos[:2]) > 0.04:
            return np.array([goal_pos[0], goal_pos[1], hand_pos[2]])
        else:
            return goal_pos

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        hand_pos = o_d["hand_pos"]
        obj_pos = o_d["obj_pos"]
        if (
            np.linalg.norm(hand_pos[:2] - obj_pos[:2]) > 0.02
            or abs(hand_pos[2] - obj_pos[2]) > 0.1
        ):
            return 0.0
        else:
            return 0.65
