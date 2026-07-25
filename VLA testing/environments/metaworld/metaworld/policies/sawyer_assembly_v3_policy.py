from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerAssemblyV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "wrench_pos": obs[4:7],
            "peg_pos": obs[-3:],
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
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        pos_peg = o_d["peg_pos"] + np.array([0.12, 0.0, 0.14])

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        # (For later) if lined up with peg, drop down on top of it
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
            return pos_peg + np.array([0.0, 0.0, -0.2])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench + np.array([0.0, 0.0, 0.03])
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
        # If XY error is greater than 0.02, place end effector above the peg
        else:
            return pos_peg

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        # pos_peg = o_d["peg_pos"] + np.array([0.12, 0.0, 0.14])

        if (
            np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02
            or abs(pos_curr[2] - pos_wrench[2]) > 0.12
        ):
            return 0.0
        # Until hovering over peg, keep hold of wrench
        else:
            return 0.6
