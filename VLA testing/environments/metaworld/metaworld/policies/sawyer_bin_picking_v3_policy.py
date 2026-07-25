from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBinPickingV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "cube_pos": obs[4:7],
            "extra_info": obs[7:],
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
        pos_cube = o_d["cube_pos"] + np.array([0.0, 0.0, 0.03])
        pos_bin = np.array([0.12, 0.7, 0.02])

        # This forces the scripted policy to pretend like the cube is located
        # more centrally in the bin (in Y direction). When the fingers close,
        # they'll drag the cube so that it's no longer located near an edge.
        # This ensures that the fingers don't get caught outside of the bin.
        pos_cube[1] = max(0.675, min(pos_cube[1], 0.725))

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.02:
            return pos_cube + np.array([0.0, 0.0, 0.15])
        elif abs(pos_curr[2] - pos_cube[2]) > 0.01:
            return pos_cube
        elif np.linalg.norm(pos_curr[:2] - pos_bin[:2]) > 0.02:
            if pos_curr[2] < 0.15:
                return pos_curr + np.array([0.0, 0.0, 0.1])
            return np.array([*pos_bin[:2], 0.18])
        else:
            return pos_bin

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_cube = o_d["cube_pos"] + np.array([0.0, 0.0, 0.03])

        # See note above in `_desired_pos`
        pos_cube[1] = max(0.675, min(pos_cube[1], 0.725))

        if (
            np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.02
            or abs(pos_curr[2] - pos_cube[2]) > 0.02
        ):
            return -1.0
        else:
            return 0.6
