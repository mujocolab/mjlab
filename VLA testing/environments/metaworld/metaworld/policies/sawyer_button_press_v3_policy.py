from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move


class SawyerButtonPressV3Policy(Policy):
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
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = 0.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_button = o_d["button_pos"] + np.array([0.0, 0.0, -0.07])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button
        if not np.all(
            np.isclose(
                np.array([hand_x, hand_z]),
                np.array([button_initial_x, button_initial_z]),
                atol=0.02,
            )
        ):
            pos_button[1] = pos_curr[1] - 0.1
            return pos_button
        # if the hand is aligned with the button, push the button in, by
        # increasing the hand's y position
        pos_button[1] += 0.02

        return pos_button
