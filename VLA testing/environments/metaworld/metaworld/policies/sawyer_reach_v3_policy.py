from __future__ import annotations

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerReachV3Policy(Policy):
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

        action["delta_pos"] = move(o_d["hand_pos"], to_xyz=o_d["goal_pos"], p=5.0)
        action["grab_effort"] = 0.0

        return action.array
