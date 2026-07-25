from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerReachWallEnvV3(SawyerXYZEnv):
    """SawyerReachWallEnv.

    Motivation for V3:
        V1 was difficult to solve since the observations didn't say where
        to move (where to reach).
    Changelog from V1 to V3:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/17/20) Separated reach from reach-push-pick-place.
        - (6/17/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        goal_low = (-0.05, 0.85, 0.05)
        goal_high = (0.05, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.6, 0.015)
        obj_high = (0.05, 0.65, 0.015)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.reward_function_version = reward_function_version

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.05, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self.num_resets = 0

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_reach_wall_v3.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, tcp_to_object, in_place = self.compute_reward(action, obs)
        success = float(tcp_to_object <= 0.05)

        info = {
            "success": success,
            "near_object": 0.0,
            "grasp_success": 0.0,
            "grasp_reward": 0.0,
            "in_place_reward": in_place,
            "obj_to_target": tcp_to_object,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        self.liftThresh = 0.04
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh

        self.maxReachDist = np.linalg.norm(self.init_tcp - np.array(self._target_pos))
        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )
        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - np.array(self._target_pos)
            )
            + self.heightTarget
        )

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            # obj = obs[4:7]
            # tcp_opened = obs[3]
            target = self._target_pos

            tcp_to_target = float(np.linalg.norm(tcp - target))
            # obj_to_target = float(np.linalg.norm(obj - target))

            in_place_margin = float(np.linalg.norm(self.hand_init_pos - target))
            in_place = reward_utils.tolerance(
                tcp_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            return (10 * in_place, tcp_to_target, in_place)
        else:
            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            reachRew = c1 * (self.maxReachDist - reachDist) + c1 * (
                np.exp(-(reachDist**2) / c2) + np.exp(-(reachDist**2) / c3)
            )
            reachRew = max(reachRew, 0)
            reward = reachRew

            return float(reward), float(reachDist), 0.0
