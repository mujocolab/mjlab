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


class SawyerPushEnvV3(SawyerXYZEnv):
    """SawyerPushEnv.

    Motivation for V3:
        V1 was very difficult to solve because the observation didn't say where
        to move after reaching the puck.
    Changelog from V1 to V3:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    TARGET_RADIUS: float = 0.05

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-0.1, 0.8, 0.01)
        goal_high = (0.1, 0.9, 0.02)

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
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.02])

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
        return full_V3_path_for("sawyer_xyz/sawyer_push_v3.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        assert self.obj_init_pos is not None
        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": float(
                self.touching_main_object
                and (tcp_opened > 0)
                and (obj[2] - 0.02 > self.obj_init_pos[2])
            ),
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def fix_extreme_obj_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return np.array(
            [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]
        )

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(
            self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        )
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = np.concatenate([goal_pos[-3:-1], [self.obj_init_pos[-1]]])
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04
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
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            obj = obs[4:7]
            tcp_opened = obs[3]
            tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))
            target_to_obj = float(np.linalg.norm(obj - self._target_pos))
            target_to_obj_init = float(
                np.linalg.norm(self.obj_init_pos - self._target_pos)
            )

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=target_to_obj_init,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.05,
                xz_thresh=0.005,
                high_density=True,
            )
            reward = 2 * object_grasped

            if tcp_to_obj < 0.02 and tcp_opened > 0:
                reward += 1.0 + reward + 5.0 * in_place
            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                target_to_obj,
                object_grasped,
                in_place,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            del action
            del obs

            assert np.all(goal == self._get_site_pos("goal"))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            if reachDist < 0.05:
                pushRew = 1000 * (self.maxPushDist - pushDist) + c1 * (
                    np.exp(-(pushDist**2) / c2) + np.exp(-(pushDist**2) / c3)
                )
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew
            return float(reward), 0.0, 0.0, float(pushDist), 0.0, 0.0
