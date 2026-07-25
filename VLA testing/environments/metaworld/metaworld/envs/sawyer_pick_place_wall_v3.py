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


class SawyerPickPlaceWallEnvV3(SawyerXYZEnv):
    """SawyerPickPlaceWallEnv.

    Motivation for V3:
        V1 was difficult to solve because the observation didn't say where
        to move after picking up the puck.
    Changelog from V1 to V3:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/24/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/24/20) Separated pick-place-wall into from
          reach-push-pick-place-wall.
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

        self.goal = np.array([0.05, 0.8, 0.2])

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
        return full_V3_path_for("sawyer_xyz/sawyer_pick_place_wall_v3.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        assert self.obj_init_pos is not None
        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (obj[2] - 0.02 > self.obj_init_pos[2])
        )
        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.geom("objGeom").xpos

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.geom("objGeom").xmat.reshape(3, 3)
        ).as_quat()

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com("obj")[:2] - self.data.geom("objGeom").xpos[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1], self.data.geom("objGeom").xpos[-1]]

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config["obj_init_pos"])
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
        self.target_rewards = [
            1000 * self.maxPlacingDist + 1000 * 2,
            1000 * self.maxReachDist + 1000 * 2,
            1000 * self.maxPushDist + 1000 * 2,
        ]

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.obj_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened: float = obs[3]
            midpoint = np.array([self._target_pos[0], 0.77, 0.25])
            target = self._target_pos

            tcp_to_obj = float(np.linalg.norm(obj - tcp))

            in_place_scaling = np.array([1.0, 1.0, 3.0])
            obj_to_midpoint = float(np.linalg.norm((obj - midpoint) * in_place_scaling))
            obj_to_midpoint_init = float(
                np.linalg.norm((self.obj_init_pos - midpoint) * in_place_scaling)
            )

            obj_to_target = float(np.linalg.norm(obj - target))
            obj_to_target_init = float(np.linalg.norm(self.obj_init_pos - target))

            in_place_part1 = reward_utils.tolerance(
                obj_to_midpoint,
                bounds=(0, _TARGET_RADIUS),
                margin=obj_to_midpoint_init,
                sigmoid="long_tail",
            )

            in_place_part2 = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=obj_to_target_init,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(
                action=action,
                obj_pos=obj,
                obj_radius=0.015,
                pad_success_thresh=0.05,
                object_reach_radius=0.01,
                xz_thresh=0.005,
                high_density=False,
            )

            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place_part1
            )
            reward = in_place_and_object_grasped

            if (
                tcp_to_obj < 0.02
                and (tcp_opened > 0)
                and (obj[2] - 0.015 > self.obj_init_pos[2])
            ):
                reward = in_place_and_object_grasped + 1.0 + 4.0 * in_place_part1
                if obj[1] > 0.75:
                    reward = (
                        in_place_and_object_grasped + 1.0 + 4.0 + 3.0 * in_place_part2
                    )

            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                float(np.linalg.norm(obj - target)),
                object_grasped,
                in_place_part2,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos

            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos("goal"))

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50

            def pickCompletionCriteria():
                tolerance = 0.01
                return objPos[2] >= (heightTarget - tolerance)

            self.pickCompleted = pickCompletionCriteria()

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

            hScale = 100
            if self.pickCompleted and not objDropped:
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not objDropped
            if cond:
                placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                    np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
                )
                placeRew = max(placeRew, 0)
                placeRew, placingDist = [placeRew, placingDist]
            else:
                placeRew, placingDist = [0, placingDist]

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew

            return float(reward), 0.0, 0.0, float(placingDist), 0.0, 0.0
