from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerPickOutOfHoleEnvV3(SawyerXYZEnv):
    _TARGET_RADIUS: float = 0.02

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, -0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.75, 0.02)
        obj_high = (0, 0.75, 0.02)
        goal_low = (-0.1, 0.5, 0.15)
        goal_high = (0.1, 0.6, 0.3)

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
            "obj_init_pos": np.array([0, 0.6, 0.0]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.0, 0.6, 0.2])
        self.obj_init_pos = None
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_pick_out_of_hole.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            grasp_success,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(grasp_success)

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

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        _site_config = [("goal", self.init_right_pad)]
        if self.obj_init_pos is not None:
            _site_config[0] = ("goal", self.obj_init_pos)
        return _site_config

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        pos_obj, pos_goal = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_obj[:2] - pos_goal[:2]) < 0.15:
            pos_obj, pos_goal = np.split(self._get_state_rand_vec(), 2)

        self.obj_init_pos = pos_obj
        self._set_obj_xyz(self.obj_init_pos)
        self._target_pos = pos_goal
        self.model.site("goal").pos = self._target_pos

        self.liftThresh = 0.11
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh
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
            gripper = self.tcp_center

            obj_to_target = float(np.linalg.norm(obj - self._target_pos))
            tcp_to_obj = float(np.linalg.norm(obj - gripper))
            in_place_margin = float(
                np.linalg.norm(self.obj_init_pos - self._target_pos)
            )

            threshold = 0.03
            # floor is a 3D funnel centered on the initial object pos
            radius = float(np.linalg.norm(gripper[:2] - self.obj_init_pos[:2]))
            if radius <= threshold:
                floor = 0.0
            else:
                floor = 0.015 * np.log(radius - threshold) + 0.15
            # prevent the hand from running into cliff edge by staying above floor
            above_floor = (
                1.0
                if gripper[2] >= floor
                else reward_utils.tolerance(
                    max(floor - gripper[2], 0.0),
                    bounds=(0.0, 0.01),
                    margin=0.02,
                    sigmoid="long_tail",
                )
            )
            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.02,
                xz_thresh=0.03,
                desired_gripper_effort=0.1,
                high_density=True,
            )
            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.02),
                margin=in_place_margin,
                sigmoid="long_tail",
            )
            reward = reward_utils.hamacher_product(object_grasped, in_place)

            near_object = tcp_to_obj < 0.04
            pinched_without_obj = obs[3] < 0.33
            lifted = obj[2] - 0.02 > self.obj_init_pos[2]
            # Increase reward when properly grabbed obj
            grasp_success = near_object and lifted and not pinched_without_obj
            if grasp_success:
                reward += 1.0 + 5.0 * reward_utils.hamacher_product(
                    in_place, above_floor
                )
            # Maximize reward on success
            if obj_to_target < self.TARGET_RADIUS:
                reward = 10.0

            return (
                reward,
                tcp_to_obj,
                grasp_success,
                obj_to_target,
                object_grasped,
                in_place,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])
            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50

            tolerance = 0.01
            self.pickCompleted = objPos[2] >= (heightTarget - tolerance)

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

            hScale = 100
            if self.pickCompleted and not objDropped:
                pickRew = hScale * (heightTarget - self.objHeight + 0.02)
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * (
                    min(heightTarget, objPos[2]) - self.objHeight + 0.02
                )
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
            else:
                placeRew, placingDist = [0, placingDist]

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew

            return float(reward), 0.0, 0.0, float(placingDist), 0.0, 0.0
