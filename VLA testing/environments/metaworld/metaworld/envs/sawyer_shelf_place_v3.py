from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerShelfPlaceEnvV3(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        goal_low = (-0.1, 0.8, 0.299)
        goal_high = (0.1, 0.9, 0.301)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.5, 0.019)
        obj_high = (0.1, 0.6, 0.021)

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
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.85, 0.301], dtype=np.float32)
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.num_resets = 0

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_shelf_placing.xml")

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
            in_place,
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
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def adjust_initObjPos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com("obj")[:2] - self.data.geom("objGeom").xpos[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return np.array([adjustedPos[0], adjustedPos[1], self.get_body_com("obj")[-1]])

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        base_shelf_pos = goal_pos - np.array([0, 0, 0, 0, 0, 0.3])
        self.obj_init_pos = np.concatenate(
            (base_shelf_pos[:2], [self.obj_init_pos[-1]])
        )

        self.model.body("shelf").pos = base_shelf_pos[-3:]
        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self.model.site("goal").pos + self.model.body("shelf").pos

        assert self.obj_init_pos is not None
        self._set_obj_xyz(self.obj_init_pos)
        assert self._target_pos is not None
        self._set_pos_site("goal", self._target_pos)

        self.liftThresh = 0.04
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
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]
            target = self._target_pos

            obj_to_target = float(np.linalg.norm(obj - target))
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            in_place_margin = np.linalg.norm(self.obj_init_pos - target)

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(
                action=action,
                obj_pos=obj,
                obj_radius=0.02,
                pad_success_thresh=0.05,
                object_reach_radius=0.01,
                xz_thresh=0.01,
                high_density=False,
            )
            reward = reward_utils.hamacher_product(object_grasped, in_place)

            if (
                0.0 < obj[2] < 0.24
                and (target[0] - 0.15 < obj[0] < target[0] + 0.15)
                and ((target[1] - 3 * _TARGET_RADIUS) < obj[1] < target[1])
            ):
                z_scaling = (0.24 - obj[2]) / 0.24
                y_scaling = (obj[1] - (target[1] - 3 * _TARGET_RADIUS)) / (
                    3 * _TARGET_RADIUS
                )
                bound_loss = reward_utils.hamacher_product(y_scaling, z_scaling)
                in_place = np.clip(in_place - bound_loss, 0.0, 1.0)

            if (
                (0.0 < obj[2] < 0.24)
                and (target[0] - 0.15 < obj[0] < target[0] + 0.15)
                and (obj[1] > target[1])
            ):
                in_place = 0.0

            if (
                tcp_to_obj < 0.025
                and (tcp_opened > 0)
                and (obj[2] - 0.01 > self.obj_init_pos[2])
            ):
                reward += 1.0 + 5.0 * in_place

            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
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
            placingGoal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)

            placingDist = np.linalg.norm(objPos - placingGoal)

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
