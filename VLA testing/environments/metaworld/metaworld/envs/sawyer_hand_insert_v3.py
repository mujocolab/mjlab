from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerHandInsertEnvV3(SawyerXYZEnv):
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
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.05)
        obj_high = (0.1, 0.7, 0.05)
        goal_low = (-0.04, 0.8, -0.0201)
        goal_high = (0.04, 0.88, -0.0199)

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
            "obj_init_pos": np.array([0, 0.6, 0.05]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.84, -0.08], dtype=np.float32)
        self.obj_init_pos = self.init_config["obj_init_pos"]
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
        return full_V3_path_for("sawyer_xyz/sawyer_table_with_hole.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        assert self.obj_init_pos is not None

        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.05),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": float(
                self.touching_main_object
                and (tcp_open > 0)
                and (obj[2] - 0.02 > self.obj_init_pos[2])
            ),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.model.geom("objGeom").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.objHeight = self.get_body_com("obj")[2]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:
            goal_pos = self._get_state_rand_vec()
        assert self.obj_init_pos is not None
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self._target_pos = goal_pos[-3:]

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        assert self._target_pos is not None and self.hand_init_pos is not None
        self.maxReachDist = np.abs(self.hand_init_pos[-1] - self._target_pos[-1])

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            obj = obs[4:7]

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
            reward = reward_utils.hamacher_product(object_grasped, in_place)

            tcp_opened = obs[3]
            tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))

            if tcp_to_obj < 0.02 and tcp_opened > 0:
                reward += 1.0 + 7.0 * in_place
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
            del action

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM[:-1] - goal[:-1])
            reachRew = -reachDist
            reachDist_z = np.abs(fingerCOM[-1] - goal[-1])

            if reachDist < 0.05:
                reachNearRew = 1000 * (self.maxReachDist - reachDist_z) + c1 * (
                    np.exp(-(reachDist_z**2) / c2) + np.exp(-(reachDist_z**2) / c3)
                )
            else:
                reachNearRew = 0.0

            reachNearRew = max(reachNearRew, 0)
            reward = reachRew + reachNearRew

            return (
                float(reward),
                0.0,
                0.0,
                float(np.linalg.norm(self.obj_init_pos - self._target_pos)),
                0.0,
                0.0,
            )
