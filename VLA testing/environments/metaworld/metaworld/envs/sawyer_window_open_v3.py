from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerWindowOpenEnvV3(SawyerXYZEnv):
    """SawyerWindowOpenEnv.

    Motivation for V3:
        When V1 scripted policy failed, it was often due to limited path length.
    Changelog from V1 to V3:
        - (8/11/20) Updated to Byron's XML
        - (7/7/20) Added 3 element handle position to the observation
            (for consistency with other environments)
        - (6/15/20) Increased max_path_length from 150 to 200
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
        obj_low = (-0.1, 0.7, 0.16)
        obj_high = (0.1, 0.9, 0.16)

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
            "obj_init_pos": np.array([-0.1, 0.785, 0.16], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self.maxPullDist = 0.2
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_window_horizontal.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("handleOpenStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        self.obj_init_pos = self._get_state_rand_vec()

        self._target_pos = self.obj_init_pos + np.array([0.2, 0.0, 0.0])
        self.model.body("window").pos = self.obj_init_pos

        self.window_handle_pos_init = self._get_pos_objects()
        self.data.joint("window_slide").qpos = 0.0
        assert self._target_pos is not None

        self.model.site("goal").pos = self._target_pos

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            del actions
            obj = self._get_pos_objects()
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj: float = obj[0] - target[0]
            target_to_obj = float(np.linalg.norm(target_to_obj))
            target_to_obj_init = self.obj_init_pos[0] - target[0]
            target_to_obj_init = float(np.linalg.norm(target_to_obj_init))

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            handle_radius = 0.02
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = float(
                np.linalg.norm(self.window_handle_pos_init - self.init_tcp)
            )
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="long_tail",
            )
            tcp_opened = 0.0
            object_grasped = reach

            reward = 10 * reward_utils.hamacher_product(reach, in_place)
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                target_to_obj,
                object_grasped,
                in_place,
            )
        else:
            del actions
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos

            pullDist = np.abs(objPos[0] - pullGoal[0])
            reachDist = np.linalg.norm(objPos - fingerCOM)

            self.reachCompleted = reachDist < 0.05

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachRew = -reachDist
            if self.reachCompleted:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
            else:
                pullRew = 0
            reward = reachRew + pullRew

            return reward, 0.0, 0.0, pullDist, 0.0, 0.0
