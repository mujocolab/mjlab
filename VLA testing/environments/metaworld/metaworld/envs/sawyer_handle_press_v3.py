from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerHandlePressEnvV3(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.02

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
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.1, 0.8, -0.001)
        obj_high = (0.1, 0.9, 0.001)
        goal_low = (-0.1, 0.55, 0.04)
        goal_high = (0.1, 0.70, 0.08)

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
            "obj_init_pos": np.array([0, 0.9, 0.0]),
            "hand_init_pos": np.array(
                (0, 0.6, 0.2),
            ),
        }
        self.goal = np.array([0, 0.8, 0.14])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_handle_press.xml")

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

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("handleStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("box").pos = self.obj_init_pos
        self._set_obj_xyz(np.array(-0.001))
        self._target_pos = self._get_site_pos("goalPress")
        self.maxDist = np.abs(
            self.data.site("handleStart").xpos[-1] - self._target_pos[-1]
        )
        self.target_reward = 1000 * self.maxDist + 1000 * 2
        self._handle_init_pos = self._get_pos_objects()

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            del actions
            obj = self._get_pos_objects()
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj[2] - target[2]
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = self._handle_init_pos[2] - target[2]
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            handle_radius = 0.02
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = np.linalg.norm(self._handle_init_pos - self.init_tcp)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="long_tail",
            )
            tcp_opened = 0
            object_grasped = reach

            reward = reward_utils.hamacher_product(reach, in_place)
            reward = 1.0 if target_to_obj <= self.TARGET_RADIUS else reward
            reward *= 10
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

            leftFinger = self._get_site_pos("leftEndEffector")
            fingerCOM = leftFinger

            pressGoal = self._target_pos[-1]

            pressDist = np.abs(objPos[-1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000 * (self.maxDist - pressDist) + c1 * (
                    np.exp(-(pressDist**2) / c2) + np.exp(-(pressDist**2) / c3)
                )
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew

            return float(reward), 0.0, 0.0, float(pressDist), 0.0, 0.0
