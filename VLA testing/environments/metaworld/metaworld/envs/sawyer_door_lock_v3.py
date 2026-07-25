from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerDoorLockEnvV3(SawyerXYZEnv):
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
        obj_low = (-0.1, 0.8, 0.15)
        obj_high = (0.1, 0.85, 0.15)

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
            "obj_init_pos": np.array([0, 0.85, 0.15], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.85, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._lock_length = 0.1

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_door_lock.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.02),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [
            ("goal_lock", self._target_pos),
            ("goal_unlock", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("lockStartLock")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("door_link").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        door_pos = self._get_state_rand_vec()
        self.model.body("door").pos = door_pos

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.obj_init_pos = self.data.body("lock_link").xpos
        self._target_pos = self.obj_init_pos + np.array([0.0, -0.04, -0.1])

        assert self._target_pos is not None and self.obj_init_pos is not None
        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            del action
            obj = obs[4:7]
            tcp = self.get_body_com("leftpad")

            scale = np.array([0.25, 1.0, 0.5])
            tcp_to_obj = float(np.linalg.norm((obj - tcp) * scale))
            tcp_to_obj_init = float(np.linalg.norm((obj - self.init_left_pad) * scale))

            obj_to_target = abs(self._target_pos[2] - obj[2])

            tcp_opened = max(obs[3], 0.0)
            near_lock = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.01),
                margin=tcp_to_obj_init,
                sigmoid="long_tail",
            )
            lock_pressed = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.005),
                margin=self._lock_length,
                sigmoid="long_tail",
            )

            reward = 2 * reward_utils.hamacher_product(tcp_opened, near_lock)
            reward += 8 * lock_pressed

            return (reward, tcp_to_obj, obs[3], obj_to_target, near_lock, lock_pressed)
        else:
            del action

            if isinstance(obs, dict):
                obs = obs["state_observation"]

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return float(reward), 0.0, 0.0, float(pullDist), 0.0, 0.0
