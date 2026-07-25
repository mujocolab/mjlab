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


class SawyerButtonPressTopdownWallEnvV3(SawyerXYZEnv):
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
        obj_low = (-0.1, 0.8, 0.115)
        obj_high = (0.1, 0.9, 0.115)

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
            "obj_init_pos": np.array([0, 0.8, 0.115], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.88, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_button_press_topdown_wall.xml")

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
            "success": float(obj_to_target <= 0.024),
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
        return []

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("btnGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("button") + np.array([0.0, 0.0, 0.193])

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("button").xquat

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()

        goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos
        self.model.body("box").pos = self.obj_init_pos
        mujoco.mj_forward(self.model, self.data)

        self._target_pos = self._get_site_pos("hole")
        self._obj_to_target_init = abs(
            self._target_pos[2] - self._get_site_pos("buttonStart")[2]
        )

        self.maxDist = np.abs(
            self._get_site_pos("buttonStart")[2] - self._target_pos[2]
        )

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
            tcp = self.tcp_center

            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = float(np.linalg.norm(obj - self.init_tcp))
            obj_to_target = abs(self._target_pos[2] - obj[2])

            tcp_closed = max(obs[3], 0.0)
            near_button = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.01),
                margin=tcp_to_obj_init,
                sigmoid="long_tail",
            )
            button_pressed = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.005),
                margin=self._obj_to_target_init,
                sigmoid="long_tail",
            )

            reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
            if tcp_to_obj <= 0.03:
                reward += 5 * button_pressed

            return (
                reward,
                tcp_to_obj,
                obs[3],
                obj_to_target,
                near_button,
                button_pressed,
            )
        else:
            del action

            if isinstance(obs, dict):
                obs = obs["state_observation"]

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pressGoal = self._target_pos[2]

            pressDist = np.abs(objPos[2] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

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
            reward = reachRew + pressRew

            return float(reward), 0.0, 0.0, float(pressDist), 0.0, 0.0
