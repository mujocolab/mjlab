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


class SawyerPlateSlideSideEnvV3(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        goal_low = (-0.3, 0.54, 0.0)
        goal_high = (-0.25, 0.66, 0.0)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.6, 0.0)
        obj_high = (0.0, 0.6, 0.0)

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
            "obj_init_pos": np.array([0.0, 0.6, 0.0], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([-0.25, 0.6, 0.015])
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
        return full_V3_path_for("sawyer_xyz/sawyer_plate_slide_sideway.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_reward": object_grasped,
            "grasp_success": 0.0,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.geom("puck").xpos

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("puck").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:11] = pos
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self._target_pos = self.goal.copy()

        rand_vec = self._get_state_rand_vec()
        self.obj_init_pos = rand_vec[:3]
        self._target_pos = rand_vec[3:]
        self.data.body("puck_goal").xpos = self._target_pos
        self._set_obj_xyz(np.zeros(2))

        self.model.site("goal").pos = self._target_pos

        self.maxDist = np.linalg.norm(self.obj_init_pos[:-1] - self._target_pos[:-1])

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]
            target = self._target_pos

            obj_to_target = float(np.linalg.norm(obj - target))
            in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))
            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin - _TARGET_RADIUS,
                sigmoid="long_tail",
            )

            tcp_to_obj = float(np.linalg.norm(tcp - obj))
            obj_grasped_margin = float(
                np.linalg.norm(self.init_tcp - self.obj_init_pos)
            )
            object_grasped = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, _TARGET_RADIUS),
                margin=obj_grasped_margin - _TARGET_RADIUS,
                sigmoid="long_tail",
            )

            # in_place_and_object_grasped = reward_utils.hamacher_product(
            #     object_grasped, in_place
            # )
            reward = 1.5 * object_grasped

            if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
                reward = 2.0 + (7.0 * in_place)

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
            return [reward, obj_to_target]
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)

            pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if reachDist < 0.05:
                pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0
            reward = -reachDist + pullRew

            return float(reward), 0.0, 0.0, float(pullDist), 0.0, 0.0
