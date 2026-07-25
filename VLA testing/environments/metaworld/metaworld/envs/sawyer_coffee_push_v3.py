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


class SawyerCoffeePushEnvV3(SawyerXYZEnv):
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
        obj_low = (-0.1, 0.55, -0.001)
        obj_high = (0.1, 0.65, +0.001)
        goal_low = (-0.05, 0.7, -0.001)
        goal_high = (0.05, 0.75, +0.001)

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
            "obj_init_pos": np.array([0.0, 0.6, 0.0]),
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
        self.goal = np.array([0.0, 0.75, 0])
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
        return full_V3_path_for("sawyer_xyz/sawyer_coffee.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
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
        grasp_success = float(self.touching_main_object and (tcp_open > 0))

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

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [("coffee_goal", self._target_pos)]

    def _get_id_main_object(self) -> int:
        return self.data.geom("mug").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("mug").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.15:
            pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        pos_machine = pos_mug_goal + np.array([0.0, 0.22, 0.0])

        self.model.body("coffee_machine").pos = pos_machine

        self._target_pos = pos_mug_goal
        self.model.site("mug_goal").pos = self._target_pos

        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            obj = obs[4:7]
            target = self._target_pos.copy()

            # Emphasize X and Y errors
            scale = np.array([2.0, 2.0, 1.0])
            target_to_obj = (obj - target) * scale
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = (self.obj_init_pos - target) * scale
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, 0.05),
                margin=target_to_obj_init,
                sigmoid="long_tail",
            )
            tcp_opened = obs[3]
            tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.04,
                obj_radius=0.02,
                pad_success_thresh=0.05,
                xz_thresh=0.05,
                desired_gripper_effort=0.7,
                medium_density=True,
            )

            reward = reward_utils.hamacher_product(object_grasped, in_place)

            if tcp_to_obj < 0.04 and tcp_opened > 0:
                reward += 1.0 + 5.0 * in_place
            if target_to_obj < 0.05:
                reward = 10.0
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                float(np.linalg.norm(obj - target)),  # recompute to avoid `scale` above
                object_grasped,
                in_place,
            )
        else:
            del action

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            assert np.all(goal == self._get_site_pos("coffee_goal"))
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

            return (
                float(reward),
                0.0,
                0.0,
                float(np.linalg.norm(objPos - goal)),
                0.0,
                0.0,
            )
