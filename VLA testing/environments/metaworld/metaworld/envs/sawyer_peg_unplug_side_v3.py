from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerPegUnplugSideEnvV3(SawyerXYZEnv):
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
        obj_low = (-0.25, 0.6, -0.001)
        obj_high = (-0.15, 0.8, 0.001)
        goal_low = obj_low + np.array([0.194, 0.0, 0.131])
        goal_high = obj_high + np.array([0.194, 0.0, 0.131])

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
            "obj_init_pos": np.array([-0.225, 0.6, 0.05]),
            "hand_init_pos": np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([-0.225, 0.6, 0.0])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_peg_unplug_side.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        # obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            grasp_success,
        ) = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

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
        return self._get_site_pos("pegEnd")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("plug1").xquat

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos
        qpos[12:16] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[9:12] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        pos_box = self._get_state_rand_vec()
        self.model.body("box").pos = pos_box
        pos_plug = pos_box + np.array([0.044, 0.0, 0.131])
        self._set_obj_xyz(pos_plug)
        self.obj_init_pos = self._get_site_pos("pegEnd")

        self._target_pos = pos_plug + np.array([0.15, 0.0, 0.0])
        self.model.site("goal").pos = self._target_pos

        assert self._target_pos is not None and self.obj_init_pos is not None
        self.maxPlacingDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened: float = obs[3]
            target = self._target_pos
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            obj_to_target = float(np.linalg.norm(obj - target))
            pad_success_margin = 0.05
            object_reach_radius = 0.01
            x_z_margin = 0.005
            obj_radius = 0.025

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=object_reach_radius,
                obj_radius=obj_radius,
                pad_success_thresh=pad_success_margin,
                xz_thresh=x_z_margin,
                desired_gripper_effort=0.8,
                high_density=True,
            )
            in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.05),
                margin=in_place_margin,
                sigmoid="long_tail",
            )
            grasp_success = tcp_opened > 0.5 and (obj[0] - self.obj_init_pos[0] > 0.015)

            reward = 2 * object_grasped

            if grasp_success and tcp_to_obj < 0.035:
                reward = 1 + 2 * object_grasped + 5 * in_place

            if obj_to_target <= 0.05:
                reward = 10.0

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                obj_to_target,
                object_grasped,
                in_place,
                float(grasp_success),
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            placingGoal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)

            placingDist = np.linalg.norm(objPos[:-1] - placingGoal[:-1])

            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])

            assert fingerCOM is not None and self.hand_init_pos is not None

            zRew = np.linalg.norm(fingerCOM[-1] - self.hand_init_pos[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50

            self.reachCompleted = reachDist < 0.05

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.reachCompleted:
                placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                    np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
                )
                placeRew = max(placeRew, 0)
                placeRew, placingDist = [placeRew, placingDist]
            else:
                placeRew, placingDist = [0, placingDist]

            assert placeRew >= 0
            reward = reachRew + placeRew

            return float(reward), 0.0, 0.0, float(placingDist), 0.0, 0.0, 0.0
