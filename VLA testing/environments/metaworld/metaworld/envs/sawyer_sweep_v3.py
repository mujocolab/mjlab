from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerSweepEnvV3(SawyerXYZEnv):
    OBJ_RADIUS: float = 0.02

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        init_puck_z = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (0.49, 0.6, 0.00)
        goal_high = (0.51, 0.7, 0.02)

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
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),  # type: ignore
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),  # type: ignore
        }
        self.goal = np.array([0.5, 0.65, 0.01])  # type: ignore
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.init_puck_z = init_puck_z

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64  # type: ignore
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)  # type: ignore

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_sweep_v3.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        grasp_success = float(self.touching_main_object and (tcp_opened > 0))

        info = {
            "success": float(target_to_obj <= 0.05),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_reward": object_grasped,
            "grasp_success": grasp_success,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xpos

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.objHeight = self._get_pos_objects()[2]

        obj_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate([obj_pos[:2], [self.obj_init_pos[-1]]])  # type: ignore
        self._target_pos[1] = obj_pos.copy()[1]

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        self.liftThresh = 0.04
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh

        self.maxReachDist = np.linalg.norm(self.init_tcp - np.array(self._target_pos))
        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )
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

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float,
        pad_success_thresh: float = 0,  # All of these args are unused
        object_reach_radius: float = 0,  # just here to match the parent's type signature
        xz_thresh: float = 0,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.01
        x_z_success_margin = 0.005

        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_pos[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_pos[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        right_gripping = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])  # type: ignore
        obj_position_x_z = np.copy(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        assert self.obj_init_pos is not None
        init_obj_x_z = self.obj_init_pos + np.array([0.0, -self.obj_init_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])

        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )
        x_z_caging = reward_utils.tolerance(
            float(tcp_obj_norm_x_z),
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        gripper_closed = min(max(0, action[-1]), 1)
        assert gripper_closed >= 0 and gripper_closed <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if caging > 0.95:
            gripping = y_gripping
        else:
            gripping = 0.0
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None
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

            object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)
            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )

            reward = (2 * object_grasped) + (6 * in_place_and_object_grasped)

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
            del action

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pushGoal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)
            pushDistxy = np.linalg.norm(objPos[:-1] - pushGoal[:-1])
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            assert objPos is not None and self.obj_init_pos is not None

            if float(objPos[-1]) < float(self.obj_init_pos[-1]) - 0.05:
                reachRew = np.array(0.0)  # type: ignore
                pushDistxy = np.array(0.0)  # type: ignore
                reachDist = 0.0  # type: ignore

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.reachCompleted:
                pushRew = 1000 * (self.maxPushDist - pushDistxy) + c1 * (
                    np.exp(-(pushDistxy**2) / c2) + np.exp(-(pushDistxy**2) / c3)
                )
                pushRew = max(pushRew, 0)
                pushRew = pushRew
            else:
                pushRew = 0

            reward = reachRew + pushRew

            return reward, 0.0, 0.0, float(pushDistxy), 0.0, 0.0
