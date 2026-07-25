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


class SawyerBoxCloseEnvV3(SawyerXYZEnv):
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
        obj_low = (-0.05, 0.5, 0.02)
        obj_high = (0.05, 0.55, 0.02)
        goal_low = (-0.1, 0.7, 0.133)
        goal_high = (0.1, 0.8, 0.133)

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
            "obj_init_pos": np.array([0, 0.55, 0.02], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.75, 0.133])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._target_to_obj_init = None

        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )

        self.init_obj_quat = None
        self.liftThresh = 0.12

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_box.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(success),
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("BoxHandleGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("top_link")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("top_link").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        box_height = self.get_body_com("boxbody")[2]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.25:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self._target_pos = goal_pos[-3:]

        self.model.body("boxbody").pos = np.concatenate(
            [self._target_pos[:2], [box_height]]
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        self.objHeight = self.data.geom("BoxHandleGeom").xpos[2]
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
        self.pickCompleted = False

        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions: npt.NDArray[Any]) -> float:
        return float(np.clip(((np.clip(actions[3], -1, 1) + 1.0) / 2.0), 0.0, 1.0))

    @staticmethod
    def _reward_quat(obs) -> float:
        # Ideal upright lid has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = float(np.linalg.norm(obs[7:11] - ideal))
        return max(1.0 - error / 0.2, 0.0)

    @staticmethod
    def _reward_pos(
        obs: npt.NDArray[np.float64], target_pos: npt.NDArray[Any]
    ) -> tuple[float, float]:
        hand = obs[:3]
        lid = obs[4:7] + np.array([0.0, 0.0, 0.02])

        threshold = 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            )
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            float(np.linalg.norm(hand - lid)),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1.0, 1.0, 3.0])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            float(np.linalg.norm(pos_error * error_scale)),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid="long_tail",
        )

        return ready_to_lift, lifted

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, bool]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            reward_grab = SawyerBoxCloseEnvV3._reward_grab_effort(actions)
            reward_quat = SawyerBoxCloseEnvV3._reward_quat(obs)
            reward_steps = SawyerBoxCloseEnvV3._reward_pos(obs, self._target_pos)

            reward = sum(
                (
                    2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
                    8.0 * reward_steps[1],
                )
            )

            # Override reward on success
            success = bool(np.linalg.norm(obs[4:7] - self._target_pos) < 0.08)
            if success:
                reward = 10.0

            # STRONG emphasis on proper lid orientation to prevent reward hacking
            # (otherwise agent learns to kick-flip the lid onto the box)
            reward *= reward_quat

            return (
                reward,
                reward_grab,
                *reward_steps,
                success,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            placeGoal = self._target_pos

            placingDist = np.linalg.norm(objPos - placeGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1], 0) / 50

            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance):
                self.pickCompleted = True
            else:
                self.pickCompleted = False

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
            success = bool(np.linalg.norm(obs[4:7] - self._target_pos) < 0.08)

            return float(reward), 0.0, 0.0, 0.0, success
