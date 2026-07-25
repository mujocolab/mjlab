from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict, ObservationDict
from metaworld.utils.reward_utils import tolerance


class SawyerNutAssemblyEnvV3(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH: float = 0.02

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
        obj_low = (0, 0.6, 0.02)
        obj_high = (0, 0.6, 0.02)
        goal_low = (-0.1, 0.75, 0.1)
        goal_high = (0.1, 0.85, 0.1)

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
            "obj_init_pos": np.array([0, 0.6, 0.02], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }

        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
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
        return full_V3_path_for("sawyer_xyz/sawyer_assembly_peg.xml")

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
        assert isinstance(
            self._target_pos, np.ndarray
        ), "`reset_model()` must be called before `_target_site_config` is accessed."
        return [("pegTop", self._target_pos)]

    def _get_id_main_object(self) -> int:
        """TODO: Reggie"""
        return self.model.geom_name2id("WrenchHandle")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.site("RoundNut-8").xpos

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("RoundNut").xquat

    def _get_obs_dict(self) -> ObservationDict:
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self.get_body_com("RoundNut")
        return obs_dict

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]
        self._target_pos = goal_pos[-3:]
        peg_pos = self._target_pos - np.array([0.0, 0.0, 0.05])
        self._set_obj_xyz(self.obj_init_pos)
        self.model.body("peg").pos = peg_pos
        self.model.site("pegTop").pos = self._target_pos

        if self.reward_function_version == "v1":
            self.obj_height = self.data.site_xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "RoundNut-8")
            ][2]
            self.heightTarget = self.obj_height + 0.1
            self.pickCompleted = False
            self.placeCompleted = False
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

    @staticmethod
    def _reward_quat(obs: npt.NDArray[np.float64]) -> float:
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = float(np.linalg.norm(obs[7:11] - ideal))
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(
        wrench_center: npt.NDArray[Any], target_pos: npt.NDArray[Any]
    ) -> tuple[float, bool]:
        pos_error = target_pos - wrench_center

        radius = np.linalg.norm(pos_error[:2])

        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = bool(aligned and hooked)

        # Target height is a 3D funnel centered on the peg.
        # use the success flag to widen the bottleneck once the agent
        # learns to place the wrench on the peg -- no reason to encourage
        # tons of alignment accuracy if task is already solved
        threshold = 0.02 if success else 0.01
        target_height = 0.0
        if radius > threshold:
            target_height = 0.02 * np.log(radius - threshold) + 0.2

        pos_error[2] = target_height - wrench_center[2]

        scale = np.array([1.0, 1.0, 3.0])
        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02 or radius < threshold
        in_place = a * float(lifted) + b * tolerance(
            float(np.linalg.norm(pos_error * scale)),
            bounds=(0, 0.02),
            margin=0.4,
            sigmoid="long_tail",
        )

        return in_place, success

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, bool]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            hand = obs[:3]
            wrench = obs[4:7]
            wrench_center = self._get_site_pos("RoundNut")
            # `self._gripper_caging_reward` assumes that the target object can be
            # approximated as a sphere. This is not true for the wrench handle, so
            # to avoid re-writing the `self._gripper_caging_reward` we pass in a
            # modified wrench position.
            # This modified position's X value will perfect match the hand's X value
            # as long as it's within a certain threshold
            wrench_threshed = wrench.copy()
            threshold = SawyerNutAssemblyEnvV3.WRENCH_HANDLE_LENGTH / 2.0
            if abs(wrench[0] - hand[0]) < threshold:
                wrench_threshed[0] = hand[0]

            reward_quat = SawyerNutAssemblyEnvV3._reward_quat(obs)
            reward_grab = self._gripper_caging_reward(
                actions,
                wrench_threshed,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.02,
                xz_thresh=0.01,
                medium_density=True,
            )
            reward_in_place, success = SawyerNutAssemblyEnvV3._reward_pos(
                wrench_center, self._target_pos
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
            # Override reward on success
            if success:
                reward = 10.0

            return (
                reward,
                reward_grab,
                reward_quat,
                reward_in_place,
                success,
            )
        else:
            graspPos = obs[4:7]
            objPos = self.get_body_com("RoundNut")

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            placingGoal = self._target_pos

            reachDist = np.linalg.norm(graspPos - fingerCOM)

            placingDist = np.linalg.norm(objPos[:2] - placingGoal[:2])
            placingDistFinal = np.abs(objPos[-1] - self.obj_height)

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])
            if reachDistxy < 0.04:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1], 0) / 50

            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance) and reachDist < 0.03:
                self.pickCompleted = True
            else:
                self.pickCompleted = False

            objDropped = (
                (objPos[2] < (self.obj_height + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )

            self.placeCompleted = (
                abs(objPos[0] - placingGoal[0]) < 0.03
                and abs(objPos[1] - placingGoal[1]) < 0.03
            )

            hScale = 100
            if self.placeCompleted or (self.pickCompleted and not objDropped):
                pickRew = hScale * heightTarget
            elif (reachDist < 0.04) and (objPos[2] > (self.obj_height + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
            )
            if self.placeCompleted:
                c4 = 2000
                c5 = 0.003
                c6 = 0.0003
                placeRew += 2000 * (heightTarget - placingDistFinal) + c4 * (
                    np.exp(-(placingDistFinal**2) / c5)
                    + np.exp(-(placingDistFinal**2) / c6)
                )
            placeRew = max(placeRew, 0)
            cond = self.placeCompleted or (
                self.pickCompleted and (reachDist < 0.04) and not objDropped
            )
            if cond:
                placeRew, placingDist, placingDistFinal = [
                    placeRew,
                    placingDist,
                    placingDistFinal,
                ]
            else:
                placeRew, placingDist, placingDistFinal = [
                    0,
                    placingDist,
                    placingDistFinal,
                ]

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew
            success = (
                abs(objPos[0] - placingGoal[0]) < 0.03
                and abs(objPos[1] - placingGoal[1]) < 0.03
                and placingDistFinal <= 0.04
            )
            return (
                float(reward),
                0.0,
                0.0,
                0.0,
                success,
            )
