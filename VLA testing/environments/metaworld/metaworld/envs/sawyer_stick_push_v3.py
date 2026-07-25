from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import ObservationDict, StickInitConfigDict
from metaworld.utils import reward_utils


class SawyerStickPushEnvV3(SawyerXYZEnv):
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
        obj_low = (-0.08, 0.58, 0.000)
        obj_high = (-0.03, 0.62, 0.001)
        goal_low = (0.399, 0.55, 0.1319)
        goal_high = (0.401, 0.6, 0.1321)

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

        self.init_config: StickInitConfigDict = {
            "stick_init_pos": np.array([-0.1, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config["stick_init_pos"]
        self.stick_init_pos = self.init_config["stick_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        # For now, fix the object initial position.
        self.obj_init_pos = np.array([0.2, 0.6, 0.0])
        self.obj_init_qpos = np.array([0.0, 0.0])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high), dtype=np.float64)
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_stick_obj.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        stick = obs[4:7]
        container = obs[11:14]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            container_to_target,
            grasp_reward,
            stick_in_place,
        ) = self.compute_reward(action, obs)
        assert self._target_pos is not None
        success = float(np.linalg.norm(container - self._target_pos) <= 0.12)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (stick[2] - 0.01 > self.stick_init_pos[2])
        )

        info = {
            "success": grasp_success and success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": stick_in_place,
            "obj_to_target": container_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.hstack(
            (
                self.get_body_com("stick").copy(),
                self._get_site_pos("insertion") + np.array([0.0, 0.09, 0.0]),
            )
        )

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.body("stick").xmat.reshape(3, 3)
        return np.hstack(
            (
                Rotation.from_matrix(geom_xmat).as_quat(),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            )
        )

    def _get_obs_dict(self) -> ObservationDict:
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self._get_site_pos("insertion") + np.array(
            [0.0, 0.09, 0.0]
        )
        return obs_dict

    def _set_stick_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:18] = pos.copy()
        qvel[16:18] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.stick_init_pos = self.init_config["stick_init_pos"]
        self._target_pos = np.array([0.4, 0.6, self.stick_init_pos[-1]])

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.stick_init_pos = np.concatenate([goal_pos[:2], [self.stick_init_pos[-1]]])
        self._target_pos = np.concatenate(
            [goal_pos[-3:-1], [self._get_site_pos("insertion")[-1]]]
        )

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com("object").copy()

        self.model.site("goal").pos = self._target_pos

        self.liftThresh = 0.04
        self.stickHeight = self.get_body_com("stick").copy()[2]
        self.heightTarget = self.stickHeight + self.liftThresh

        assert self.obj_init_pos is not None and self.stick_init_pos is not None

        self.maxPlaceDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - np.array(self.stick_init_pos)
            )
            + self.heightTarget
        )
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])

        return self._get_obs()

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float,
        pad_success_thresh: float,
        object_reach_radius: float,
        xz_thresh: float,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        """Reward for agent grasping obj.

        Args:
            action(np.ndarray): (4,) array representing the action
                delta(x), delta(y), delta(z), gripper_effort
            obj_pos(np.ndarray): (3,) array representing the obj x,y,z
            obj_radius(float):radius of object's bounding sphere
            pad_success_thresh(float): successful distance of gripper_pad
                to object
            object_reach_radius(float): successful distance of gripper center
                to the object.
            xz_thresh(float): successful distance of gripper in x_z axis to the
                object. Y axis not included since the caging function handles
                    successful grasping in the Y axis.
            desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
            high_density(bool): flag for high-density. Cannot be used with medium-density.
            medium_density(bool): flag for medium-density. Cannot be used with high-density.
        """
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.stick_init_pos[1])

        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            reward_utils.tolerance(
                pad_to_obj_lr[i],  # "x" in the description above
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],  # "margin" in the description above
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        caging_xz_margin = np.linalg.norm(self.stick_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            float(
                np.linalg.norm(tcp[xz] - obj_pos[xz])
            ),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid="long_tail",
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        )

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.stick_init_pos - self.init_tcp)
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                float(tcp_to_obj),
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2

        return caging_and_gripping

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.12
            tcp = self.tcp_center
            stick = obs[4:7] + np.array([0.015, 0.0, 0.0])
            container = obs[11:14]
            tcp_opened: float = obs[3]
            target = self._target_pos

            tcp_to_stick = float(np.linalg.norm(stick - tcp))
            stick_to_target = float(np.linalg.norm(stick - target))
            stick_in_place_margin = float(
                np.linalg.norm(self.stick_init_pos - target) - _TARGET_RADIUS
            )
            stick_in_place = reward_utils.tolerance(
                stick_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=stick_in_place_margin,
                sigmoid="long_tail",
            )

            container_to_target = float(np.linalg.norm(container - target))
            container_in_place_margin = float(
                np.linalg.norm(self.obj_init_pos - target) - _TARGET_RADIUS
            )
            container_in_place = reward_utils.tolerance(
                container_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=container_in_place_margin,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(
                action=action,
                obj_pos=stick,
                obj_radius=0.04,
                pad_success_thresh=0.05,
                object_reach_radius=0.01,
                xz_thresh=0.01,
                high_density=True,
            )

            reward = object_grasped

            if (
                tcp_to_stick < 0.02
                and (tcp_opened > 0)
                and (stick[2] - 0.01 > self.stick_init_pos[2])
            ):
                object_grasped = 1
                reward = 2.0 + 5.0 * stick_in_place + 3.0 * container_in_place

                if container_to_target <= _TARGET_RADIUS:
                    reward = 10.0
            return (
                reward,
                tcp_to_stick,
                tcp_opened,
                container_to_target,
                object_grasped,
                stick_in_place,
            )
        else:
            stickPos = obs[4:7]
            objPos = obs[6:9]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            pushGoal = self._target_pos

            pushDist = np.linalg.norm(objPos[:2] - pushGoal[:2])
            placeDist = np.linalg.norm(objPos - stickPos)
            reachDist = np.linalg.norm(stickPos - fingerCOM)

            reachRew = -reachDist
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50

            tolerance = 0.01
            self.pickCompleted = stickPos[2] >= (heightTarget - tolerance)

            objDropped = (
                (stickPos[2] < (self.stickHeight + 0.005))
                and (pushDist > 0.02)
                and (reachDist > 0.02)
            )
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

            hScale = 100
            if self.pickCompleted and not objDropped:
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (stickPos[2] > (self.stickHeight + 0.005)):
                pickRew = hScale * min(heightTarget, stickPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not objDropped
            if cond:
                pushRew = 1000 * (self.maxPlaceDist - placeDist) + c1 * (
                    np.exp(-(placeDist**2) / c2) + np.exp(-(placeDist**2) / c3)
                )
                if placeDist < 0.05:
                    c4 = 2000
                    c5 = 0.001
                    c6 = 0.0001
                    pushRew += 1000 * (self.maxPushDist - pushDist) + c4 * (
                        np.exp(-(pushDist**2) / c5) + np.exp(-(pushDist**2) / c6)
                    )
                pushRew = max(pushRew, 0)

                pushRew, pushDist = [pushRew, pushDist]
            else:
                pushRew, pushDist = [0, pushDist]

            assert (pushRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + pushRew

            return float(reward), 0.0, 0.0, float(pushDist), 0.0, 0.0
