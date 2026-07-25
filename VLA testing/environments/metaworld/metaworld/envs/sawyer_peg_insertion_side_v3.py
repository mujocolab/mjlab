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


class SawyerPegInsertionSideEnvV3(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.07
    """
    Motivation for V3:
        V1 was difficult to solve because the observation didn't say where
        to insert the peg (the hole's location). Furthermore, the hole object
        could be initialized in such a way that it severely restrained the
        sawyer's movement.
    Changelog from V1 to V3:
        - (8/21/20) Updated to Byron's XML
        - (7/7/20) Removed 1 element vector. Replaced with 3 element position
            of the hole (for consistency with other environments)
        - (6/16/20) Added a 1 element vector to the observation. This vector
            points from the end effector to the hole in the Y direction.
            i.e. (self._target_pos - pos_hand)[1]
        - (6/16/20) Used existing goal_low and goal_high values to constrain
            the hole's position, as opposed to hand_low and hand_high
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_init_pos = (0, 0.6, 0.2)

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.5, 0.02)
        obj_high = (0.2, 0.7, 0.02)
        goal_low = (-0.35, 0.4, -0.001)
        goal_high = (-0.25, 0.7, 0.001)

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
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.3, 0.6, 0.0])

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.hand_init_pos = np.array(hand_init_pos)

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0.03, 0.0, 0.13]),
            np.array(goal_high) + np.array([0.03, 0.0, 0.13]),
            dtype=np.float64,
        )

        self.liftThresh = 0.11

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_peg_insertion_side.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            collision_box_front,
            ip_orig,
        ) = self.compute_reward(action, obs)
        assert self.obj_init_pos is not None
        grasp_success = float(
            tcp_to_obj < 0.02
            and (tcp_open > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        )
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
        return self._get_site_pos("pegGrasp")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.site("pegGrasp").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        self.obj_init_pos = pos_peg
        self.peg_head_pos_init = self._get_site_pos("pegHead")
        self._set_obj_xyz(self.obj_init_pos)
        self.model.body("box").pos = pos_box
        self._target_pos = pos_box + np.array([0.03, 0.0, 0.13])
        self.model.site("goal").pos = self._target_pos

        self.objHeight = self.get_body_com("peg").copy()[2]
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

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            tcp = self.tcp_center
            obj = obs[4:7]
            obj_head = self._get_site_pos("pegHead")
            tcp_opened: float = obs[3]
            target = self._target_pos
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            scale = np.array([1.0, 2.0, 2.0])
            #  force agent to pick up object then insert
            obj_to_target = float(np.linalg.norm((obj_head - target) * scale))

            in_place_margin = float(
                np.linalg.norm((self.peg_head_pos_init - target) * scale)
            )
            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, self.TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )
            ip_orig = in_place
            brc_col_box_1 = self._get_site_pos("bottom_right_corner_collision_box_1")
            tlc_col_box_1 = self._get_site_pos("top_left_corner_collision_box_1")

            brc_col_box_2 = self._get_site_pos("bottom_right_corner_collision_box_2")
            tlc_col_box_2 = self._get_site_pos("top_left_corner_collision_box_2")
            collision_box_bottom_1 = reward_utils.rect_prism_tolerance(
                curr=obj_head, one=tlc_col_box_1, zero=brc_col_box_1
            )
            collision_box_bottom_2 = reward_utils.rect_prism_tolerance(
                curr=obj_head, one=tlc_col_box_2, zero=brc_col_box_2
            )
            collision_boxes = reward_utils.hamacher_product(
                collision_box_bottom_2, collision_box_bottom_1
            )
            in_place = reward_utils.hamacher_product(in_place, collision_boxes)

            pad_success_margin = 0.03
            object_reach_radius = 0.01
            x_z_margin = 0.005
            obj_radius = 0.0075

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=object_reach_radius,
                obj_radius=obj_radius,
                pad_success_thresh=pad_success_margin,
                xz_thresh=x_z_margin,
                high_density=True,
            )
            if (
                tcp_to_obj < 0.08
                and (tcp_opened > 0)
                and (obj[2] - 0.01 > self.obj_init_pos[2])
            ):
                object_grasped = 1.0
            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )
            reward = in_place_and_object_grasped

            if (
                tcp_to_obj < 0.08
                and (tcp_opened > 0)
                and (obj[2] - 0.01 > self.obj_init_pos[2])
            ):
                reward += 1.0 + 5 * in_place

            if obj_to_target <= 0.07:
                reward = 10.0

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                obj_to_target,
                object_grasped,
                in_place,
                collision_boxes,
                ip_orig,
            )
        else:
            objPos = obs[4:7]
            pegHeadPos = self._get_site_pos("pegHead")

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            placingGoal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)

            placingDistHead = np.linalg.norm(pegHeadPos - placingGoal)
            placingDist = np.linalg.norm(objPos - placingGoal)

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50

            tolerance = 0.01
            self.pickCompleted = objPos[2] >= (heightTarget - tolerance)

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

            hScale = 100
            if self.pickCompleted and not (objDropped):
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )

            cond = self.pickCompleted and (reachDist < 0.1) and not (objDropped)

            if cond:
                if placingDistHead <= 0.05:
                    placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                        np.exp(-(placingDist**2) / c2)
                        + np.exp(-(placingDist**2) / c3)
                    )
                else:
                    placeRew = 1000 * (self.maxPlacingDist - placingDistHead) + c1 * (
                        np.exp(-(placingDistHead**2) / c2)
                        + np.exp(-(placingDistHead**2) / c3)
                    )
                placeRew = max(placeRew, 0)
                placeRew, placingDist = placeRew, placingDist
            else:
                placeRew, placingDist = [0, placingDist]

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew

            return float(reward), 0.0, 0.0, float(placingDist), 0.0, 0.0, 0.0, 0.0
