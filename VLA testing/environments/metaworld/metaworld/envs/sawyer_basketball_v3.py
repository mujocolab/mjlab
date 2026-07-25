from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerBasketballEnvV3(SawyerXYZEnv):
    PAD_SUCCESS_MARGIN: float = 0.06
    TARGET_RADIUS: float = 0.08

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
        obj_low = (-0.1, 0.6, 0.0299)
        obj_high = (0.1, 0.7, 0.0301)
        goal_low = (-0.1, 0.85, 0.0)
        goal_high = (0.1, 0.9 + 1e-7, 0.0)

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
            "obj_init_pos": np.array([0, 0.6, 0.03], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.9, 0])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0, -0.083, 0.2499]),
            np.array(goal_high) + np.array([0, -0.083, 0.2501]),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_basketball.xml")

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
        ) = self.compute_reward(action, obs)

        assert self.obj_init_pos is not None
        info = {
            "success": float(obj_to_target <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(
                (tcp_open > 0) and (obj[2] - 0.03 > self.obj_init_pos[2])
            ),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("bsktball")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("bsktball").xquat

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        goal_pos = self._get_state_rand_vec()
        basket_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - basket_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            basket_pos = goal_pos[3:]
        assert self.obj_init_pos is not None
        self.obj_init_pos = np.concatenate([goal_pos[:2], [self.obj_init_pos[-1]]])
        self.model.body("basket_goal").pos = basket_pos
        self._target_pos = self.data.site("goal").xpos
        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        self.liftThresh = 0.3
        self.objHeight = self.data.geom("objGeom").xpos[2]
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

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.obj_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            obj = obs[4:7]
            # Force target to be slightly above basketball hoop
            target = self._target_pos.copy()
            target[2] = 0.3

            # Emphasize Z error
            scale = np.array([1.0, 1.0, 2.0])
            target_to_obj = (obj - target) * scale
            target_to_obj = float(np.linalg.norm(target_to_obj))
            target_to_obj_init = (self.obj_init_pos - target) * scale
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=target_to_obj_init,
                sigmoid="long_tail",
            )
            tcp_opened = float(obs[3])
            tcp_to_obj = float(np.linalg.norm(obj - self.tcp_center))

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.01,
                obj_radius=0.025,
                pad_success_thresh=0.06,
                xz_thresh=0.005,
                high_density=True,
            )
            if (
                tcp_to_obj < 0.035
                and tcp_opened > 0
                and obj[2] - 0.01 > self.obj_init_pos[2]
            ):
                object_grasped = 1.0
            reward = reward_utils.hamacher_product(object_grasped, in_place)

            if (
                tcp_to_obj < 0.035
                and tcp_opened > 0
                and obj[2] - 0.01 > self.obj_init_pos[2]
            ):
                reward += 1.0 + 5.0 * in_place
            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0
            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                target_to_obj,
                object_grasped,
                in_place,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos("goal"))

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])
            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50

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

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )

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
            return reward, 0.0, 0.0, float(placingDist), 0.0, 0.0
