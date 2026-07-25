from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerLeverPullEnvV3(SawyerXYZEnv):
    """SawyerLeverPullEnv.

    Motivation for V3:
        V1 was impossible to solve because the lever would have to be pulled
        through the table in order to reach the target location.
    Changelog from V1 to V3:
        - (8/12/20) Updated to Byron's XML
        - (7/7/20) Added 3 element lever position to the observation
            (for consistency with other environments)
        - (6/23/20) In `reset_model`, changed `final_pos[2] -= .17` to `+= .17`
            This ensures that the target point is above the table.
    """

    LEVER_RADIUS = 0.2

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.0)
        obj_high = (0.1, 0.8, 0.0)

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
            "obj_init_pos": np.array([0, 0.7, 0.0]),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.12, 0.88, 0.05])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._lever_pos_init = None

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_lever_pull.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            shoulder_to_lever,
            ready_to_lift,
            lever_error,
            lever_engagement,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(lever_error <= np.pi / 24),
            "near_object": float(shoulder_to_lever < 0.03),
            "grasp_success": float(ready_to_lift > 0.9),
            "grasp_reward": ready_to_lift,
            "in_place_reward": lever_engagement,
            "obj_to_target": shoulder_to_lever,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("leverStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lever")
        ] = self.obj_init_pos
        self._lever_pos_init = self.obj_init_pos + np.array(
            [0.12, -self.LEVER_RADIUS, 0.25]
        )
        self._target_pos = self.obj_init_pos + np.array(
            [0.12, 0.0, 0.25 + self.LEVER_RADIUS]
        )
        self.model.site("goal").pos = self._target_pos

        assert self._target_pos is not None and self.obj_init_pos is not None
        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float]:
        assert self._lever_pos_init is not None
        if self.reward_function_version == "v2":
            gripper = obs[:3]
            lever = obs[4:7]

            # De-emphasize y error so that we get Sawyer's shoulder underneath the
            # lever prior to bumping on against
            scale = np.array([4.0, 1.0, 4.0])
            # Offset so that we get the Sawyer's shoulder underneath the lever,
            # rather than its fingers
            offset = np.array([0.0, 0.055, 0.07])

            shoulder_to_lever = (gripper + offset - lever) * scale
            shoulder_to_lever_init = (
                self.init_tcp + offset - self._lever_pos_init
            ) * scale

            # This `ready_to_lift` reward should be a *hint* for the agent, not an
            # end in itself. Make sure to devalue it compared to the value of
            # actually lifting the lever
            ready_to_lift = reward_utils.tolerance(
                float(np.linalg.norm(shoulder_to_lever)),
                bounds=(0, 0.02),
                margin=np.linalg.norm(shoulder_to_lever_init),
                sigmoid="long_tail",
            )

            # The skill of the agent should be measured by its ability to get the
            # lever to point straight upward. This means we'll be measuring the
            # current angle of the lever's joint, and comparing with 90deg.
            lever_angle = float(-self.data.joint("LeverAxis").qpos.item())
            lever_angle_desired = np.pi / 2.0

            lever_error = abs(lever_angle - lever_angle_desired)

            # We'll set the margin to 15deg from horizontal. Angles below that will
            # receive some reward to incentivize exploration, but we don't want to
            # reward accidents too much. Past 15deg is probably intentional movement
            lever_engagement = reward_utils.tolerance(
                lever_error,
                bounds=(0, np.pi / 48.0),
                margin=(np.pi / 2.0) - (np.pi / 12.0),
                sigmoid="long_tail",
            )

            target = self._target_pos
            obj_to_target = float(np.linalg.norm(lever - target))
            in_place_margin = float(np.linalg.norm(self._lever_pos_init - target))

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.04),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            # reward = 2.0 * ready_to_lift + 8.0 * lever_engagement
            reward = 10.0 * reward_utils.hamacher_product(ready_to_lift, in_place)
            return (
                reward,
                float(np.linalg.norm(shoulder_to_lever)),
                ready_to_lift,
                lever_error,
                lever_engagement,
            )
        else:
            del action

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return (reward, 0.0, 0.0, float(pullDist), 0.0)
