import numpy as np

from mjlab import Robot
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG

robot = Robot(G1_ROBOT_CFG)

joint_names = [
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
]
robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

body_names = [
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]
body_indexes = robot.find_bodies(body_names, preserve_order=True)[0]

mj_npz = np.load("/home/kevin/dev/mjlab/motions/motion.npz")
is_npz = np.load("/home/kevin/Downloads/motion.npz")

assert mj_npz["fps"] == is_npz["fps"]

mj_qpos = mj_npz["joint_pos"][:, robot_joint_indexes]
is_qpos = is_npz["joint_pos"]
np.testing.assert_allclose(mj_qpos, is_qpos)

mj_qvel = mj_npz["joint_vel"][:, robot_joint_indexes]
is_qvel = is_npz["joint_vel"]
np.testing.assert_allclose(mj_qvel, is_qvel)

mj_body_pos = mj_npz["body_pos_w"][:, body_indexes]
is_body = is_npz["body_pos_w"]
np.testing.assert_allclose(mj_body_pos, is_body, atol=1e-5)

mj_body_quat = mj_npz["body_quat_w"][:, body_indexes]
is_body_quat = is_npz["body_quat_w"]
np.testing.assert_allclose(mj_body_quat, is_body_quat, atol=1e-5)

mj_body_lin_vel = mj_npz["body_lin_vel_w"][:, body_indexes]
is_body_lin_vel = is_npz["body_lin_vel_w"]
np.testing.assert_allclose(mj_body_lin_vel, is_body_lin_vel, atol=1e-5)

mj_body_ang_vel = mj_npz["body_ang_vel_w"][:, body_indexes]
is_body_ang_vel = is_npz["body_ang_vel_w"]
np.testing.assert_allclose(mj_body_ang_vel, is_body_ang_vel, atol=1e-5)
