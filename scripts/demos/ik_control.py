"""Interactive IK control demo.

Drag the 3D transform control in the viser viewer to move the YAM end-effector.
Use the gripper slider to open/close the gripper.

Run with:
  uv run python scripts/demos/ik_control.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
import viser

from mjlab.asset_zoo.robots.i2rt_yam.yam_constants import get_yam_robot_cfg
from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils.lab_api.math import quat_from_matrix
from mjlab.viewer.viser import ViserMujocoScene

DEMO_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.01),
  joint_pos={
    "joint2": 0.6,
    "joint3": 0.6,
    "joint4": 0.0,
    "left_finger": 0.037,
    "right_finger": -0.037,
  },
  joint_vel={".*": 0.0},
)

IK_ITERATIONS = 20
GRIPPER_MAX = 0.037


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  robot_cfg = get_yam_robot_cfg()
  robot_cfg.init_state = DEMO_INIT_STATE
  entity = Entity(robot_cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg(mujoco=MujocoCfg(gravity=(0, 0, -9.81)))
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  env = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
  ik_cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    ee_name="grasp_site",
    ee_type="site",
    posture_weight=0.0,
    joint_limit_weight=1e-1,
    damping=1e-1,
    use_relative_mode=False,
  )
  ik_action: DifferentialIKAction = ik_cfg.build(env)  # type: ignore[arg-type]
  joint_ids = ik_action._joint_ids

  grip_ids, _ = entity.find_joints("left_finger")
  grip_joint_ids = torch.tensor(grip_ids, device=device, dtype=torch.long)

  server = viser.ViserServer(label="IK Control Demo")
  scene = ViserMujocoScene.create(server, sim.mj_model, num_envs=1)
  scene.create_visualization_gui(
    camera_distance=1.0,
    camera_azimuth=135.0,
    camera_elevation=30.0,
  )

  site_id = ik_action._ee_global_id
  pos = sim.data.site_xpos[0, site_id].cpu().numpy()
  xmat = sim.data.site_xmat[0, site_id]
  quat = quat_from_matrix(xmat).cpu().numpy()

  transform_ctrl = server.scene.add_transform_controls(
    "/ik_target",
    position=(float(pos[0]), float(pos[1]), float(pos[2])),
    wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    scale=0.12,
  )

  with server.gui.add_folder("IK Control"):
    gripper_slider = server.gui.add_slider(
      "Gripper Opening",
      min=0.0,
      max=GRIPPER_MAX,
      step=0.001,
      initial_value=GRIPPER_MAX,
    )
    iterations_slider = server.gui.add_slider(
      "IK Iterations",
      min=1,
      max=50,
      step=1,
      initial_value=IK_ITERATIONS,
    )

  with server.gui.add_folder("IK Weights"):
    damping_slider = server.gui.add_slider(
      "Damping (λ)",
      min=1e-2,
      max=1.0,
      step=1e-3,
      initial_value=ik_cfg.damping,
    )
    pos_w_slider = server.gui.add_slider(
      "Position Weight",
      min=0.0,
      max=10.0,
      step=0.1,
      initial_value=ik_cfg.position_weight,
    )
    ori_w_slider = server.gui.add_slider(
      "Orientation Weight",
      min=0.0,
      max=10.0,
      step=0.1,
      initial_value=ik_cfg.orientation_weight,
    )
    jlim_w_slider = server.gui.add_slider(
      "Joint Limit Weight",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=ik_cfg.joint_limit_weight,
    )
    posture_w_slider = server.gui.add_slider(
      "Posture Weight",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=ik_cfg.posture_weight,
    )

  print("=" * 60)
  print("IK Control Demo")
  print("  Open the viser URL printed above")
  print("  Drag the 3D transform control to move the end-effector")
  print("  Use the Gripper Opening slider to open/close")
  print("=" * 60)

  target_action = torch.zeros(1, 7, device=device)
  grip_q = torch.zeros(1, 1, device=device)

  try:
    while True:
      ik_cfg.damping = damping_slider.value
      ik_cfg.position_weight = pos_w_slider.value
      ik_cfg.orientation_weight = ori_w_slider.value
      ik_cfg.joint_limit_weight = jlim_w_slider.value
      ik_cfg.posture_weight = posture_w_slider.value

      p = transform_ctrl.position
      w = transform_ctrl.wxyz
      target_action[0, :3] = torch.tensor([p[0], p[1], p[2]], device=device)
      target_action[0, 3:] = torch.tensor([w[0], w[1], w[2], w[3]], device=device)
      ik_action.process_actions(target_action)

      n_iter = int(iterations_slider.value)
      for _ in range(n_iter):
        dq = ik_action.compute_dq()
        q = entity.data.joint_pos[:, joint_ids] + dq
        entity.write_joint_position_to_sim(q, joint_ids=joint_ids)
        sim.forward()

      grip_q[0, 0] = gripper_slider.value
      entity.write_joint_position_to_sim(grip_q, joint_ids=grip_joint_ids)
      sim.forward()

      scene.update(sim.wp_data)
      if scene.needs_update:
        scene.refresh_visualization()

      time.sleep(1 / 30)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


if __name__ == "__main__":
  main()
