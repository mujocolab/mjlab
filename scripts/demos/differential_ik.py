"""Interactive IK control demo.

Drag the 3D transform control in the viser viewer to move the YAM end-effector.

Run with:
  MJLAB_WARP_QUIET=1 uv run scripts/demos/differential_ik.py
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

IK_ITERATIONS = 10


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  robot_cfg = get_yam_robot_cfg()
  robot_cfg.init_state = DEMO_INIT_STATE
  entity = Entity(robot_cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg(mujoco=MujocoCfg(gravity=(0, 0, -9.81)))
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
  sim.forward()

  env = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
  ik_cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="grasp_site",
    frame_type="site",
    posture_weight=0.02,
    joint_limit_weight=1e-1,
    damping=1e-1,
    use_relative_mode=False,
  )
  ik_action: DifferentialIKAction = ik_cfg.build(env)  # type: ignore[arg-type]
  joint_ids = ik_action._joint_ids

  grip_ids, _ = entity.find_joints("left_finger")
  grip_joint_ids = torch.tensor(grip_ids, device=device, dtype=torch.long)
  grip_open = torch.tensor([[0.037]], device=device)

  server = viser.ViserServer(label="IK Control Demo")
  scene = ViserMujocoScene.create(server, sim.mj_model, num_envs=1)
  scene.create_visualization_gui(
    camera_distance=0.1,
    camera_azimuth=135.0,
    camera_elevation=30.0,
  )

  site_id = ik_action._frame_id
  pos = sim.data.site_xpos[0, site_id].cpu().numpy()
  xmat = sim.data.site_xmat[0, site_id]
  quat = quat_from_matrix(xmat).cpu().numpy()

  transform_ctrl = server.scene.add_transform_controls(
    "/ik_target",
    position=(float(pos[0]), float(pos[1]), float(pos[2])),
    wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    scale=0.12,
  )

  needs_reset = [False]

  with server.gui.add_folder("IK Control"):
    reset_button = server.gui.add_button("Reset")
    reset_button.on_click(lambda _: needs_reset.__setitem__(0, True))
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
  print("=" * 60)

  target_action = torch.zeros(1, 7, device=device)

  def _reset() -> None:
    entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
    sim.forward()
    ik_action.reset()
    p = sim.data.site_xpos[0, site_id].cpu().numpy()
    q = quat_from_matrix(sim.data.site_xmat[0, site_id]).cpu().numpy()
    transform_ctrl.position = (float(p[0]), float(p[1]), float(p[2]))
    transform_ctrl.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

  try:
    while True:
      if needs_reset[0]:
        needs_reset[0] = False
        _reset()

      ik_cfg.damping = max(damping_slider.value, 1e-2)
      ik_cfg.position_weight = max(pos_w_slider.value, 0.0)
      ik_cfg.orientation_weight = max(ori_w_slider.value, 0.0)
      ik_cfg.joint_limit_weight = max(jlim_w_slider.value, 0.0)
      ik_cfg.posture_weight = max(posture_w_slider.value, 0.0)

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
        entity.write_joint_position_to_sim(grip_open, joint_ids=grip_joint_ids)
        sim.forward()

      scene.update(sim.wp_data)
      if scene.needs_update:
        scene.refresh_visualization()

      time.sleep(1.0 / 30.0)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


if __name__ == "__main__":
  main()
