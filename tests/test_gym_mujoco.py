"""Native articulation, observation-oracle and Warp lifecycle checks."""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointEffortAction
from mjlab.tasks.gym_mujoco.config import TASK_CONFIGS
from mjlab.utils.lab_api.math import matrix_from_quat

gym = pytest.importorskip("gymnasium")


def _coordinates(asset):
  q = asset.data.joint_pos[0]
  v = asset.data.joint_vel[0]
  if not asset.is_fixed_base:
    q = torch.cat((asset.data.root_link_pose_w[0], q))
    v = torch.cat(
      (asset.data.root_link_lin_vel_w[0], asset.data.root_link_ang_vel_b[0], v)
    )
  return q.cpu().numpy(), v.cpu().numpy()


@pytest.mark.slow
@pytest.mark.parametrize("gym_id", tuple(TASK_CONFIGS))
def test_native_warp_observations_actions_and_resets(gym_id):
  cfg = TASK_CONFIGS[gym_id](num_envs=2)
  benchmark_episode_steps = round(
    cfg.episode_length_s / (cfg.decimation * cfg.sim.mujoco.timestep)
  )
  # Compare the quaternion representation to Gym at the same forwarded state.
  # Production configs use rotation-6D and have a separate check below.
  for group in cfg.observations.values():
    if "root_orientation" in group.terms:
      group.terms["root_orientation"].params["representation"] = "quaternion"
  cfg.episode_length_s = 2 * cfg.decimation * cfg.sim.mujoco.timestep
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    obs, _ = env.reset(seed=13)
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    robot = env.scene["robot"]
    assert robot.is_actuated and robot.is_articulated
    with gym.make(gym_id.removeprefix("Mjlab-Gym-")) as wrapped:
      ref = wrapped.unwrapped
      assert cfg.decimation == ref.frame_skip
      assert benchmark_episode_steps == wrapped.spec.max_episode_steps
      # Scene attachment and explicit cfg overrides must preserve XML physics.
      actual_opt = env.sim.mj_model.opt
      for field in ("gravity", "timestep", "integrator", "density", "viscosity"):
        np.testing.assert_allclose(
          getattr(actual_opt, field), getattr(ref.model.opt, field)
        )
      assert actor_obs.shape == (2, *ref.observation_space.shape)
      assert env.single_action_space.shape == ref.action_space.shape
      q, v = _coordinates(robot)
      ref.set_state(q, v)
      mujoco.mj_rnePostConstraint(ref.model, ref.data)
      expected_obs = ref._get_obs()
      if gym_id in ("Mjlab-Gym-Hopper-v5", "Mjlab-Gym-Walker2d-v5"):
        # Agreed constant reference shift; physical qpos/reset are unchanged.
        expected_obs[0] -= 1.25
      np.testing.assert_allclose(actor_obs[0].cpu(), expected_obs, rtol=2e-4, atol=2e-4)
      # Check masses, including HalfCheetah's compiler normalization.
      for original in range(1, ref.model.nbody):
        name = ref.model.body(original).name or f"body_{original}"
        actual = env.sim.mj_model.body(f"robot/{name}").id
        np.testing.assert_allclose(
          env.sim.mj_model.body_mass[actual], ref.model.body_mass[original]
        )
      # Non-uniform controls expose actuator permutation errors (not just shape).
      values = np.linspace(-0.1, 0.1, ref.model.nu, dtype=np.float32)
      action = torch.tensor(values, device=env.device).repeat(2, 1)
      env.action_manager.process_action(action)
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      motor_joints = tuple(
        ref.model.joint(int(j)).name for j in ref.model.actuator_trnid[:, 0]
      )
      action_term = env.action_manager.get_term("joint_effort")
      assert isinstance(action_term, JointEffortAction)
      action_joints = action_term.target_names
      controls = np.array([values[action_joints.index(name)] for name in motor_joints])
      np.testing.assert_allclose(env.sim.data.ctrl[0].cpu(), controls, atol=1e-7)
      env.sim.forward()
      ref.data.ctrl[:] = controls
      mujoco.mj_forward(ref.model, ref.data)
      start = 0 if robot.is_fixed_base else 6
      np.testing.assert_allclose(
        robot.data.qfrc_actuator[0].cpu(),
        ref.data.qfrc_actuator[start:],
        rtol=2e-5,
        atol=2e-5,
      )
      # Compare reward composition at the same state/control. Forward rewards
      # deliberately use instantaneous velocity here, not Gym's step difference.
      mujoco.mj_rnePostConstraint(ref.model, ref.data)
      env.termination_manager.compute()
      task = gym_id.removeprefix("Mjlab-Gym-")
      if task in ("Reacher-v5", "Pusher-v5"):
        expected_reward, _ = ref._get_rew(controls)
      elif task == "InvertedDoublePendulum-v5":
        tip = ref.data.site_xpos[0]
        expected_reward, _ = ref._get_rew(tip[0], tip[2], tip[2] <= 1.0)
      elif task == "InvertedPendulum-v5":
        expected_reward = float(
          np.isfinite(q).all() and np.isfinite(v).all() and abs(q[1]) <= 0.2
        )
      elif task == "HumanoidStandup-v5":
        expected_reward, _ = ref._get_rew(ref.data.qpos[2], controls)
      else:
        velocity = ref.data.qvel[0]
        if task == "Humanoid-v5":
          mujoco.mj_subtreeVel(ref.model, ref.data)
          velocity = ref.data.subtree_linvel[1, 0]
        expected_reward, _ = ref._get_rew(velocity, controls)
      actual_reward = env.reward_manager.compute(env.step_dt)[0].item()
      np.testing.assert_allclose(actual_reward, expected_reward, rtol=2e-4, atol=2e-4)
    zero = torch.zeros_like(action)
    obs, reward, _, truncated, _ = env.step(zero)
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert torch.isfinite(actor_obs).all() and torch.isfinite(reward).all()
    assert not truncated.any()
    _, _, _, truncated, _ = env.step(zero)
    assert truncated.all() and (env.episode_length_buf == 0).all()
    retained = robot.data.joint_pos[1].clone()
    env.reset(env_ids=torch.tensor([0], device=env.device))
    torch.testing.assert_close(robot.data.joint_pos[1], retained)
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize(
  "gym_id",
  ["Mjlab-Gym-Ant-v5", "Mjlab-Gym-Humanoid-v5", "Mjlab-Gym-HumanoidStandup-v5"],
)
def test_rotation_6d(gym_id):
  env = ManagerBasedRlEnv(TASK_CONFIGS[gym_id](num_envs=2), device=get_test_device())
  try:
    obs, _ = env.reset(seed=2)
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    quat = env.scene["robot"].data.root_link_quat_w
    expected = matrix_from_quat(quat)[..., :, :2].transpose(-2, -1).flatten(1)
    # The first term is height, followed by 6D rotation.
    torch.testing.assert_close(actor_obs[:, 1:7], expected)
    assert torch.isfinite(actor_obs).all()
  finally:
    env.close()


@pytest.mark.slow
def test_reacher_target_command_lifecycle():
  cfg = TASK_CONFIGS["Mjlab-Gym-Reacher-v5"](num_envs=2)
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    obs, _ = env.reset(seed=42)
    target = env.command_manager.get_command("target")
    assert target is not None
    initial = target.clone()
    assert (initial.norm(dim=-1) < 0.2).all()
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    torch.testing.assert_close(actor_obs[:, 4:6], target)
    robot = env.scene["robot"]
    target_body, _ = robot.find_bodies("target")
    torch.testing.assert_close(
      robot.data.body_link_pos_w[:, target_body[0], :2], target
    )
    for _ in range(5):
      env.step(torch.zeros(2, 2, device=env.device))
    torch.testing.assert_close(target, initial)
    env.reset(env_ids=torch.tensor([0], device=env.device))
    torch.testing.assert_close(target[1], initial[1])
    assert not torch.equal(target[0], initial[0])
    torch.testing.assert_close(
      robot.data.body_link_pos_w[:, target_body[0], :2], target
    )
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize(
  "name,heights",
  [
    ("Ant", (0.15, 0.25, 0.95, 1.05)),
    ("Hopper", (0.65, 0.75, 1.25)),
    ("Walker2d", (0.75, 0.85, 1.95, 2.05)),
    ("Humanoid", (0.95, 1.05, 1.95, 2.05)),
  ],
)
def test_health_thresholds_match_gym(name, heights):
  gym = pytest.importorskip("gymnasium")
  env = ManagerBasedRlEnv(
    TASK_CONFIGS[f"Mjlab-Gym-{name}-v5"](num_envs=1), device=get_test_device()
  )
  try:
    env.reset(seed=9)
    robot = env.scene["robot"]
    q, v = _coordinates(robot)
    with gym.make(f"{name}-v5") as wrapped:
      ref = wrapped.unwrapped
      for height in heights:
        q[1 if robot.is_fixed_base else 2] = height
        q_tensor = torch.tensor(q, device=env.device).unsqueeze(0)
        if robot.is_fixed_base:
          robot.write_joint_position_to_sim(q_tensor)
        else:
          robot.write_root_link_pose_to_sim(q_tensor[:, :7])
        env.sim.forward()
        ref.set_state(q, v)
        env.termination_manager.compute()
        assert bool(env.termination_manager.terminated[0]) == (not ref.is_healthy)
  finally:
    env.close()
