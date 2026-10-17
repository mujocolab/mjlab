"""Tests for the tracking evaluation script."""

from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

import mjlab.tasks.tracking.scripts.evaluate as evaluate
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity

_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"
_EPISODE_S = 1.0
_SPEED = 2.0  # Reference root speed along +x [m/s].


def _write_sliding_motion(path: Path) -> None:
  """Write a G1 motion whose root slides along +x in the default pose."""
  model = Entity(get_g1_robot_cfg()).compile()
  data = mujoco.MjData(model)
  fps = 50
  num_frames = int(_EPISODE_S * fps) + 50  # Longer than the episode.
  joint_pos, body_pos, body_quat = [], [], []
  for t in range(num_frames):
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    data.qpos[0] += _SPEED * t / fps
    mujoco.mj_forward(model, data)
    joint_pos.append(data.qpos[7:].copy())
    body_pos.append(data.xpos[1:].copy())
    body_quat.append(data.xquat[1:].copy())
  body_lin_vel = np.zeros((num_frames, len(body_pos[0]), 3))
  body_lin_vel[..., 0] = _SPEED
  np.savez(
    path,
    fps=[fps],
    joint_pos=np.array(joint_pos),
    joint_vel=np.zeros((num_frames, len(joint_pos[0]))),
    body_pos_w=np.array(body_pos),
    body_quat_w=np.array(body_quat),
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=np.zeros_like(body_lin_vel),
  )


class _OracleRunner:
  """Runner whose policy writes the reference state into the sim each step."""

  def __init__(self, env, *args, **kwargs):
    self.env = env.unwrapped

  def load(self, *args, **kwargs):
    return {}

  def get_inference_policy(self, device=None):
    env = self.env
    cmd = env.command_manager.get_term("motion")
    ids = torch.arange(env.num_envs, device=env.device)

    def policy(obs):
      cmd._write_reference_state_to_sim(
        ids,
        cmd.body_pos_w[:, 0],
        cmd.body_quat_w[:, 0],
        cmd.body_lin_vel_w[:, 0],
        cmd.body_ang_vel_w[:, 0],
        cmd.joint_pos,
        cmd.joint_vel,
      )
      return torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
      )

    return policy


@pytest.mark.slow
def test_evaluate_does_not_score_post_reset_state(tmp_path, monkeypatch):
  """An env's final step must not be scored against its auto-reset robot.

  The oracle puts the robot on the reference each step, then physics carries it
  one step ahead, so every honest sample is about _SPEED * step_dt. Scoring the
  post-reset robot (back at the start, _SPEED * _EPISODE_S away) as well adds
  another _SPEED * step_dt to the mean and doubles the metric.
  """
  _write_sliding_motion(tmp_path / "motion.npz")

  artifact = Mock(type="motions", download=Mock(return_value=str(tmp_path)))
  api = Mock()
  api.run.return_value.used_artifacts.return_value = [artifact]
  monkeypatch.setattr(evaluate.wandb, "Api", Mock(return_value=api))
  monkeypatch.setattr(
    evaluate, "get_wandb_checkpoint_path", Mock(return_value=(Path("x.pt"), True))
  )
  monkeypatch.setattr(evaluate, "load_runner_cls", Mock(return_value=_OracleRunner))
  load_env_cfg = evaluate.load_env_cfg
  step_dts = []

  def short_episode_env_cfg(task_id, play=False):
    cfg = load_env_cfg(task_id, play=play)
    cfg.episode_length_s = _EPISODE_S
    step_dts.append(cfg.sim.mujoco.timestep * cfg.decimation)
    return cfg

  monkeypatch.setattr(evaluate, "load_env_cfg", short_episode_env_cfg)

  metrics = evaluate.run_evaluate(
    _TASK_ID,
    evaluate.EvaluateConfig(
      wandb_run_path="entity/project/run", num_envs=2, device=get_test_device()
    ),
  )

  assert metrics["success_rate"] == 1.0
  expected = _SPEED * step_dts[0]
  assert metrics["mpkpe"] == pytest.approx(expected, rel=0.25)
  assert metrics["r_mpkpe"] == pytest.approx(expected, rel=0.25)
