"""Tests for MjlabOnPolicyRunner."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import mujoco
import onnx
import pytest
import torch
from conftest import get_test_device
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking.rl.runner import _OnnxMotionModel
from mjlab.terrains import TerrainEntityCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture
def env(device):
  robot_xml = """
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
        <body name="link1" pos="0 0 0">
          <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <motor name="actuator1" joint="joint1" gear="1.0"/>
    </actuator>
  </mujoco>
  """
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
      "critic": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  yield env
  env.close()


def test_runner_persists_common_step_counter(env, device, monkeypatch):
  """MjlabOnPolicyRunner should save and restore common_step_counter."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)
    runner.logger.logger_type = "tensorboard"  # Normally set in learn().

    wrapped_env.unwrapped.common_step_counter = 12345
    checkpoint_path = str(Path(tmpdir) / "test_checkpoint.pt")
    runner.save(checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 0
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 12345


def test_runner_handles_old_checkpoints_without_env_state(env, device):
  """Old checkpoints without env_state should load without crashing."""

  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )

    checkpoint_path = str(Path(tmpdir) / "old_checkpoint.pt")
    old_checkpoint = {
      "actor_state_dict": runner.alg.actor.state_dict(),
      "critic_state_dict": runner.alg.critic.state_dict(),
      "optimizer_state_dict": runner.alg.optimizer.state_dict(),
      "iter": 100,
      "infos": None,
    }
    torch.save(old_checkpoint, checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 999
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 999


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_policy_to_onnx(env, device):
  """runner.export_policy_to_onnx() produces a valid ONNX file."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    runner.export_policy_to_onnx(tmpdir, "test_policy.onnx")
    onnx_path = Path(tmpdir) / "test_policy.onnx"
    assert onnx_path.exists()
    onnx.checker.check_model(str(onnx_path))


def _make_actor(obs_dim=8, output_dim=4, obs_normalization=True):
  obs = TensorDict({"actor": torch.zeros(1, obs_dim)})
  obs_groups = {"actor": ["actor"]}
  return MLPModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=output_dim,
    hidden_dims=[32, 32],
    activation="elu",
    obs_normalization=obs_normalization,
    stochastic=False,
  )


def _train_normalizer(actor, n_batches=50, batch_size=64):
  actor.train()
  for _ in range(n_batches):
    obs = TensorDict({"actor": torch.randn(batch_size, actor.obs_dim) * 5 + 3})
    actor.update_normalization(obs)
  actor.eval()


def _model_output(actor, x_flat):
  obs = TensorDict({"actor": x_flat})
  with torch.no_grad():
    return actor(obs)


def test_onnx_export_matches_actor():
  """as_onnx() model produces the same output as the full actor with normalization."""
  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()
  x = torch.randn(4, actor.obs_dim)
  model_out = _model_output(actor, x)
  with torch.no_grad():
    onnx_out = onnx_model(x)
  torch.testing.assert_close(model_out, onnx_out, atol=1e-6, rtol=0)


def test_onnx_export_without_normalization():
  """as_onnx() works when normalization is disabled."""
  actor = _make_actor(obs_normalization=False)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()
  x = torch.randn(4, actor.obs_dim)
  with torch.no_grad():
    out = onnx_model(x)
  assert out.shape == (4, 4)


# -- CNN (spatial-softmax) ONNX export tests ---------------------------------

_IMG_H, _IMG_W, _IMG_C = 16, 16, 3
_OBS_DIM_1D = 8
_OUTPUT_DIM = 4


def _make_cnn_actor(obs_normalization=True):
  obs = TensorDict(
    {
      "actor": torch.zeros(1, _OBS_DIM_1D),
      "camera": torch.zeros(1, _IMG_C, _IMG_H, _IMG_W),
    }
  )
  obs_groups = {"actor": ["actor", "camera"]}
  cnn_cfg = {
    "output_channels": [8],
    "kernel_size": [3],
    "stride": [1],
    "spatial_softmax_temperature": 1.0,
  }
  return SpatialSoftmaxCNNModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=_OUTPUT_DIM,
    cnn_cfg=cnn_cfg,
    hidden_dims=[32, 32],
    activation="elu",
    obs_normalization=obs_normalization,
    stochastic=False,
  )


def _cnn_model_output(actor, x_1d, x_2d):
  obs = TensorDict({"actor": x_1d, "camera": x_2d})
  with torch.no_grad():
    return actor(obs)


def test_cnn_onnx_export_matches_actor():
  """as_onnx() with SpatialSoftmaxCNNModel matches the original model."""
  actor = _make_cnn_actor(obs_normalization=True)
  # Train normalizer on 1D observations.
  actor.train()
  for _ in range(50):
    obs = TensorDict(
      {
        "actor": torch.randn(64, _OBS_DIM_1D) * 5 + 3,
        "camera": torch.randn(64, _IMG_C, _IMG_H, _IMG_W),
      }
    )
    actor.update_normalization(obs)
  actor.eval()

  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()

  x_1d = torch.randn(4, _OBS_DIM_1D)
  x_2d = torch.randn(4, _IMG_C, _IMG_H, _IMG_W)

  expected = _cnn_model_output(actor, x_1d, x_2d)
  with torch.no_grad():
    actual = onnx_model(x_1d, x_2d)
  torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cnn_onnx_export_to_file():
  """SpatialSoftmaxCNNModel exports to a valid ONNX file."""
  actor = _make_cnn_actor(obs_normalization=False)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.to("cpu")
  onnx_model.eval()

  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = Path(tmpdir) / "cnn_policy.onnx"
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # pyright: ignore[reportCallIssue]
      str(onnx_path),
      export_params=True,
      opset_version=18,
      input_names=onnx_model.input_names,  # pyright: ignore[reportArgumentType]
      output_names=onnx_model.output_names,  # pyright: ignore[reportArgumentType]
      dynamic_axes={},
      dynamo=False,
    )
    assert onnx_path.exists()
    onnx.checker.check_model(str(onnx_path))


class _MockMotion:
  """Minimal mock of a motion object with tensor attributes."""

  def __init__(self, num_steps, num_joints=12, num_bodies=5):
    self.joint_pos = torch.randn(num_steps, num_joints)
    self.joint_vel = torch.randn(num_steps, num_joints)
    self.body_pos_w = torch.randn(num_steps, num_bodies, 3)
    self.body_quat_w = torch.randn(num_steps, num_bodies, 4)
    self.body_lin_vel_w = torch.randn(num_steps, num_bodies, 3)
    self.body_ang_vel_w = torch.randn(num_steps, num_bodies, 3)


def test_onnx_motion_model_policy_matches_actor():
  """_OnnxMotionModel actions output matches calling the actor directly."""

  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)
  motion = _MockMotion(num_steps=50)

  model = _OnnxMotionModel(actor, motion)
  model.eval()

  x = torch.randn(4, actor.obs_dim)
  time_step = torch.tensor([[5]], dtype=torch.float32)

  with torch.no_grad():
    actions, *_ = model(x, time_step)
  expected = _model_output(actor, x)
  torch.testing.assert_close(actions, expected, atol=1e-6, rtol=0)


def test_onnx_motion_model_returns_correct_motion_frame():
  """_OnnxMotionModel returns the motion data at the requested time step."""

  actor = _make_actor(obs_normalization=False)
  motion = _MockMotion(num_steps=50)

  model = _OnnxMotionModel(actor, motion)
  model.eval()

  x = torch.randn(1, actor.obs_dim)
  t = 17
  time_step = torch.tensor([[t]], dtype=torch.float32)

  with torch.no_grad():
    _, joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = model(
      x, time_step
    )

  torch.testing.assert_close(joint_pos, motion.joint_pos[t : t + 1])
  torch.testing.assert_close(joint_vel, motion.joint_vel[t : t + 1])
  torch.testing.assert_close(body_pos, motion.body_pos_w[t : t + 1])
  torch.testing.assert_close(body_quat, motion.body_quat_w[t : t + 1])
  torch.testing.assert_close(body_lin_vel, motion.body_lin_vel_w[t : t + 1])
  torch.testing.assert_close(body_ang_vel, motion.body_ang_vel_w[t : t + 1])


def test_onnx_motion_model_clamps_out_of_bounds_time_step():
  """_OnnxMotionModel clamps time_step beyond motion length to last frame."""

  num_steps = 20
  actor = _make_actor(obs_normalization=False)
  motion = _MockMotion(num_steps=num_steps)

  model = _OnnxMotionModel(actor, motion)
  model.eval()

  x = torch.randn(1, actor.obs_dim)
  time_step = torch.tensor([[999]], dtype=torch.float32)

  with torch.no_grad():
    _, joint_pos, *_ = model(x, time_step)

  torch.testing.assert_close(joint_pos, motion.joint_pos[num_steps - 1 : num_steps])
