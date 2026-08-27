"""Tests for temporal-depth football perception and supervised training."""

from types import SimpleNamespace
from typing import Any, cast

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity_football_depth import V1_TASK_ID
from mjlab.tasks.velocity_football_depth.algorithm import BallAuxiliaryPPO
from mjlab.tasks.velocity_football_depth.env_cfg import (
  DEPTH_HISTORY_LENGTH,
  DEPTH_POLICY_HEIGHT,
  DEPTH_POLICY_WIDTH,
  DEPTH_SENSOR_NAME,
)
from mjlab.tasks.velocity_football_depth.model import DepthAuxCNNModel
from mjlab.tasks.velocity_football_depth.runner import (
  DepthAuxVelocityOnPolicyRunner,
)


def _distribution_cfg() -> dict[str, Any]:
  return {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }


def test_v1_task_contract() -> None:
  env_cfg = load_env_cfg(V1_TASK_ID)
  rl_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(V1_TASK_ID))
  camera = cast(
    CameraSensorCfg,
    next(
      sensor
      for sensor in env_cfg.scene.sensors or ()
      if sensor.name == DEPTH_SENSOR_NAME
    ),
  )

  assert camera.data_types == ("depth", "segmentation")
  assert env_cfg.observations["depth"].history_length == DEPTH_HISTORY_LENGTH
  assert not env_cfg.observations["depth"].flatten_history_dim
  assert tuple(env_cfg.observations["ball_aux_target"].terms) == ("target",)
  assert rl_cfg.obs_groups == {
    "actor": ("actor", "depth"),
    "critic": ("critic", "depth", "critic_ball"),
  }
  assert "BallAuxiliaryPPO" in rl_cfg.algorithm.class_name
  assert load_runner_cls(V1_TASK_ID) is DepthAuxVelocityOnPolicyRunner


def test_depth_model_outputs_policy_and_supervised_ball_bottleneck() -> None:
  batch_size = 4
  obs = TensorDict(
    {
      "actor": torch.randn(batch_size, 490),
      "depth": torch.rand(
        batch_size,
        DEPTH_HISTORY_LENGTH,
        DEPTH_POLICY_HEIGHT,
        DEPTH_POLICY_WIDTH,
      ),
    },
    batch_size=[batch_size],
  )
  model = DepthAuxCNNModel(
    obs=obs,
    obs_groups={"actor": ["actor", "depth"]},
    obs_set="actor",
    output_dim=29,
    hidden_dims=(64, 32),
    obs_normalization=True,
    distribution_cfg=_distribution_cfg(),
    cnn_cfg={"output_channels": (8, 16, 32), "latent_dim": 32},
  )

  actions = model(obs)
  prediction = model.auxiliary_prediction
  assert actions.shape == (batch_size, 29)
  assert prediction.shape == (batch_size, 4)
  assert torch.all(prediction[:, :2].abs() <= 1.0)
  assert torch.all((prediction[:, 2] >= 0.0) & (prediction[:, 2] <= 1.0))
  assert model.mlp[0].in_features == 490 + 32 + 4

  exported = model.as_onnx()
  with torch.no_grad():
    exported_actions = exported(obs["actor"], obs["depth"])
  assert exported_actions.shape == (batch_size, 29)


def test_auxiliary_loss_masks_hidden_ball_position() -> None:
  prediction = torch.tensor(
    [[0.2, -0.1, 0.3, 2.0], [1.0, 1.0, 1.0, -2.0]], requires_grad=True
  )
  actor = SimpleNamespace(auxiliary_prediction=prediction)
  observations = {
    "ball_aux_target": torch.tensor([[0.0, 0.0, 0.2, 1.0], [-1.0, -1.0, 0.0, 0.0]])
  }
  algorithm = object.__new__(BallAuxiliaryPPO)
  algorithm.auxiliary_target_group = "ball_aux_target"
  algorithm.auxiliary_position_coef = 1.0
  algorithm.auxiliary_visibility_coef = 0.2

  total, position, visibility = algorithm._auxiliary_loss(  # noqa: SLF001
    cast(Any, actor), observations
  )
  total.backward()

  assert total.item() > 0.0
  assert position.item() > 0.0
  assert visibility.item() > 0.0
  assert prediction.grad is not None
  assert torch.count_nonzero(prediction.grad[1, :3]) == 0


class _FakeActor:
  def __init__(self, state: dict[str, torch.Tensor]) -> None:
    self.state = state

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in self.state.items()}

  def load_state_dict(
    self, state: dict[str, torch.Tensor], strict: bool = True
  ) -> None:
    assert strict
    self.state = {key: value.clone() for key, value in state.items()}


def test_v1_walk_transfer_preserves_policy_and_initializes_visual_columns(
  tmp_path,
) -> None:
  source = {
    "obs_normalizer._mean": torch.zeros(1, 490),
    "obs_normalizer._var": torch.ones(1, 490),
    "obs_normalizer._std": torch.ones(1, 490),
    "obs_normalizer.count": torch.tensor(10),
    "distribution.std_param": torch.ones(29),
    "mlp.0.weight": torch.randn(8, 490),
    "mlp.0.bias": torch.randn(8),
    "mlp.2.weight": torch.randn(29, 8),
    "mlp.2.bias": torch.randn(29),
  }
  target = {
    **{key: value.clone() for key, value in source.items()},
    "mlp.0.weight": torch.randn(8, 526),
    "cnns.depth.features.0.weight": torch.randn(4, 5, 3, 3),
  }
  actor = _FakeActor(target)
  runner = object.__new__(DepthAuxVelocityOnPolicyRunner)
  untyped_runner = cast(Any, runner)
  untyped_runner.alg = SimpleNamespace(actor=actor)
  untyped_runner.device = "cpu"
  checkpoint = tmp_path / "walk.pt"
  torch.save({"actor_state_dict": source}, checkpoint)

  runner.load_pretrained(str(checkpoint))
  actual = actor.state_dict()
  torch.testing.assert_close(actual["mlp.0.weight"][:, :490], source["mlp.0.weight"])
  assert torch.count_nonzero(actual["mlp.0.weight"][:, 490:]) == 0
  torch.testing.assert_close(
    actual["cnns.depth.features.0.weight"], target["cnns.depth.features.0.weight"]
  )
