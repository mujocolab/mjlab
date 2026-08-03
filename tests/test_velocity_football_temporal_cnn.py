"""Tests for the football causal/dilated temporal encoder."""

from typing import Any, cast

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict

from mjlab.tasks.velocity_football.rl.conv1d_encoder import Conv1dEncoder
from mjlab.tasks.velocity_football.rl.temporal_cnn_model import TemporalCNNModel


def test_b1_causal_dilated_encoder_outputs_last_step_latent() -> None:
  encoder = Conv1dEncoder(
    input_channels=7,
    output_channels=(64, 64, 64),
    kernel_size=3,
    dilations=(1, 2, 4),
    causal=True,
    output_mode="last",
  )

  output = encoder(torch.randn(3, 7, 10))

  assert output.shape == (3, 64)
  convolutions = [layer for layer in encoder.net if isinstance(layer, torch.nn.Conv1d)]
  assert [layer.dilation for layer in convolutions] == [(1,), (2,), (4,)]
  assert [layer.padding for layer in convolutions] == [(0,), (0,), (0,)]


def test_b1_temporal_model_exports_dual_input_onnx(tmp_path) -> None:
  obs = TensorDict(
    {
      "actor": torch.zeros(2, 490),
      "actor_history": torch.zeros(2, 10, 7),
    },
    batch_size=[2],
  )
  model = TemporalCNNModel(
    obs=obs,
    obs_groups={"actor": ["actor", "actor_history"]},
    obs_set="actor",
    output_dim=29,
    hidden_dims=(32, 16),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    cnn_cfg={
      "output_channels": (64, 64, 64),
      "kernel_size": 3,
      "dilations": (1, 2, 4),
      "causal": True,
      "output_mode": "last",
    },
  )
  onnx_model = cast(Any, model.as_onnx())
  output_path = tmp_path / "b1.onnx"
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    output_path,
    opset_version=18,
    input_names=onnx_model.input_names,
    output_names=onnx_model.output_names,
    dynamo=False,
  )

  session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
  assert [(item.name, item.shape) for item in session.get_inputs()] == [
    ("obs", [1, 490]),
    ("obs_history", [1, 10, 7]),
  ]
  result = np.asarray(
    session.run(
      None,
      {
        "obs": np.zeros((1, 490), dtype=np.float32),
        "obs_history": np.zeros((1, 10, 7), dtype=np.float32),
      },
    )[0]
  )
  assert result.shape == (1, 29)
