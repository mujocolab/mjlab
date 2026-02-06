"""Test that the ONNX exporter applies observation normalization exactly once."""

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def _make_actor(obs_dim=8, output_dim=4, obs_normalization=True):
  """Create an MLPModel actor with a single obs group."""
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
  """Feed random data through the normalizer so it has non-trivial statistics."""
  actor.train()
  for _ in range(n_batches):
    obs = TensorDict({"actor": torch.randn(batch_size, actor.obs_dim) * 5 + 3})
    actor.update_normalization(obs)
  actor.eval()


def _model_output(actor, x_flat):
  """Get actor output via the full MLPModel forward (normalizes then MLP)."""
  obs = TensorDict({"actor": x_flat})
  with torch.no_grad():
    return actor(obs)


def _exporter_output(actor, normalizer, x_flat):
  """Get output via _OnnxPolicyExporter (the path used for ONNX export)."""
  exporter = _OnnxPolicyExporter(actor, normalizer=normalizer)
  exporter.eval()
  with torch.no_grad():
    return exporter(x_flat)


def test_exporter_with_normalizer_matches_model():
  """Exporter with normalizer must produce the same output as the full model.

  MLPModel.forward does: obs_normalizer(x) -> mlp(x).
  _OnnxPolicyExporter extracts .mlp (raw MLP without normalizer) and applies
  the normalizer separately: normalizer(x) -> mlp(x).
  These two paths must agree.
  """
  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)

  x = torch.randn(4, actor.obs_dim)
  model_out = _model_output(actor, x)
  exporter_out = _exporter_output(actor, normalizer=actor.obs_normalizer, x_flat=x)

  assert torch.allclose(model_out, exporter_out, atol=1e-6), (
    f"Exporter output diverges from model output.\n"
    f"Max diff: {(model_out - exporter_out).abs().max().item()}"
  )


def test_exporter_without_normalizer_skips_normalization():
  """Exporter with normalizer=None skips normalization.

  Passing normalizer=None means no normalization is applied,
  which would be wrong for models trained with obs normalization.
  """
  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)

  x = torch.randn(4, actor.obs_dim)
  model_out = _model_output(actor, x)
  exporter_out = _exporter_output(actor, normalizer=None, x_flat=x)

  assert not torch.allclose(model_out, exporter_out, atol=1e-4), (
    "Exporter without normalizer should NOT match model output, "
    "but they match -- normalization is being applied somewhere unexpected."
  )
