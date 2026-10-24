"""Data-only alias and custom component protocol tests."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import warp as wp

from mjlab.rl.training_state import ComponentState


def runtime(memory) -> Any:
  sim = SimpleNamespace(
    wp_model=object(),
    wp_data=object(),
    model=object(),
    data=object(),
    _default_model_fields={},
    _world_to_variant={},
  )
  return SimpleNamespace(sim=sim, device="cpu", memory=memory)


def restore(source, destination):
  snapshot = ComponentState(source).capture()
  ComponentState(destination).restore(snapshot, validate=True)
  ComponentState(destination).restore(snapshot)


def test_mutable_cycle_retains_identity():
  source = runtime([])
  source.memory.append(source.memory)
  destination = runtime([])
  destination.memory.append(destination.memory)
  restore(source, destination)
  assert destination.memory[0] is destination.memory


def test_existing_tensor_view_keeps_its_storage():
  base = torch.arange(4.0)
  source = runtime({"base": base, "view": base[::2]})
  destination_base = torch.zeros(4)
  destination = runtime({"base": destination_base, "view": destination_base[::2]})
  restore(source, destination)
  assert destination.memory["base"] is destination_base
  destination.memory["base"][0] = 99
  assert destination.memory["view"][0] == 99


def test_native_logger_scalar_view_reuses_existing_allocation():
  source_scalar = torch.tensor(3.0)
  source = runtime({"scalar": source_scalar, "log_view": source_scalar.unsqueeze(0)})
  destination_scalar = torch.tensor(0.0)
  destination = runtime({"scalar": destination_scalar, "log_view": destination_scalar})
  restore(source, destination)
  assert destination.memory["scalar"] is destination_scalar
  assert destination.memory["log_view"].shape == (1,)
  destination_scalar.fill_(19)
  assert destination.memory["log_view"][0] == 19


def test_lazy_aliased_tensors_fail_before_mutation():
  base = torch.arange(4.0)
  source = runtime({"base": base, "view": base[::2]})
  destination = runtime({"base": None, "view": None})
  with pytest.raises(ValueError, match="explicit training-state hook"):
    ComponentState(destination).restore(ComponentState(source).capture(), validate=True)
  assert destination.memory == {"base": None, "view": None}


def test_incompatible_existing_aliases_are_rejected():
  base = torch.arange(4.0)
  source = runtime({"base": base, "view": base[::2]})
  destination = runtime({"base": torch.zeros(4), "view": torch.zeros(4)[::2]})
  with pytest.raises(ValueError, match="storage aliases differ"):
    ComponentState(destination).restore(ComponentState(source).capture(), validate=True)


def test_distinct_source_storages_cannot_collapse_into_one_target():
  source = runtime({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
  shared = torch.zeros(1)
  destination = runtime({"a": shared, "b": shared})
  with pytest.raises(ValueError, match="storage aliases differ"):
    ComponentState(destination).restore(ComponentState(source).capture(), validate=True)
  assert shared.item() == 0


def test_new_tuple_data_keys_can_be_restored():
  source = runtime({(1, "phase"): torch.ones(2)})
  destination = runtime({})
  restore(source, destination)
  assert torch.equal(destination.memory[(1, "phase")], torch.ones(2))


def test_custom_term_hook_restores_private_memory():
  class Term:
    def __init__(self, value):
      self._counter = value

    def training_state_dict(self):
      return {"counter": self._counter}

    def load_training_state_dict(self, state):
      self._counter = state["counter"]

  source = runtime(Term(19))
  destination = runtime(Term(0))
  restore(source, destination)
  assert destination.memory._counter == 19


def _stateless_reward(scale, default=0):
  def reward(env=None, offset=default, *, gain=1):
    return scale * gain + offset

  reward.__dict__["training_state_stateless"] = True
  return reward


@pytest.mark.parametrize("changed", ["closure", "default", "kwdefault"])
def test_custom_function_captured_configuration_must_match(changed):
  source = runtime(_stateless_reward(1))
  destination = runtime(_stateless_reward(2 if changed == "closure" else 1))
  if changed == "default":
    destination.memory.__defaults__ = (None, 2)
  if changed == "kwdefault":
    destination.memory.__kwdefaults__ = {"gain": 2}
  with pytest.raises(ValueError, match="static component differs"):
    ComponentState(destination).restore(ComponentState(source).capture(), validate=True)


def test_stateless_function_mutable_closure_requires_explicit_component():
  source = runtime(_stateless_reward([1]))
  with pytest.raises(ValueError, match="immutable data"):
    ComponentState(source).capture()


def test_zero_dimensional_numpy_buffer_restores_in_place():
  source = runtime(np.array(3.0))
  buffer = np.array(0.0)
  destination = runtime(buffer)
  restore(source, destination)
  assert buffer.item() == 3


def test_warp_buffer_dtype_mismatch_rejected_before_mutation():
  source = runtime(wp.array(np.ones(2, dtype=np.float32), device="cpu"))
  buffer = wp.array(np.zeros(2, dtype=np.int32), device="cpu")
  destination = runtime(buffer)
  with pytest.raises(ValueError, match="Warp array shape/dtype differs"):
    ComponentState(destination).restore(ComponentState(source).capture(), validate=True)
  np.testing.assert_array_equal(buffer.numpy(), np.zeros(2, dtype=np.int32))
