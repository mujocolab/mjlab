"""Tests for sim/registry.py."""

import pytest

from mjlab.sim import registry
from mjlab.sim.registry import (
  get_simulation_backend,
  register_simulation_backend,
)


@pytest.fixture(autouse=True)
def clean_registry():
  """Isolate each test from the module-level backend registry."""
  saved = dict(registry._BACKEND_REGISTRY)
  registry._BACKEND_REGISTRY.clear()
  yield
  registry._BACKEND_REGISTRY.clear()
  registry._BACKEND_REGISTRY.update(saved)


class BackendOne:
  pass


class BackendTwo:
  pass


def test_register_then_get_returns_class():
  register_simulation_backend("one", BackendOne)
  assert get_simulation_backend("one") is BackendOne


def test_register_idempotent_same_class():
  register_simulation_backend("one", BackendOne)
  register_simulation_backend("one", BackendOne)
  assert get_simulation_backend("one") is BackendOne


def test_register_conflicting_class_raises():
  register_simulation_backend("one", BackendOne)
  with pytest.raises(ValueError, match="already registered"):
    register_simulation_backend("one", BackendTwo)


def test_get_unknown_backend_raises():
  with pytest.raises(ValueError, match="Unknown simulation backend"):
    get_simulation_backend("unknown")


def test_get_unknown_lists_registered_names():
  register_simulation_backend("one", BackendOne)
  with pytest.raises(ValueError, match="'one'"):
    get_simulation_backend("unknown")
