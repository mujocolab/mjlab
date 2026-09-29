"""Tests for control-step sensor wiring in the env loop.

Covers the Scene forwarding hooks (begin_control_step, invalidate_sensor_caches)
and their invocation from ManagerBasedRlEnv.step/reset.
"""

import pytest
import torch
from conftest import get_test_device
from test_contact_sensor import FALLING_BOX_XML, create_scene_with_sensor

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.sensor import Sensor
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


##
# Stub sensors.
##


class _RecordingSensor(Sensor):
  """Minimal sensor that counts begin_control_step calls."""

  def __init__(self, name: str):
    super().__init__()
    self.name = name
    self.begin_calls = 0

  def edit_spec(self, scene_spec, entities):
    del scene_spec, entities

  def initialize(self, mj_model, model, data, device):
    del mj_model, model, data, device

  def _compute_data(self):
    return None

  def begin_control_step(self):
    self.begin_calls += 1


##
# Unit tests.
##


def test_sensor_base_begin_control_step_is_noop():
  """The base Sensor.begin_control_step is a callable no-op."""
  sensor = _RecordingSensor("base")
  sensor.begin_control_step()  # Must not raise (base would, stub overrides).
  assert sensor.begin_calls == 1


def test_scene_begin_control_step_forwards_to_all_sensors(device):
  """Scene.begin_control_step invokes the hook on every registered sensor."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found",),
  )
  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  stub_a = _RecordingSensor("stub_a")
  stub_b = _RecordingSensor("stub_b")
  scene._sensors["stub_a"] = stub_a
  scene._sensors["stub_b"] = stub_b

  scene.begin_control_step()

  assert stub_a.begin_calls == 1
  assert stub_b.begin_calls == 1


def test_scene_invalidate_sensor_caches(device):
  """Scene.invalidate_sensor_caches drops cached data on every sensor."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found",),
  )
  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  _ = scene["box_contact"].data  # Populate the cache.
  sensor = scene["box_contact"]
  assert sensor._cache_valid

  scene.invalidate_sensor_caches()
  assert not sensor._cache_valid


##
# Integration tests (real ManagerBasedRlEnv).
##


def _make_cfg():
  cfg = cartpole_balance_env_cfg()
  cfg.episode_length_s = 0.5
  cfg.scene.num_envs = 4
  # Cart and floor never touch in this scene, so the latch stays clear
  # after begin_control_step unless the env fails to call it.
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="cart_floor_contact",
      primary=ContactMatch(mode="geom", pattern="cart", entity="cartpole"),
      secondary=ContactMatch(mode="geom", pattern="floor", entity="cartpole"),
      fields=("found",),
      catch_substep_contacts=True,
    ),
  )
  return cfg


def test_env_clears_latch_every_control_step(device):
  """The contact latch set before env.step is cleared by the step boundary."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(), device=device)
  env.reset()
  sensor = env.scene["cart_floor_contact"]

  sensor._latch_state["found_any"][:] = True
  obs, _, _, _, _ = env.step(torch.zeros((env.num_envs, 1), device=device))

  assert obs is not None
  data = sensor.data
  assert data.found_any is not None
  assert torch.all(~data.found_any), (
    "latch must be cleared at the control-step boundary"
  )
  env.close()


def test_env_calls_control_step_and_cache_hooks(device, monkeypatch):
  """env.step calls begin_control_step once and invalidates caches after forward."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(), device=device)
  env.reset()

  scene = env.scene
  begin_calls = []
  invalidate_calls = []
  orig_begin = scene.begin_control_step
  orig_invalidate = scene.invalidate_sensor_caches

  monkeypatch.setattr(
    scene,
    "begin_control_step",
    lambda: (begin_calls.append(1), orig_begin()),
  )
  monkeypatch.setattr(
    scene,
    "invalidate_sensor_caches",
    lambda: (invalidate_calls.append(1), orig_invalidate()),
  )

  action = torch.zeros((env.num_envs, 1), device=device)
  env.step(action)
  assert len(begin_calls) == 1
  assert len(invalidate_calls) >= 1

  env.reset()
  assert len(invalidate_calls) >= 2
  env.close()


def test_env_latch_covers_only_latest_control_step(device):
  """The latch read during rewards reflects exactly the latest control step.

  With decimation substeps the latch must accumulate across substeps within
  one env.step (found_any stays False here because the cart never touches the
  floor, but the field must be present and per-step fresh).
  """
  cfg = _make_cfg()
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="cart_floor_contact",
      primary=ContactMatch(mode="geom", pattern="cart", entity="cartpole"),
      secondary=ContactMatch(mode="geom", pattern="floor", entity="cartpole"),
      fields=("found", "force"),
      catch_substep_contacts=True,
    ),
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  for _ in range(3):
    obs, _, _, _, _ = env.step(torch.zeros((env.num_envs, 1), device=device))
    sensor = env.scene["cart_floor_contact"]
    data = sensor.data
    assert data.found_any is not None
    assert data.force_peak is not None
    assert torch.all(~data.found_any)
    assert torch.all(data.force_peak == 0)
  env.close()
