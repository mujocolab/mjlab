"""Tests for the recorder manager."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import (
  NullRecorderManager,
  RecorderManager,
  RecorderTerm,
  RecorderTermCfg,
)
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg


class _CountingRecorder(RecorderTerm):
  """Records calls to each lifecycle method for inspection."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.pre_reset_calls: list[torch.Tensor] = []
    self.post_reset_calls: list[torch.Tensor] = []
    self.post_step_count = 0
    self.closed = False

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    self.pre_reset_calls.append(env_ids.clone())

  def record_post_reset(self, env_ids: torch.Tensor) -> None:
    self.post_reset_calls.append(env_ids.clone())

  def record_post_step(self) -> None:
    self.post_step_count += 1

  def close(self) -> None:
    self.closed = True


@pytest.fixture
def mock_env() -> Mock:
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  return env


@pytest.fixture
def manager_and_term(mock_env: Mock) -> tuple[RecorderManager, _CountingRecorder]:
  cfg = {"recorder": RecorderTermCfg(func=_CountingRecorder, params={})}
  manager = RecorderManager(cfg, mock_env)
  term = manager._terms[0]
  assert isinstance(term, _CountingRecorder)
  return manager, term


# Lifecycle semantics.


def test_terminal_action_available_at_record_pre_reset(mock_env: Mock) -> None:
  """record_pre_reset fires before _reset_idx, so action_manager.action is intact.

  In step(), the sequence is:
    record_pre_reset(reset_env_ids)   <- action still set
    _reset_idx(reset_env_ids)         <- zeroes action_manager.action
    ...
    record_post_step()                <- action is now 0 for reset envs

  A recorder that wants the terminal action MUST capture it at record_pre_reset.
  """
  seen: dict[str, torch.Tensor] = {}

  class ActionCapture(RecorderTerm):
    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
      seen["pre_reset_action"] = self._env.action_manager.action[env_ids].clone()

    def record_post_step(self) -> None:
      seen["post_step_action"] = self._env.action_manager.action.clone()

  mock_env.action_manager = Mock()
  mock_env.action_manager.action = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
  )

  cfg = {"r": RecorderTermCfg(func=ActionCapture, params={})}
  manager = RecorderManager(cfg, mock_env)

  reset_env_ids = torch.tensor([0, 2])

  # Simulate step(): record_pre_reset fires before _reset_idx zeros actions.
  manager.record_pre_reset(reset_env_ids)
  assert torch.equal(
    seen["pre_reset_action"],
    torch.tensor([[1.0, 2.0], [5.0, 6.0]]),
  ), "Terminal action must be intact at record_pre_reset"

  # Simulate _reset_idx: zeros action for reset envs.
  mock_env.action_manager.action[reset_env_ids] = 0.0

  manager.record_post_step()
  assert torch.all(seen["post_step_action"][reset_env_ids] == 0.0), (
    "Action is zeroed for reset envs at record_post_step; use record_pre_reset"
    " to capture terminal actions"
  )
  # Non-reset envs retain their action.
  non_reset = torch.tensor([1, 3])
  assert torch.equal(
    seen["post_step_action"][non_reset],
    torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
  )


def test_record_pre_reset_obs_buf_is_previous_step(mock_env: Mock) -> None:
  """obs_buf at record_pre_reset is the previous step's observation, not the
  post-action terminal state. It is the input obs the agent used to pick the
  terminal action.
  """
  seen: dict[str, torch.Tensor] = {}

  # Use a plain tensor for obs_buf to avoid dict-subscript complexity in the
  # type checker. The point of this test is lifecycle timing, not data structure.
  class ObsCapture(RecorderTerm):
    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
      obs: torch.Tensor = self._env.obs_buf  # type: ignore[assignment]
      seen["pre_reset_obs"] = obs[env_ids].clone()

    def record_post_step(self) -> None:
      obs: torch.Tensor = self._env.obs_buf  # type: ignore[assignment]
      seen["post_step_obs"] = obs.clone()

  prev_obs = torch.arange(8, dtype=torch.float).reshape(4, 2)
  mock_env.obs_buf = prev_obs.clone()

  cfg = {"r": RecorderTermCfg(func=ObsCapture, params={})}
  manager = RecorderManager(cfg, mock_env)

  reset_env_ids = torch.tensor([1])
  manager.record_pre_reset(reset_env_ids)
  # obs_buf at pre_reset is unchanged from the previous step.
  assert torch.equal(seen["pre_reset_obs"], prev_obs[reset_env_ids])

  # Simulate _reset_idx + obs recompute: reset env gets new initial obs.
  mock_env.obs_buf[reset_env_ids] = torch.zeros(1, 2)
  manager.record_post_step()
  assert torch.all(seen["post_step_obs"][reset_env_ids] == 0.0), (
    "After reset, obs_buf for the reset env holds the new-episode initial obs"
  )


def test_record_post_reset_fires_after_obs_are_fresh(mock_env: Mock) -> None:
  """record_post_reset is called after obs are computed, so obs_buf already
  holds the new-episode initial observation for the reset environments.
  """
  seen: dict[str, torch.Tensor] = {}

  class ResetObsCapture(RecorderTerm):
    def record_post_reset(self, env_ids: torch.Tensor) -> None:
      obs: torch.Tensor = self._env.obs_buf  # type: ignore[assignment]
      seen["obs"] = obs[env_ids].clone()

  reset_obs = torch.ones(4, 2) * 99.0
  mock_env.obs_buf = reset_obs

  cfg = {"r": RecorderTermCfg(func=ResetObsCapture, params={})}
  manager = RecorderManager(cfg, mock_env)

  env_ids = torch.tensor([0, 3])
  manager.record_post_reset(env_ids)
  assert torch.equal(seen["obs"], reset_obs[env_ids])


def test_record_pre_reset_before_record_post_step_in_step(mock_env: Mock) -> None:
  """record_pre_reset fires before record_post_step within a single step."""
  call_log: list[str] = []

  class OrderRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
      call_log.append("pre_reset")

    def record_post_step(self) -> None:
      call_log.append("post_step")

  cfg = {"r": RecorderTermCfg(func=OrderRecorder, params={})}
  manager = RecorderManager(cfg, mock_env)

  manager.record_pre_reset(torch.tensor([0]))
  manager.record_post_step()

  assert call_log == ["pre_reset", "post_step"]


# Manager dispatch.


def test_record_post_step_called_each_step(
  manager_and_term: tuple[RecorderManager, _CountingRecorder],
) -> None:
  manager, term = manager_and_term
  for _ in range(5):
    manager.record_post_step()
  assert term.post_step_count == 5


def test_record_pre_reset_passes_env_ids(
  manager_and_term: tuple[RecorderManager, _CountingRecorder],
) -> None:
  manager, term = manager_and_term
  env_ids = torch.tensor([0, 2])
  manager.record_pre_reset(env_ids)
  assert len(term.pre_reset_calls) == 1
  assert torch.equal(term.pre_reset_calls[0], env_ids)


def test_record_post_reset_passes_env_ids(
  manager_and_term: tuple[RecorderManager, _CountingRecorder],
) -> None:
  manager, term = manager_and_term
  env_ids = torch.tensor([1, 3])
  manager.record_post_reset(env_ids)
  assert len(term.post_reset_calls) == 1
  assert torch.equal(term.post_reset_calls[0], env_ids)


def test_close_propagates_to_all_terms(mock_env: Mock) -> None:
  cfg = {
    "a": RecorderTermCfg(func=_CountingRecorder, params={}),
    "b": RecorderTermCfg(func=_CountingRecorder, params={}),
  }
  manager = RecorderManager(cfg, mock_env)
  term_a, term_b = manager._terms
  assert isinstance(term_a, _CountingRecorder)
  assert isinstance(term_b, _CountingRecorder)

  manager.close()
  assert term_a.closed
  assert term_b.closed


def test_terms_called_in_registration_order(mock_env: Mock) -> None:
  call_log: list[str] = []

  class NamedRecorder(RecorderTerm):
    def __init__(self, cfg, env):
      super().__init__(cfg, env)
      self._name: str = cfg.params["name"]

    def record_post_step(self) -> None:
      call_log.append(self._name)

  cfg = {
    "first": RecorderTermCfg(func=NamedRecorder, params={"name": "first"}),
    "second": RecorderTermCfg(func=NamedRecorder, params={"name": "second"}),
    "third": RecorderTermCfg(func=NamedRecorder, params={"name": "third"}),
  }
  manager = RecorderManager(cfg, mock_env)
  manager.record_post_step()
  assert call_log == ["first", "second", "third"]


def test_multiple_terms_have_independent_state(mock_env: Mock) -> None:
  cfg = {
    "a": RecorderTermCfg(func=_CountingRecorder, params={}),
    "b": RecorderTermCfg(func=_CountingRecorder, params={}),
  }
  manager = RecorderManager(cfg, mock_env)
  term_a, term_b = manager._terms
  assert isinstance(term_a, _CountingRecorder)
  assert isinstance(term_b, _CountingRecorder)

  manager.record_post_step()
  manager.record_post_step()
  manager.record_pre_reset(torch.tensor([0]))

  assert term_a.post_step_count == 2
  assert term_b.post_step_count == 2
  assert len(term_a.pre_reset_calls) == 1
  assert len(term_b.pre_reset_calls) == 1


def test_active_terms_matches_registration_order(mock_env: Mock) -> None:
  cfg = {
    "first": RecorderTermCfg(func=_CountingRecorder, params={}),
    "second": RecorderTermCfg(func=_CountingRecorder, params={}),
  }
  manager = RecorderManager(cfg, mock_env)
  assert manager.active_terms == ["first", "second"]


def test_none_terms_skipped(mock_env: Mock) -> None:
  cfg: dict[str, RecorderTermCfg | None] = {
    "valid": RecorderTermCfg(func=_CountingRecorder, params={}),
    "skipped": None,
  }
  manager = RecorderManager(cfg, mock_env)  # type: ignore[arg-type]
  assert manager.active_terms == ["valid"]
  assert len(manager._terms) == 1


def test_function_based_term_raises(mock_env: Mock) -> None:
  cfg = {"bad": RecorderTermCfg(func=lambda env: None, params={})}
  with pytest.raises(TypeError, match="RecorderTerm subclass"):
    RecorderManager(cfg, mock_env)


def test_cfg_params_accessible_on_term(mock_env: Mock) -> None:
  cfg = {"t": RecorderTermCfg(func=_CountingRecorder, params={"path": "/tmp/out.csv"})}
  manager = RecorderManager(cfg, mock_env)
  term = manager._terms[0]
  assert isinstance(term, _CountingRecorder)
  assert term.cfg.params["path"] == "/tmp/out.csv"


def test_deepcopy_protects_original_cfg(mock_env: Mock) -> None:
  """Creating a manager must not mutate the caller's cfg dict.

  _resolve_common_term_cfg replaces cfg.func with the instantiated object
  in-place on the internal copy. Without deepcopy the caller's cfg would
  be silently corrupted, breaking any second use.
  """
  original_cfg = {"t": RecorderTermCfg(func=_CountingRecorder, params={})}
  RecorderManager(original_cfg, mock_env)
  assert original_cfg["t"].func is _CountingRecorder


def test_null_recorder_manager_no_ops() -> None:
  manager = NullRecorderManager()
  env_ids = torch.tensor([0, 1])
  manager.record_pre_reset(env_ids)
  manager.record_post_reset(env_ids)
  manager.record_post_step()
  manager.close()
  assert manager.active_terms == []


def test_base_term_all_hooks_are_no_ops(mock_env: Mock) -> None:
  """Base RecorderTerm hooks do nothing; __call__ raises NotImplementedError."""

  class MinimalRecorder(RecorderTerm):
    pass

  term = MinimalRecorder(
    cfg=RecorderTermCfg(func=MinimalRecorder, params={}), env=mock_env
  )
  env_ids = torch.tensor([0])
  term.record_pre_reset(env_ids)
  term.record_post_reset(env_ids)
  term.record_post_step()
  term.close()

  with pytest.raises(NotImplementedError):
    term()


# Integration: real ManagerBasedRlEnv.


_SLIDING_MASS_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="mass" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1" limited="true"/>
      <geom name="mass_geom" type="sphere" size="0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_env_cfg(recorders: dict | None = None) -> ManagerBasedRlEnvCfg:
  """One-step episode (time_out fires after a single step)."""
  timestep = 0.002
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      extent=1.0,
      entities={
        "robot": EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_string(_SLIDING_MASS_XML),
          articulation=EntityArticulationInfoCfg(
            actuators=(
              BuiltinPositionActuatorCfg(
                target_names_expr=(".*",), stiffness=100.0, damping=10.0
              ),
            )
          ),
        )
      },
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={"obs": ObservationTermCfg(func=mdp.joint_pos_rel)}
      )
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
    events={},
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=timestep, iterations=1)),
    decimation=1,
    episode_length_s=timestep,  # terminates every env after exactly one step
    recorders=recorders or {},
  )


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def test_integration_terminal_action_available_at_record_pre_reset(device: str) -> None:
  """In a real env.step(), record_pre_reset fires before _reset_idx zeroes
  action_manager.action for reset envs. This verifies the hook ordering
  described in the docstring is enforced by the actual step() implementation.
  """
  seen: dict[str, torch.Tensor] = {}

  class HookOrderRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
      # Action must still be non-zero; _reset_idx has not run yet.
      seen["pre_reset_action"] = self._env.action_manager.action[env_ids].clone()

    def record_post_step(self) -> None:
      seen["post_step_action"] = self._env.action_manager.action.clone()
      seen["reset_buf"] = self._env.reset_buf.clone()

  cfg = _make_env_cfg(
    recorders={"r": RecorderTermCfg(func=HookOrderRecorder, params={})}
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  action = torch.ones(2, 1, device=device) * 0.5
  env.step(action)

  # With episode_length_s = one timestep, both envs time out on the first step.
  assert seen["reset_buf"].all(), "Both envs should have reset after one step"

  # Terminal action must have been captured intact at record_pre_reset.
  assert seen["pre_reset_action"].shape == (2, 1)
  assert torch.allclose(seen["pre_reset_action"], action), (
    "record_pre_reset must see the applied action before _reset_idx zeroes it"
  )

  # By record_post_step the action is zeroed for all reset envs.
  assert torch.all(seen["post_step_action"] == 0.0), (
    "action_manager.action is zeroed for reset envs by the time record_post_step runs"
  )

  env.close()
