"""Tests for hierarchical RL action terms (CommandAction and PreTrainedPolicyAction)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import onnx
import pytest
import torch
import tyro
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


class _FixedCommandTerm(CommandTerm):
  """Minimal 3D command term for testing CommandAction integration."""

  def __init__(self, cfg: _FixedCommandTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._command[env_ids] = 0.0

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids


@dataclass(kw_only=True)
class _FixedCommandTermCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (10.0, 10.0)

  def build(self, env: ManagerBasedRlEnv) -> _FixedCommandTerm:
    return _FixedCommandTerm(self, env)


class _CountingPolicy(torch.nn.Module):
  """Deterministic stateful policy recording calls, observations, and resets."""

  def __init__(self, action_dim: int) -> None:
    super().__init__()
    self.action_dim = action_dim
    self.call_count = 0
    self.last_obs: torch.Tensor | None = None
    self.reset_masks: list[torch.Tensor] = []

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    self.call_count += 1
    self.last_obs = obs.clone()
    # Map the first action_dim elements of obs into low-level joint actions.
    return obs[:, : self.action_dim] * 0.5

  def reset(self, dones: torch.Tensor) -> None:
    self.reset_masks.append(dones.clone())


def _make_robot_scene(num_envs: int = 3) -> SceneCfg:
  xml = load_fixture_xml("fixed_base_articulated")
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint.*",),
          stiffness=40.0,
          damping=4.0,
        ),
      ),
    ),
  )
  return SceneCfg(
    num_envs=num_envs,
    terrain=TerrainEntityCfg(),
    entities={"robot": robot_cfg},
  )


def test_command_action_standalone_and_command_manager_sync(device: str) -> None:
  """CommandAction applies scale/offset/clamp, exposes action_command, and syncs."""
  cfg = ManagerBasedRlEnvCfg(
    decimation=2,
    scene=_make_robot_scene(num_envs=2),
    commands={"twist": _FixedCommandTermCfg()},
    actions={
      "nav_cmd": mdp.CommandActionCfg(
        command_name="twist",
        scale=(2.0, 1.0, 0.5),
        offset=(0.1, -0.1, 0.0),
        low=(-1.0, -0.5, -0.25),
        high=(1.0, 0.5, 0.25),
      ),
      "aux_cmd": mdp.CommandActionCfg(
        action_dim=2,
        scale=3.0,
        low=-2.0,
        high=2.0,
      ),
    },
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "twist": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
          ),
          "aux": ObservationTermCfg(
            func=mdp.action_command, params={"action_name": "aux_cmd"}
          ),
        }
      )
    },
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    raw = torch.tensor(
      [
        [0.3, 1.0, -1.0, 0.5, -1.0],
        [-1.0, -1.0, 1.0, 2.0, 0.0],
      ],
      device=device,
    )
    obs, _, _, _, _ = env.step(raw)
    expected_twist = torch.tensor(
      [
        [0.7, 0.5, -0.25],
        [-1.0, -0.5, 0.25],
      ],
      device=device,
    )
    expected_aux = torch.tensor(
      [
        [1.5, -2.0],
        [2.0, 0.0],
      ],
      device=device,
    )
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    torch.testing.assert_close(actor_obs[:, :3], expected_twist)
    torch.testing.assert_close(actor_obs[:, 3:], expected_aux)

    # Partial reset clears only the reset environment's command buffer.
    aux_term = env.action_manager.get_term("aux_cmd")
    assert isinstance(aux_term, mdp.CommandAction)
    aux_term.reset(torch.tensor([0], device=device))
    torch.testing.assert_close(aux_term.command[0], torch.zeros(2, device=device))
    torch.testing.assert_close(aux_term.command[1], expected_aux[1])
  finally:
    env.close()


def test_pretrained_policy_action_substep_decimation_and_resets(device: str) -> None:
  """PreTrainedPolicyAction steps low-level policy at low_level_decimation rate."""
  policy = _CountingPolicy(action_dim=2)

  low_level_obs_cfg = ObservationGroupCfg(
    terms={
      # Put command first so _CountingPolicy maps command[:, :2] -> low-level action.
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
      ),
      "hier_cmd": ObservationTermCfg(func=mdp.hierarchical_command),
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, history_length=2),
      "last_low_action": ObservationTermCfg(func=mdp.last_action),
    }
  )

  cfg = ManagerBasedRlEnvCfg(
    decimation=6,
    scene=_make_robot_scene(num_envs=3),
    actions={
      "locomotion": mdp.PreTrainedPolicyActionCfg(
        action_dim=3,
        policy=policy,
        low_level_decimation=2,
        command_name="twist",
        scale=(1.0, 2.0, 0.5),
        low=-1.5,
        high=1.5,
        low_level_actions={
          "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint.*",),
            scale=0.25,
          ),
        },
        low_level_observations=low_level_obs_cfg,
      )
    },
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "high_last_action": ObservationTermCfg(func=mdp.last_action),
          "high_cmd": ObservationTermCfg(
            func=mdp.action_command, params={"action_name": "locomotion"}
          ),
        }
      )
    },
  )

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    policy.call_count = 0
    policy.reset_masks.clear()

    high_action = torch.tensor(
      [
        [0.4, 0.5, 1.0],
        [-0.8, 1.0, -2.0],
        [2.0, -1.0, 0.0],
      ],
      device=device,
    )
    obs, _, _, _, _ = env.step(high_action)

    # decimation=6, low_level_decimation=2 => 3 low-level policy evaluations per step.
    assert policy.call_count == 3
    assert policy.last_obs is not None

    expected_cmd = torch.tensor(
      [
        [0.4, 1.0, 0.5],
        [-0.8, 1.5, -1.0],
        [1.5, -1.5, 0.0],
      ],
      device=device,
    )
    # Both generated_commands("twist") and hierarchical_command() match expected_cmd.
    torch.testing.assert_close(policy.last_obs[:, :3], expected_cmd)
    torch.testing.assert_close(policy.last_obs[:, 3:6], expected_cmd)

    # Low-level last_action observed on the 3rd tick equals 0.5 * expected_cmd[:, :2]
    # from the 2nd tick, while outer last_action remains the 3D high-level action.
    torch.testing.assert_close(policy.last_obs[:, -2:], expected_cmd[:, :2] * 0.5)
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    torch.testing.assert_close(actor_obs[:, :3], high_action)
    torch.testing.assert_close(actor_obs[:, 3:6], expected_cmd)

    # Partial reset propagates env_ids mask to recurrent/stateful policy.reset(dones).
    env.reset(env_ids=torch.tensor([1], device=device))
    assert len(policy.reset_masks) == 1
    assert policy.reset_masks[0].tolist() == [False, True, False]
  finally:
    env.close()


def test_pretrained_policy_action_loads_torchscript_and_onnx(
  tmp_path: Path, device: str
) -> None:
  """PreTrainedPolicyAction loads both TorchScript (.pt) and batch-1 ONNX (.onnx)."""

  class _LinearScriptPolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
      return obs[:, :2] * 0.2

  ts_path = tmp_path / "low_level_policy.pt"
  scripted = torch.jit.script(_LinearScriptPolicy())
  scripted.save(str(ts_path))

  # Create a fixed batch-1 ONNX model (matching rsl_rl sim2real export shape [1, 4]).
  onnx_path = tmp_path / "low_level_policy.onnx"
  in_info = onnx.helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 4])
  out_info = onnx.helper.make_tensor_value_info(
    "actions", onnx.TensorProto.FLOAT, [1, 2]
  )
  starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, [1], [0])
  ends = onnx.helper.make_tensor("ends", onnx.TensorProto.INT64, [1], [2])
  axes = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], [1])
  slice_node = onnx.helper.make_node(
    "Slice", ["obs", "starts", "ends", "axes"], ["actions"]
  )
  graph = onnx.helper.make_graph(
    [slice_node],
    "low_level_graph",
    [in_info],
    [out_info],
    initializer=[starts, ends, axes],
  )
  model = onnx.helper.make_model(
    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
  )
  onnx.save(model, str(onnx_path))

  low_obs = ObservationGroupCfg(
    terms={
      "cmd": ObservationTermCfg(func=mdp.hierarchical_command),
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    }
  )

  for model_path in (ts_path, onnx_path):
    cfg = ManagerBasedRlEnvCfg(
      decimation=2,
      scene=_make_robot_scene(num_envs=3),
      actions={
        "hier": mdp.PreTrainedPolicyActionCfg(
          action_dim=2,
          policy_file=str(model_path),
          low_level_decimation=1,
          low_level_actions={
            "joint_pos": mdp.JointPositionActionCfg(
              entity_name="robot",
              actuator_names=("joint.*",),
            ),
          },
          low_level_observations=low_obs,
        )
      },
      observations={
        "actor": ObservationGroupCfg(
          terms={"joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel)}
        )
      },
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
      env.reset()
      cmd = torch.tensor([[0.5, -0.5], [1.0, 0.25], [-0.2, 0.4]], device=device)
      env.step(cmd)
      term = env.action_manager.get_term("hier")
      assert isinstance(term, mdp.PreTrainedPolicyAction)
      assert term.low_level_actions.shape == (3, 2)
    finally:
      env.close()


def test_hierarchical_action_cfgs_tyro_cli(tmp_path: Path) -> None:
  """tyro CLI parses PreTrainedPolicyActionCfg and CommandActionCfg cleanly."""
  dummy_pt = tmp_path / "dummy.pt"
  dummy_pt.write_bytes(b"placeholder")

  default_cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(terrain=TerrainEntityCfg()),
    actions={
      "cmd": mdp.CommandActionCfg(action_dim=3, scale=1.0),
      "hier": mdp.PreTrainedPolicyActionCfg(
        action_dim=3,
        policy_file=str(dummy_pt),
        low_level_decimation=2,
        low_level_actions={
          "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint.*",),
          )
        },
        low_level_observations=ObservationGroupCfg(terms={}),
      ),
    },
  )
  parsed = tyro.cli(
    ManagerBasedRlEnvCfg,
    default=default_cfg,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    args=["--actions.hier.low-level-decimation", "4"],
  )
  hier_cfg = parsed.actions["hier"]
  assert isinstance(hier_cfg, mdp.PreTrainedPolicyActionCfg)
  assert hier_cfg.low_level_decimation == 4
