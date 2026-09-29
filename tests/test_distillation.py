"""Tests for teacher-student distillation and RL fine-tuning."""

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import pytest
import torch
from conftest import get_test_device
from tensordict import TensorDict

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import (
  MjlabDistillationRunner,
  MjlabOnPolicyRunner,
  MultiTeacherModel,
  RslRlCriticWarmupPpoAlgorithmCfg,
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlModelCfg,
  RslRlMultiTeacherModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlVecEnvWrapper,
)
from mjlab.rl.utils import clean_model_cfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

_ROBOT_XML = """
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


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_env(device, observations: dict[str, ObservationGroupCfg]):
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(_ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )
  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _joint_pos_term():
  return ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_pos)


def _joint_vel_term():
  return ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_vel)


@pytest.fixture(scope="module")
def distill_env(device):
  """Env with student/teacher groups plus an expert-assignment group."""
  env = _make_env(
    device,
    observations={
      "student": ObservationGroupCfg(terms={"joint_pos": _joint_pos_term()}),
      "teacher": ObservationGroupCfg(
        terms={"joint_pos": _joint_pos_term(), "joint_vel": _joint_vel_term()}
      ),
      "teacher_assignment": ObservationGroupCfg(
        terms={
          "expert_id": ObservationTermCfg(
            func=lambda env: (
              torch.arange(env.num_envs, device=env.device) % 2
            ).unsqueeze(-1)
          )
        },
      ),
    },
  )
  yield env
  env.close()


def _distillation_runner_cfg(**overrides) -> RslRlDistillationRunnerCfg:
  defaults: dict[str, Any] = dict(
    student=RslRlModelCfg(
      hidden_dims=(16, 16),
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    teacher=RslRlModelCfg(hidden_dims=(16, 16)),
    algorithm=RslRlDistillationAlgorithmCfg(gradient_length=4),
    num_steps_per_env=4,
    max_iterations=2,
    save_interval=100,
    logger="tensorboard",
  )
  defaults.update(overrides)
  return RslRlDistillationRunnerCfg(**defaults)


def test_clean_model_cfg_strips_unset_options():
  cfg = {
    "class_name": "MLPModel",
    "hidden_dims": (16,),
    "cnn_cfg": None,
    "distribution_cfg": None,
    "rnn_type": None,
    "rnn_hidden_dim": 256,
    "rnn_num_layers": 1,
  }
  cleaned = clean_model_cfg(cfg)
  assert cleaned == {"class_name": "MLPModel", "hidden_dims": (16,)}
  # Set options are preserved.
  cfg["rnn_type"] = "lstm"
  assert clean_model_cfg(cfg)["rnn_hidden_dim"] == 256


def test_distillation_runner_learns_from_teacher(distill_env, device):
  """Full DAgger loop: teacher loads from a PPO-style checkpoint and the
  student regresses onto its labels."""
  wrapped_env = RslRlVecEnvWrapper(distill_env)

  with tempfile.TemporaryDirectory() as tmpdir:
    # Create a synthetic "PPO" teacher checkpoint with matching architecture.
    bootstrap = MjlabDistillationRunner(
      wrapped_env, asdict(_distillation_runner_cfg()), log_dir=None, device=device
    )
    assert not bootstrap.alg.teacher_loaded
    teacher_path = str(Path(tmpdir) / "teacher.pt")
    torch.save(
      {
        "actor_state_dict": bootstrap.alg._raw_teacher.state_dict(),
        "iter": 7,
        "infos": {"env_state": {"common_step_counter": 42}},
      },
      teacher_path,
    )

    cfg = _distillation_runner_cfg(teacher_checkpoints=(teacher_path,))
    runner = MjlabDistillationRunner(
      wrapped_env, asdict(cfg), log_dir=tmpdir, device=device
    )
    assert runner.alg.teacher_loaded
    assert wrapped_env.unwrapped.common_step_counter == 42

    student = runner.alg.get_policy()
    params_before = [p.clone() for p in student.mlp.parameters()]
    runner.learn(num_learning_iterations=2)
    params_after = list(student.mlp.parameters())
    assert any(
      not torch.allclose(b, a) for b, a in zip(params_before, params_after, strict=True)
    ), "Student parameters did not change during distillation."

    # The distillation checkpoint round-trips.
    ckpt_path = str(Path(tmpdir) / "distilled.pt")
    runner.save(ckpt_path)
    loaded = torch.load(ckpt_path, weights_only=False)
    assert "student_state_dict" in loaded
    assert "teacher_state_dict" in loaded


def test_multi_teacher_dispatch(distill_env, device):
  """Each env's labels come from the expert selected by the assignment obs."""
  wrapped_env = RslRlVecEnvWrapper(distill_env)
  teacher_cfg = RslRlModelCfg(hidden_dims=(16, 16))
  cfg = _distillation_runner_cfg(
    teacher=RslRlMultiTeacherModelCfg(
      teachers=(teacher_cfg, teacher_cfg),
      assignment_group="teacher_assignment",
    ),
  )
  runner = MjlabDistillationRunner(
    wrapped_env, asdict(cfg), log_dir=None, device=device
  )
  teacher = runner.alg._raw_teacher
  assert isinstance(teacher, MultiTeacherModel)

  obs = wrapped_env.get_observations().to(device)
  with torch.no_grad():
    combined = teacher(obs)
    expected_0 = teacher.teachers[0](obs)
    expected_1 = teacher.teachers[1](obs)
  # Env 0 is assigned expert 0, env 1 expert 1 (see distill_env fixture).
  torch.testing.assert_close(combined[0], expected_0[0])
  torch.testing.assert_close(combined[1], expected_1[1])

  # Loading requires one checkpoint per expert.
  with tempfile.TemporaryDirectory() as tmpdir:
    paths = []
    for i in range(2):
      path = str(Path(tmpdir) / f"expert_{i}.pt")
      torch.save(
        {"actor_state_dict": teacher.teachers[i].state_dict(), "infos": None}, path
      )
      paths.append(path)
    with pytest.raises(ValueError, match="one checkpoint per teacher"):
      runner.load_teacher_checkpoints((paths[0],))
    runner.load_teacher_checkpoints(tuple(paths))
    assert runner.alg.teacher_loaded
    runner.learn(num_learning_iterations=1)


@pytest.fixture(scope="module")
def finetune_env(device):
  """Env with the student group as policy obs and a privileged critic group."""
  env = _make_env(
    device,
    observations={
      "student": ObservationGroupCfg(terms={"joint_pos": _joint_pos_term()}),
      "critic": ObservationGroupCfg(
        terms={"joint_pos": _joint_pos_term(), "joint_vel": _joint_vel_term()}
      ),
    },
  )
  yield env
  env.close()


def test_finetune_from_distilled_student(finetune_env, device):
  """init_checkpoint maps student weights into the PPO actor, the configured
  init_std overrides the student's, and the critic warmup freezes the actor."""
  wrapped_env = RslRlVecEnvWrapper(finetune_env)

  with tempfile.TemporaryDirectory() as tmpdir:
    # Fabricate a distillation checkpoint for a student with matching arch.
    student = TensorDict({"student": torch.zeros(2, 1, device=device)}, batch_size=[2])
    from rsl_rl.models import MLPModel

    donor = MLPModel(
      obs=student,
      obs_groups={"student": ["student"]},
      obs_set="student",
      output_dim=wrapped_env.num_actions,
      hidden_dims=[16, 16],
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ).to(device)
    distill_ckpt = str(Path(tmpdir) / "distilled.pt")
    torch.save({"student_state_dict": donor.state_dict(), "infos": None}, distill_ckpt)

    cfg = RslRlOnPolicyRunnerCfg(
      actor=RslRlModelCfg(
        hidden_dims=(16, 16),
        distribution_cfg={
          "class_name": "GaussianDistribution",
          "init_std": 0.2,
          "std_type": "scalar",
        },
      ),
      critic=RslRlModelCfg(hidden_dims=(16, 16)),
      algorithm=RslRlCriticWarmupPpoAlgorithmCfg(critic_warmup_updates=1),
      obs_groups={"actor": ("student",), "critic": ("critic",)},
      init_checkpoint=distill_ckpt,
      num_steps_per_env=4,
      max_iterations=2,
      save_interval=100,
      logger="tensorboard",
    )
    runner = MjlabOnPolicyRunner(wrapped_env, asdict(cfg), log_dir=None, device=device)

    actor = runner.alg.get_policy()
    for (name, p), p_donor in zip(
      actor.mlp.named_parameters(), donor.mlp.parameters(), strict=True
    ):
      torch.testing.assert_close(p, p_donor, msg=f"actor mlp param {name}")
    # The distilled std (0.5) must not override the configured init_std.
    assert actor.distribution is not None
    std_param = actor.distribution.state_dict()["std_param"]
    torch.testing.assert_close(std_param, torch.full_like(std_param, 0.2))

    # First update: actor frozen (critic warmup), critic trains.
    actor_before = [p.clone() for p in actor.mlp.parameters()]
    critic = runner.alg._raw_critic
    critic_before = [p.clone() for p in critic.mlp.parameters()]
    runner.learn(num_learning_iterations=1)
    assert all(
      torch.equal(b, a)
      for b, a in zip(actor_before, actor.mlp.parameters(), strict=True)
    ), "Actor changed during critic warmup."
    assert any(
      not torch.allclose(b, a)
      for b, a in zip(critic_before, critic.mlp.parameters(), strict=True)
    ), "Critic did not train during warmup."

    # Second update: warmup over, actor trains.
    runner.learn(num_learning_iterations=1)
    assert any(
      not torch.allclose(b, a)
      for b, a in zip(actor_before, actor.mlp.parameters(), strict=True)
    ), "Actor did not train after critic warmup."


def test_init_checkpoint_rejects_mismatched_architecture(finetune_env, device):
  wrapped_env = RslRlVecEnvWrapper(finetune_env)
  with tempfile.TemporaryDirectory() as tmpdir:
    bad_ckpt = str(Path(tmpdir) / "bad.pt")
    torch.save(
      {"student_state_dict": {"mlp.layers.0.weight": torch.zeros(3, 3)}}, bad_ckpt
    )
    cfg = RslRlOnPolicyRunnerCfg(
      actor=RslRlModelCfg(hidden_dims=(16, 16)),
      critic=RslRlModelCfg(hidden_dims=(16, 16)),
      obs_groups={"actor": ("student",), "critic": ("critic",)},
      init_checkpoint=bad_ckpt,
      num_steps_per_env=4,
      logger="tensorboard",
    )
    with pytest.raises((RuntimeError, KeyError)):
      MjlabOnPolicyRunner(wrapped_env, asdict(cfg), log_dir=None, device=device)


def test_g1_distillation_task_configs():
  """The registered G1 distillation tasks have consistent obs layouts."""
  import mjlab.tasks  # noqa: F401  (populates the registry)
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  cfg = load_env_cfg("Mjlab-Velocity-Rough-Unitree-G1-Distill")
  assert set(cfg.observations) == {"student", "teacher"}
  assert "height_scan" not in cfg.observations["student"].terms
  assert "height_scan" in cfg.observations["teacher"].terms
  assert not cfg.observations["teacher"].enable_corruption
  assert cfg.observations["student"].enable_corruption

  ft_cfg = load_env_cfg("Mjlab-Velocity-Rough-Unitree-G1-Distill-Finetune")
  assert set(ft_cfg.observations) == {"student", "critic"}
  assert (
    ft_cfg.observations["student"].terms.keys()
    == cfg.observations["student"].terms.keys()
  )

  rl_cfg = load_rl_cfg("Mjlab-Velocity-Rough-Unitree-G1-Distill")
  assert isinstance(rl_cfg, RslRlDistillationRunnerCfg)
  # Teacher must mirror the rough task's actor so its checkpoints load.
  from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

  assert rl_cfg.teacher == unitree_g1_ppo_runner_cfg().actor

  ft_rl_cfg = load_rl_cfg("Mjlab-Velocity-Rough-Unitree-G1-Distill-Finetune")
  assert isinstance(ft_rl_cfg, RslRlOnPolicyRunnerCfg)
  assert ft_rl_cfg.obs_groups == {"actor": ("student",), "critic": ("critic",)}
  # Fine-tuning starts from a reduced action std (arXiv:2505.11164).
  assert isinstance(rl_cfg.student.distribution_cfg, dict)
  assert isinstance(ft_rl_cfg.actor.distribution_cfg, dict)
  assert (
    ft_rl_cfg.actor.distribution_cfg["init_std"]
    < rl_cfg.student.distribution_cfg["init_std"]
  )
