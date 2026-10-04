"""Actuator events must preserve controls outside the selected physical subset."""

from types import SimpleNamespace
from typing import Any

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  BuiltinMotorActuatorCfg,
  BuiltinPdActuator,
  BuiltinPdActuatorCfg,
  BuiltinPositionActuatorCfg,
  DcMotorActuatorCfg,
  DcMotorDatasheetParams,
  DcMotorInputMode,
  IdealPdActuator,
  IdealPdActuatorCfg,
  XmlActuatorCfg,
)
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.dr import actuator as actuator_dr
from mjlab.envs.mdp.dr._types import Distribution
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

FIELDS = (
  "actuator_gainprm",
  "actuator_biasprm",
  "actuator_forcerange",
  "jnt_actfrcrange",
  "tendon_actfrcrange",
)
NUM_ENVS = 4
SELECTED_ENVS = [1, 3]


def _xml(xml_position=False, tendon=False):
  bodies = "".join(
    f'<body name="link{i}" pos="{i * 0.3} 0 0">'
    f'<joint name="joint{i}" type="hinge" range="-2 2" armature="0.1"/>'
    '<geom type="sphere" size="0.05" mass="1"/></body>'
    for i in range(1, 6)
  )
  actuators = ""
  if xml_position:
    actuators = (
      "<actuator>"
      + "".join(
        f'<position name="joint{i}" joint="joint{i}" '
        f'kp="{10 if i <= 3 else 20}" kv="{2 if i <= 3 else 4}" '
        f'forcerange="{-50 if i <= 3 else -70} {50 if i <= 3 else 70}"/>'
        for i in range(1, 6)
      )
      + "</actuator>"
    )
  tendons = ""
  if tendon:
    tendons = (
      "<tendon>"
      + "".join(
        f'<fixed name="tendon{i}"><joint joint="joint{i}" coef="1"/></fixed>'
        for i in range(1, 6)
      )
      + "</tendon>"
    )
  return (
    '<mujoco><worldbody><body name="base">'
    '<geom type="sphere" size="0.05" mass="1"/>'
    f"{bodies}</body></worldbody>{tendons}{actuators}</mujoco>"
  )


def _cfg(kind, targets, kp, kd, limit, transmission=TransmissionType.JOINT):
  kwargs: dict[str, Any] = dict(
    target_names_expr=targets, transmission_type=transmission
  )
  if kind == "xml":
    return XmlActuatorCfg(**kwargs, command_field="position")
  if kind == "motor":
    return BuiltinMotorActuatorCfg(**kwargs, effort_limit=limit)
  if kind in ("dc_position", "dc_velocity"):
    return BuiltinDcMotorActuatorCfg(
      **kwargs,
      mode=(
        DcMotorInputMode.POSITION
        if kind == "dc_position"
        else DcMotorInputMode.VELOCITY
      ),
      motor_params=DcMotorDatasheetParams(
        nominal_voltage=24.0, stall_torque=2.0, no_load_speed=100.0
      ),
      stiffness=kp,
      damping=kd,
      voltage_limit=24.0,
      effort_limit=limit,
    )
  classes = {
    "builtin": BuiltinPositionActuatorCfg,
    "ideal": IdealPdActuatorCfg,
    "dc": DcMotorActuatorCfg,
    "pd": BuiltinPdActuatorCfg,
  }
  kwargs.update(stiffness=kp, damping=kd, effort_limit=limit)
  if kind == "dc":
    kwargs.update(saturation_effort=100.0, velocity_limit=30.0)
  return classes[kind](**kwargs)


def _env(device, kind, *, sorted_pd=False, shared=False, tendon=False):
  transmission = TransmissionType.TENDON if tendon else TransmissionType.JOINT
  prefix = "tendon" if tendon else "joint"
  second_targets = (f"{prefix}[245]",) if shared else (f"{prefix}[45]",)
  groups = (
    _cfg(
      "builtin" if kind == "mixed" else kind,
      (f"{prefix}[123]",),
      10.0,
      2.0,
      50.0,
      transmission,
    ),
    _cfg(
      "motor" if kind == "mixed" else kind,
      second_targets,
      20.0,
      4.0,
      70.0,
      transmission,
    ),
  )
  xml = _xml(xml_position=kind == "xml", tendon=True)
  other_xml = _xml(tendon=True)

  scene = Scene(
    SceneCfg(
      num_envs=NUM_ENVS,
      entities={
        "other": EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_string(other_xml),
          articulation=EntityArticulationInfoCfg(
            actuators=(_cfg("builtin", ("joint.*",), 30.0, 6.0, 90.0),)
          ),
        ),
        "robot": EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_string(xml),
          sort_actuators=sorted_pd,
          articulation=EntityArticulationInfoCfg(actuators=groups),
        ),
      },
    ),
    device,
  )
  model = scene.compile()
  sim = Simulation(NUM_ENVS, SimulationCfg(), model, device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(FIELDS)
  assert scene["robot"].indexing.ctrl_ids.min().item() > 0
  return SimpleNamespace(scene=scene, sim=sim, device=device, num_envs=NUM_ENVS)


def _snapshot(env):
  result: dict[str | tuple[str, int, str], torch.Tensor] = {
    field: getattr(env.sim.model, field).clone() for field in FIELDS
  }
  for entity_name, entity in env.scene.entities.items():
    for i, act in enumerate(entity.actuators):
      if isinstance(act, IdealPdActuator):
        for name in act.param_names:
          result[(entity_name, i, name)] = getattr(act, name).clone()
  return result


def _restore(env, snapshot):
  for key, value in snapshot.items():
    if isinstance(key, str):
      getattr(env.sim.model, key).copy_(value)
    else:
      entity, index, name = key
      getattr(env.scene[entity].actuators[index], name).copy_(value)


def _assert_snapshot(env, expected):
  actual = _snapshot(env)
  assert actual.keys() == expected.keys()
  for key, value in expected.items():
    assert torch.equal(actual[key], value), key


def _selection(env, selector):
  names = env.scene["robot"].actuator_names
  selections = {
    "names": SceneEntityCfg(
      "robot", actuator_names=[names[4], names[1]], preserve_order=True
    ),
    "list": SceneEntityCfg("robot", actuator_ids=[1]),
    "slice": SceneEntityCfg("robot", actuator_ids=slice(1, 5, 2)),
    "reordered": SceneEntityCfg("robot", actuator_ids=[4, 1]),
    "empty": SceneEntityCfg("robot", actuator_ids=[]),
    "full": SceneEntityCfg("robot"),
  }
  cfg = selections[selector]
  cfg.resolve(env.scene)
  ids = (
    list(range(len(names)))[cfg.actuator_ids]
    if isinstance(cfg.actuator_ids, slice)
    else cfg.actuator_ids
  )
  return cfg, ids


def _expected_single_controls(env, before, local_ids, operation):
  expected = {key: value.clone() for key, value in before.items()}
  selected = set(local_ids)
  worlds = torch.tensor(SELECTED_ENVS, device=env.device)
  for group_index, act in enumerate(env.scene["robot"].actuators):
    for column, local_id in enumerate(act.ctrl_ids.tolist()):
      if local_id not in selected:
        continue
      global_id = act.global_ctrl_ids[column]
      if isinstance(act, IdealPdActuator):
        for name, factor, absolute in (
          ("stiffness", 2.0, 100.0),
          ("damping", 3.0, 12.0),
          ("force_limit", 2.0, 15.0),
        ):
          key = ("robot", group_index, name)
          expected[key][worlds, column] = (
            before[key][worlds, column] * factor if operation == "scale" else absolute
          )
      else:
        for field, slot, factor, absolute in (
          ("actuator_gainprm", 0, 2.0, 100.0),
          ("actuator_biasprm", 1, 2.0, -100.0),
          ("actuator_biasprm", 2, 3.0, -12.0),
        ):
          expected[field][worlds, global_id, slot] = (
            before[field][worlds, global_id, slot] * factor
            if operation == "scale"
            else absolute
          )
        expected["actuator_forcerange"][worlds, global_id] = (
          before["actuator_forcerange"][worlds, global_id] * 2.0
          if operation == "scale"
          else torch.tensor([-15.0, 15.0], device=env.device)
        )
  return expected


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module", params=["builtin", "ideal"])
def family_env(request, device):
  env = _env(device, request.param)
  return request.param, env, _snapshot(env)


@pytest.fixture
def restored_env(family_env):
  kind, env, baseline = family_env
  _restore(env, baseline)
  return kind, env


@pytest.mark.parametrize("operation", ["scale", "abs"])
@pytest.mark.parametrize(
  "selector", ["names", "list", "slice", "reordered", "empty", "full"]
)
def test_events_preserve_unselected_controls(restored_env, operation, selector):
  _, env = restored_env
  _exercise_single_controls(env, operation, selector)


def _exercise_single_controls(env, operation, selector, repeat=1):
  cfg, local_ids = _selection(env, selector)
  before = _snapshot(env)
  expected = _expected_single_controls(env, before, local_ids, operation)
  ids = torch.tensor(SELECTED_ENVS, device=env.device)
  for _ in range(repeat):
    dr.pd_gains(
      env,
      ids,
      kp_range=(2.0, 2.0) if operation == "scale" else (100.0, 100.0),
      kd_range=(3.0, 3.0) if operation == "scale" else (12.0, 12.0),
      operation=operation,
      asset_cfg=cfg,
    )
    dr.effort_limits(
      env,
      ids,
      effort_limit_range=(2.0, 2.0) if operation == "scale" else (15.0, 15.0),
      operation=operation,
      asset_cfg=cfg,
    )
  _assert_snapshot(env, expected)


@pytest.mark.parametrize("kind", ["xml", "dc"])
@pytest.mark.parametrize("operation", ["scale", "abs"])
def test_xml_and_dc_subsets_preserve_other_controls(device, kind, operation):
  _exercise_single_controls(_env(device, kind), operation, "reordered", repeat=2)


def test_repeated_subset_scale_uses_defaults(restored_env):
  _, env = restored_env
  cfg, local_ids = _selection(env, "reordered")
  before = _snapshot(env)
  expected = _expected_single_controls(env, before, local_ids, "scale")
  for _ in range(3):
    ids = torch.tensor(SELECTED_ENVS, device=env.device)
    dr.pd_gains(env, ids, (2.0, 2.0), (3.0, 3.0), asset_cfg=cfg, operation=dr.scale)
    dr.effort_limits(env, ids, (2.0, 2.0), asset_cfg=cfg, operation=dr.scale)
  _assert_snapshot(env, expected)


@pytest.mark.parametrize("kind", ["ideal", "dc"])
def test_custom_subset_changes_fused_control_output(device, kind):
  env = _env(device, kind)
  cfg, local_ids = _selection(env, "reordered")
  robot = env.scene["robot"]
  ids = torch.tensor(SELECTED_ENVS, device=env.device)
  dr.pd_gains(env, ids, (100.0, 100.0), (0.0, 0.0), asset_cfg=cfg, operation="abs")
  dr.effort_limits(env, ids, (15.0, 15.0), asset_cfg=cfg, operation="abs")
  zeros = torch.zeros(NUM_ENVS, 5, device=env.device)
  robot.write_joint_state_to_sim(zeros, zeros)
  robot.set_joint_position_target(torch.full_like(zeros, 2.0))
  robot.set_joint_velocity_target(zeros)
  robot.set_joint_effort_target(zeros)
  robot.write_data_to_sim()
  expected = torch.tensor([20.0] * 3 + [40.0] * 2, device=env.device).repeat(
    NUM_ENVS, 1
  )
  expected[ids[:, None], torch.tensor(local_ids, device=env.device)] = 15.0
  assert torch.equal(env.sim.data.ctrl[:, robot.indexing.ctrl_ids], expected)
  env.sim.step()
  assert torch.isfinite(env.sim.data.qpos).all()
  assert torch.isfinite(env.sim.data.qvel).all()


@pytest.mark.parametrize("operation", ["scale", "abs"])
@pytest.mark.parametrize("roles", ["position", "velocity", "both"])
def test_sorted_builtin_pd_changes_only_selected_roles(device, operation, roles):
  env = _env(device, "pd", sorted_pd=True)
  robot = env.scene["robot"]
  names = [
    f"joint2_pd_{suffix}"
    for suffix in (
      ("pos",)
      if roles == "position"
      else ("vel",)
      if roles == "velocity"
      else ("vel", "pos")
    )
  ]
  cfg = SceneEntityCfg("robot", actuator_names=names, preserve_order=True)
  cfg.resolve(env.scene)
  before = _snapshot(env)
  expected = {key: value.clone() for key, value in before.items()}
  ids = torch.tensor(SELECTED_ENVS, device=device)
  for act in robot.actuators:
    assert isinstance(act, BuiltinPdActuator)
    for column, local_id in enumerate(act.ctrl_ids.tolist()):
      name = robot.actuator_names[local_id]
      target_index = act.target_names.index(
        name.removesuffix("_pd_pos").removesuffix("_pd_vel")
      )
      assert act.ctrl_target_ids[column].item() == target_index
      assert act.position_mask[column].item() == name.endswith("_pd_pos")
  for name in names:
    local_id = robot.actuator_names.index(name)
    global_id = robot.indexing.ctrl_ids[local_id]
    position = name.endswith("_pd_pos")
    factor, absolute, bias_slot = (2.0, 100.0, 1) if position else (3.0, 12.0, 2)
    expected["actuator_gainprm"][ids, global_id, 0] = (
      before["actuator_gainprm"][ids, global_id, 0] * factor
      if operation == "scale"
      else absolute
    )
    expected["actuator_biasprm"][ids, global_id, bias_slot] = (
      before["actuator_biasprm"][ids, global_id, bias_slot] * factor
      if operation == "scale"
      else -absolute
    )
  dr.pd_gains(
    env,
    ids,
    (2.0, 2.0) if operation == "scale" else (100.0, 100.0),
    (3.0, 3.0) if operation == "scale" else (12.0, 12.0),
    asset_cfg=cfg,
    operation=operation,
  )
  _assert_snapshot(env, expected)


@pytest.mark.parametrize("tendon", [False, True])
@pytest.mark.parametrize("operation", ["scale", "abs"])
def test_shared_pd_clamp_samples_each_physical_target_once(
  device, monkeypatch, tendon, operation
):
  original_edit_spec = BuiltinPdActuator.edit_spec
  group_index = 0

  def edit_with_unique_names(act, spec, names):
    nonlocal group_index
    original_edit_spec(act, spec, names)
    # Only names change. The controls still act on the same physical target.
    for element in act._mjs_actuators:
      element.name = f"{element.name}_group{group_index}"
    group_index += 1

  monkeypatch.setattr(BuiltinPdActuator, "edit_spec", edit_with_unique_names)
  env = _env(device, "pd", shared=True, tendon=tendon)
  calls = []

  def sample(lo, hi, shape, device):
    calls.append(shape)
    return torch.full(shape, 2.0, device=device)

  monkeypatch.setattr(
    actuator_dr, "resolve_distribution", lambda _: Distribution("counted", sample)
  )
  before = _snapshot(env)
  expected = {key: value.clone() for key, value in before.items()}
  robot = env.scene["robot"]
  field = "tendon_actfrcrange" if tendon else "jnt_actfrcrange"
  global_ids = robot.indexing.tendon_ids if tendon else robot.indexing.joint_ids
  ids = torch.tensor(SELECTED_ENVS, device=device)
  expected[field][ids[:, None], global_ids] = (
    before[field][ids[:, None], global_ids] * 2.0
    if operation == "scale"
    else torch.tensor([-2.0, 2.0], device=device)
  )
  dr.effort_limits(
    env,
    ids,
    (1.0, 3.0),
    operation=operation,
    distribution="uniform",
    asset_cfg=SceneEntityCfg("robot"),
  )
  assert sum(shape[0] * shape[1] for shape in calls) == 2 * 5
  _assert_snapshot(env, expected)


def test_empty_selection_does_not_sample_or_visit_unsupported_groups(
  device, monkeypatch
):
  env = _env(device, "mixed")

  def unexpected_sample(*args):
    raise AssertionError("An empty actuator selection must not sample.")

  monkeypatch.setattr(
    actuator_dr,
    "resolve_distribution",
    lambda _: Distribution("unexpected", unexpected_sample),
  )
  before = _snapshot(env)
  cfg = SceneEntityCfg("robot", actuator_ids=[])
  ids = torch.tensor(SELECTED_ENVS, device=device)
  dr.pd_gains(env, ids, (1.0, 2.0), (1.0, 2.0), asset_cfg=cfg)
  dr.effort_limits(env, ids, (1.0, 2.0), asset_cfg=cfg)
  _assert_snapshot(env, before)


def test_supported_selection_skips_excluded_unsupported_group(device):
  _exercise_single_controls(_env(device, "mixed"), "scale", "list")


@pytest.mark.parametrize("kind", ["dc_position", "dc_velocity"])
@pytest.mark.parametrize("operation", ["scale", "abs"])
def test_native_dc_subset_changes_only_pid_slots(device, kind, operation):
  env = _env(device, kind)
  cfg, local_ids = _selection(env, "reordered")
  before = _snapshot(env)
  expected = {key: value.clone() for key, value in before.items()}
  ids = torch.tensor(SELECTED_ENVS, device=device)
  global_ids = env.scene["robot"].indexing.ctrl_ids[local_ids]
  for slot, factor, absolute in ((4, 2.0, 8.0), (6, 3.0, 3.0)):
    expected["actuator_gainprm"][ids[:, None], global_ids, slot] = (
      before["actuator_gainprm"][ids[:, None], global_ids, slot] * factor
      if operation == "scale"
      else absolute
    )
  dr.pd_gains(
    env,
    ids,
    (2.0, 2.0) if operation == "scale" else (8.0, 8.0),
    (3.0, 3.0),
    asset_cfg=cfg,
    operation=operation,
  )
  _assert_snapshot(env, expected)
