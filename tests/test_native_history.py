"""Tests for native MuJoCo 4.0+ actuator and sensor history/delay."""

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  BuiltinMotorActuatorCfg,
  BuiltinPdActuatorCfg,
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
  DcMotorDatasheetParams,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

ROBOT_XML = load_fixture_xml("floating_base_articulated")


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def test_native_history_validation() -> None:
  with pytest.raises(ValueError, match="nsample must be > 0"):
    BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",),
      stiffness=50.0,
      damping=5.0,
      delay=0.01,
      nsample=0,
    )

  with pytest.raises(ValueError, match="delay must be non-negative"):
    BuiltinMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=10.0,
      delay=-0.01,
      nsample=4,
    )

  with pytest.raises(ValueError, match="nsample must be non-negative"):
    BuiltinVelocityActuatorCfg(
      target_names_expr=("joint.*",),
      damping=5.0,
      nsample=-1,
    )

  with pytest.raises(ValueError, match="interp must be one of"):
    BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",),
      stiffness=50.0,
      damping=5.0,
      delay=0.01,
      nsample=4,
      interp="quadratic",  # ty: ignore[invalid-argument-type]
    )

  with pytest.raises(ValueError, match="nsample must be > 0"):
    BuiltinSensorCfg(
      name="clock_delayed",
      sensor_type="clock",
      delay=0.01,
      nsample=0,
    )

  with pytest.raises(ValueError, match="nsample must be > 0"):
    BuiltinSensorCfg(
      name="clock_sampled",
      sensor_type="clock",
      interval=(0.02, 0.0),
      nsample=0,
    )

  with pytest.raises(ValueError, match="interval period must be non-negative"):
    BuiltinSensorCfg(
      name="clock_bad_interval",
      sensor_type="clock",
      interval=(-0.01, 0.0),
      nsample=2,
    )


def test_builtin_actuator_spec_history_compilation() -> None:
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPdActuatorCfg(
          target_names_expr=("joint1",),
          stiffness=60.0,
          damping=6.0,
          delay=0.015,
          nsample=5,
          interp="linear",
        ),
        BuiltinDcMotorActuatorCfg(
          target_names_expr=("joint2",),
          motor_params=DcMotorDatasheetParams(
            nominal_voltage=24.0,
            stall_torque=20.0,
            no_load_speed=30.0,
          ),
          stiffness=40.0,
          damping=4.0,
          voltage_limit=24.0,
          delay=0.01,
          nsample=4,
          interp="cubic",
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  assert model.nhistory > 0
  # BuiltinPdActuator creates 2 elements (pos + vel) for joint1; DcMotor creates 1.
  assert model.nu == 3
  assert list(model.actuator_history[:, 0]) == [5, 5, 4]
  assert list(model.actuator_history[:, 1]) == [1, 1, 2]
  assert torch.allclose(
    torch.tensor(model.actuator_delay, dtype=torch.float32),
    torch.tensor([0.015, 0.015, 0.01]),
  )


def test_native_actuator_delay_and_interpolation(device: str) -> None:
  dt = 0.01
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("joint1",),
          effort_limit=100.0,
          delay=0.015,  # 1.5 physics steps: exercises sub-step interpolation.
          nsample=5,
          interp="linear",
        ),
        BuiltinMotorActuatorCfg(
          target_names_expr=("joint2",),
          effort_limit=100.0,
          delay=0.015,
          nsample=5,
          interp="zoh",
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(
    num_envs=1,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=dt)),
    model=model,
    device=device,
  )
  entity.initialize(model, sim.model, sim.data, device)

  # Step 0 (t=0.00): command 10.0, inserted into history at t=0.00.
  # Step 1 (t=0.01): command 20.0, inserted into history at t=0.01.
  for val in (10.0, 20.0):
    entity.set_joint_effort_target(torch.full((1, 2), val, device=device))
    entity.write_data_to_sim()
    sim.step()

  # Now sim.data.time == 0.02. Query time t - delay = 0.02 - 0.015 = 0.005,
  # halfway between t=0.00 (val=10) and t=0.01 (val=20).
  sim.forward()
  act_force = sim.data.actuator_force[0]
  # Linear interpolation gives 15.0; ZOH holds the t=0.00 sample (10.0).
  torch.testing.assert_close(act_force[0], torch.tensor(15.0, device=device))
  torch.testing.assert_close(act_force[1], torch.tensor(10.0, device=device))


def test_native_position_actuator_reset_backfills_post_reset_pose(
  device: str,
) -> None:
  dt = 0.01
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint.*",),
          stiffness=80.0,
          damping=0.0,
          effort_limit=200.0,
          delay=0.02,
          nsample=4,
          interp="zoh",
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(
    num_envs=2,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=dt)),
    model=model,
    device=device,
  )
  entity.initialize(model, sim.model, sim.data, device)

  # Advance both envs with non-zero commands so env 1 builds live history.
  for step in range(3):
    entity.set_joint_position_target(
      torch.full((2, 2), 0.1 * (step + 1), device=device)
    )
    entity.write_data_to_sim()
    sim.step()

  env1_history_before = sim.data.history[1].clone()

  # Reset only env 0 to pose [0.6, -0.4] and initialize controls.
  env_ids = torch.tensor([0], device=device)
  sim.reset(env_ids)
  entity.reset(env_ids)
  pose = torch.tensor([[0.6, -0.4]], device=device)
  entity.write_joint_state_to_sim(pose, torch.zeros_like(pose), env_ids=env_ids)
  entity.write_data_to_sim(env_ids=env_ids)

  # Env 1's native history buffer must remain untouched.
  assert torch.equal(sim.data.history[1], env1_history_before)

  # Env 0's delayed position actuator must hold pose [0.6, -0.4] (zero PD error ->
  # zero actuator force), NOT pull toward 0.0!
  sim.forward()
  torch.testing.assert_close(
    sim.data.actuator_force[0],
    torch.zeros(2, device=device),
    atol=1e-5,
    rtol=1e-5,
  )


def test_native_sensor_delay_interval_and_reset_history(device: str) -> None:
  dt = 0.01
  robot_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(
    num_envs=2,
    entities={"robot": robot_cfg},
    sensors=(
      BuiltinSensorCfg(
        name="jpos_delayed",
        sensor_type="jointpos",
        obj=ObjRef(type="joint", name="joint1", entity="robot"),
        delay=0.02,
        nsample=4,
        interp="zoh",
      ),
      BuiltinSensorCfg(
        name="clock_sampled",
        sensor_type="clock",
        interval=(0.03, 0.0),
        nsample=4,
        interp="zoh",
      ),
    ),
  )
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(
    num_envs=2,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=dt)),
    model=model,
    device=device,
  )
  scene.initialize(model, sim.model, sim.data)

  robot = scene["robot"]
  jpos_sensor = scene["robot/jpos_delayed"]
  clock_sensor = scene["clock_sampled"]

  # Step 0 (t=0.00): set qpos=0.5 on joint1 before stepping.
  robot.write_joint_state_to_sim(
    torch.tensor([[0.5, 0.0], [0.5, 0.0]], device=device),
    torch.zeros((2, 2), device=device),
  )
  sim.step()  # Inserts t=0.00 sample (0.5), advances time to 0.01.

  # Step 1 (t=0.01): change qpos to 0.9; with 0.02s delay, t-delay = -0.01 -> still 0.0.
  robot.write_joint_state_to_sim(
    torch.tensor([[0.9, 0.0], [0.9, 0.0]], device=device),
    torch.zeros((2, 2), device=device),
  )
  sim.step()  # Inserts t=0.01 sample (0.9), advances time to 0.02.

  # At t=0.02, forward() reads t - 0.02 = 0.00 -> 0.5!
  sim.forward()
  torch.testing.assert_close(
    jpos_sensor.data[:, 0], torch.tensor([0.5, 0.5], device=device)
  )
  # Clock sensor has interval=0.03s starting at t=0.0, so at t=0.02 it still holds 0.0.
  torch.testing.assert_close(
    clock_sensor.data[:, 0], torch.tensor([0.0, 0.0], device=device)
  )

  # Advance one more step to t=0.03: clock interval triggers at 0.03!
  sim.step()
  sim.forward()
  torch.testing.assert_close(
    clock_sensor.data[:, 0], torch.tensor([0.03, 0.03], device=device)
  )

  # Selective sim.reset_history on env 0 clears env 0's history without touching env 1.
  env1_hist = sim.data.history[1].clone()
  sim.reset_history(torch.tensor([0], device=device))
  assert torch.equal(sim.data.history[1], env1_hist)
  assert not torch.equal(sim.data.history[0], env1_hist)


def test_per_world_native_delay_randomization(device: str) -> None:
  dt = 0.01
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("joint1",),
          effort_limit=100.0,
          delay=0.01,
          nsample=4,
          interp="zoh",
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(
    num_envs=2,
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=dt)),
    model=model,
    device=device,
  )
  entity.initialize(model, sim.model, sim.data, device)

  # Expand actuator_delay per world: env 0 has 0.0s delay, env 1 has 0.01s delay.
  sim.expand_model_fields(("actuator_delay",))
  sim.model.actuator_delay[0, 0] = 0.0
  sim.model.actuator_delay[1, 0] = 0.01

  # Step 0 at t=0.00 with effort=10.0 -> inserted at t=0.00.
  entity.set_joint_effort_target(
    torch.tensor([[10.0, 0.0], [10.0, 0.0]], device=device)
  )
  entity.write_data_to_sim()
  sim.step()

  # At t=0.01, set ctrl=30.0 and call forward():
  # env 0 (delay=0.0) sees latest history/ctrl? Wait: when nsample > 0 and delay == 0,
  # _read_ctrl_delayed reads d.ctrl directly when delay == 0! Let's verify:
  # env 0 sees 30.0, env 1 (delay=0.01) reads t=0.00 -> 10.0!
  entity.set_joint_effort_target(
    torch.tensor([[30.0, 0.0], [30.0, 0.0]], device=device)
  )
  entity.write_data_to_sim()
  sim.forward()
  torch.testing.assert_close(
    sim.data.actuator_force[:, 0],
    torch.tensor([30.0, 10.0], device=device),
  )
