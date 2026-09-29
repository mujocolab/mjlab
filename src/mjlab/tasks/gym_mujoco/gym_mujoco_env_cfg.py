"""Common scene, simulation, and action wiring for Gymnasium MuJoCo tasks."""

from functools import partial

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.gym_mujoco.assets import get_entity_cfg, scene_options


def make_base_env_cfg(
  xml_name: str,
  num_envs: int,
  *,
  sim: MujocoCfg,
  decimation: int,
  max_episode_steps: int,
  init_state: EntityCfg.InitialStateCfg,
) -> ManagerBasedRlEnvCfg:
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=0.0,
      entities={"robot": get_entity_cfg(xml_name, init_state)},
      spec_fn=partial(scene_options, xml_name=xml_name),
    ),
    sim=SimulationCfg(mujoco=sim),
    decimation=decimation,
    episode_length_s=max_episode_steps * decimation * sim.timestep,
    scale_rewards_by_dt=False,
    actions={
      "joint_effort": JointEffortActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=1.0,
      )
    },
    events={
      "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset")
    },
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)
    },
  )
