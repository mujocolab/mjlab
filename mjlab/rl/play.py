from dataclasses import asdict
from pathlib import Path
import time
from typing import Annotated, Optional

import jax
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab import TaskConfigUnion
from mjlab._src import registry, mjx_task
from mjlab.rl import config, utils, wrapper, exporter

import mujoco
from mujoco import mjx
import mujoco.viewer

_HERE = Path(__file__).parent

TaskConfig = Annotated[
  TaskConfigUnion,
  tyro.conf.subcommand(),
]


def main(
  task_cfg: TaskConfig,
  agent_cfg: config.OnPolicyRunnerConfig,
  video: bool = False,
  checkpoint: Optional[Path] = None,
  wandb_run_path: Optional[Path] = None,
  num_envs: int = 1,
  camera: Optional[str] = None,
):
  """Play a checkpoint from an agent trained with RSL-RL."""

  log_root_path = _HERE / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  print(f"[INFO]: Loading experiment from: {log_root_path}")
  if checkpoint is not None:
    resume_path = checkpoint.resolve()
  elif wandb_run_path is not None:
    resume_path = utils.get_wandb_checkpoint_path(log_root_path, wandb_run_path)
  else:
    resume_path = utils.get_checkpoint_path(
      log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
  print(f"[INFO]: Loading checkpoint: {resume_path}")

  # log_dir = resume_path.parent

  task_name = registry.get_task_name_by_config_class_name(task_cfg.__class__.__name__)
  env = registry.make(task_name, task_cfg)

  if video:
    # TODO(kevin): Implement this.
    print("[info]: Recording video during eval.")
    pass

  env = wrapper.RslRlVecEnvWrapper(
    env, num_envs=num_envs, seed=0, device=agent_cfg.device
  )

  print(f"[INFO]: Loading model checkpoint from: {resume_path}")
  ppo_runner = OnPolicyRunner(
    env, asdict(agent_cfg), log_dir=None, device=agent_cfg.device
  )
  ppo_runner.load(resume_path, map_location=agent_cfg.device)

  policy = ppo_runner.get_inference_policy(device=env.device)
  policy_nn = ppo_runner.alg.policy

  export_model_dir = resume_path.parent / "exported"
  exporter.export_policy_as_onnx(
    policy=policy_nn,
    normalizer=ppo_runner.obs_normalizer,
    path=export_model_dir,
    filename="policy.onnx",
  )

  task: mjx_task.MjxTask = env.unwrapped.task
  m = task.model  # C Mujoco model.
  d = mujoco.MjData(m)  # C Mujoco data.

  obs, _ = env.get_observations()

  viewer = mujoco.viewer.launch_passive(m, d)
  with viewer:
    while viewer.is_running():
      start = time.time()
      with torch.inference_mode():
        actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      viewer.user_scn.ngeom = 0
      dx = jax.tree.map(lambda x: x[0, 0], env.state.data)
      mjx.get_data_into(d, m, dx)
      task.visualize(jax.tree.map(lambda x: x[0, 0], env.state), viewer.user_scn)
      viewer.sync()
      elapsed = time.time() - start
      if elapsed < task.dt:
        time.sleep(task.dt - elapsed)

  # states = []
  # obs, _ = env.get_observations()
  # timestep = 0
  # while timestep < env.unwrapped.task.cfg.max_episode_length:
  #   with torch.inference_mode():
  #     actions = policy(obs)
  #   obs, _, _, _ = env.step(actions)
  #   states.append(env.state)
  #   timestep += 1

  # def grab_first(x):
  #   return x[0, 0]

  # states = [jax.tree.map(grab_first, state) for state in states]

  # frames = env.unwrapped.render(states, camera=camera, height=480, width=640)
  # import mediapy as mp

  # mp.write_video(f"{task_name}.mp4", frames, fps=(1.0 / env.unwrapped.task.dt))


if __name__ == "__main__":
  tyro.cli(main, config=(tyro.conf.ConsolidateSubcommandArgs,))
