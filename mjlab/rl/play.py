from dataclasses import asdict
from pathlib import Path
import time
from typing import Annotated, List, Optional

import jax
import jax.numpy as jp
import torch
import tyro
from mjlab.rl.on_policy_runner import MjlabOnPolicyRunner as OnPolicyRunner

from mjlab.envs import TaskConfigUnion
from mjlab.core.types import State
from mjlab.core import mjx_task
from mjlab.rl import config, utils, wrapper, exporter

import mujoco
from mujoco import mjx
import mujoco.viewer

from mjlab.envs import registry

_HERE = Path(__file__).parent

TaskConfig = Annotated[
  TaskConfigUnion,
  tyro.conf.subcommand(),
]


def main(
  task_cfg: TaskConfig,
  agent_cfg: config.OnPolicyRunnerConfig,
  video: bool = False,
  video_length: int = 200,
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

  task_name = registry.get_task_name_by_config_class_name(task_cfg.__class__.__name__)
  env = registry.make(task_name, task_cfg)

  tic = time.time()
  env = wrapper.RslRlVecEnvWrapper(
    env,
    num_envs=num_envs,
    seed=0,
    device=agent_cfg.device,
    resample_on_reset=True,
  )
  toc = time.time()
  print(f"[INFO]: Time to wrap env: {toc - tic} seconds")

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
  exporter.attach_onnx_metadata(env.unwrapped, wandb_run_path, path=export_model_dir)
  print(f"[INFO] Saved policy to: {export_model_dir / 'policy.onnx'}")

  task: mjx_task.MjxTask = env.unwrapped.task
  m = task.model
  d = mujoco.MjData(m)

  tic = time.time()
  obs, _ = env.get_observations()
  toc = time.time()
  print(f"[INFO]: Time to get observations: {toc - tic} seconds")

  tic = time.time()
  with torch.inference_mode():
    actions = policy(obs)
  toc = time.time()
  print(f"[INFO]: Time to get actions: {toc - tic} seconds")

  tic = time.time()
  obs, _, _, _ = env.step(actions)
  toc = time.time()
  print(f"[INFO]: Time to step: {toc - tic} seconds")

  def unbatch(state_or_data):
    return jax.tree.map(lambda x: x.squeeze(0), state_or_data)

  states: List[State] = []

  viewer = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
  with viewer:
    if camera is not None:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = m.camera(camera).id
    else:
      mujoco.mjv_defaultFreeCamera(m, viewer.cam)
    while viewer.is_running():
      start = time.time()
      env.state = env.state.tree_replace(
        {"data.xfrc_applied": jp.array(d.xfrc_applied)[None]}
      )
      with torch.inference_mode():
        actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      viewer.user_scn.ngeom = 0
      unbatched_state = unbatch(env.state)
      unbatched_data = unbatch(env.state.data)
      with viewer.lock():
        mjx.get_data_into(d, m, unbatched_data)
        task.visualize(unbatched_state, viewer.user_scn)
        mujoco.mj_forward(m, d)
      viewer.sync()
      if len(states) < video_length:
        states.append(unbatched_state)
      elapsed = time.time() - start
      if elapsed < task.dt:
        time.sleep(task.dt - elapsed)

  if video:
    import mediapy as mp

    frames = env.unwrapped.render(states, camera=camera)
    video_path = resume_path.parent / "video"
    if not video_path.exists():
      video_path.mkdir(parents=True)
    video_name = f"{task_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    mp.write_video(str(video_path / video_name), frames, fps=(1.0 / task.dt))
    print(f"[INFO] Saved video to: {video_path / video_name}")


if __name__ == "__main__":
  tyro.cli(main, config=(tyro.conf.ConsolidateSubcommandArgs,))
