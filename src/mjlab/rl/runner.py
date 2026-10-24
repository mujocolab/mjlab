import os
import time
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import warp as wp
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from mjlab.rl import training_state
from mjlab.rl.checkpoint import save_checkpoint
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self._stateful_resume = train_cfg.get("stateful_resume", False)
    if self._stateful_resume and type(self) is not MjlabOnPolicyRunner:
      raise ValueError(
        "Stateful resume supports the audited MjlabOnPolicyRunner only; "
        "custom runner subclasses are unsupported"
      )
    self._continuation_observation: TensorDict | None = None
    self._pending_rng = None
    self._next_iteration = 0
    self._at_update_boundary = False
    self._coordinated_states = None
    self._owns_process_group = False
    if self._stateful_resume:
      if train_cfg["algorithm"]["class_name"] != "PPO" or any(
        train_cfg["algorithm"].get(k) for k in ("rnd_cfg", "symmetry_cfg")
      ):
        raise ValueError(
          "Stateful resume currently supports native PPO without RND/symmetry"
        )
      if train_cfg.get("torch_compile_mode") is not None:
        raise ValueError("Stateful resume currently requires uncompiled native PPO")
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    if self._stateful_resume:
      error = None
      try:
        if type(self.alg._raw_actor) not in (MLPModel, RNNModel) or type(
          self.alg._raw_critic
        ) not in (MLPModel, RNNModel):
          raise ValueError(
            "Stateful resume supports audited native MLPModel/RNNModel only"
          )
        self._state_compatibility = training_state.preflight(env, train_cfg, device)
      except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
      try:
        self._check_rank_errors(error, "stateful resume preflight")
      except BaseException:
        self.close()
        raise

  def _configure_multi_gpu(self) -> None:
    if not self._stateful_resume or int(os.environ.get("WORLD_SIZE", "1")) == 1:
      super()._configure_multi_gpu()
      return
    self.gpu_world_size = int(os.environ["WORLD_SIZE"])
    self.gpu_global_rank = int(os.environ["RANK"])
    self.gpu_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    self.is_distributed = True
    self.cfg["multi_gpu"] = {
      "global_rank": self.gpu_global_rank,
      "local_rank": self.gpu_local_rank,
      "world_size": self.gpu_world_size,
    }
    cpu = torch.device(self.device).type == "cpu"
    if not cpu and self.device != f"cuda:{self.gpu_local_rank}":
      raise ValueError("Stateful distributed learner device must match LOCAL_RANK")
    if not 0 <= self.gpu_global_rank < self.gpu_world_size:
      raise ValueError("Invalid distributed RANK/WORLD_SIZE")
    if not cpu:
      torch.cuda.set_device(self.gpu_local_rank)
    if not torch.distributed.is_initialized():
      torch.distributed.init_process_group(
        backend="gloo" if cpu else "nccl",
        rank=self.gpu_global_rank,
        world_size=self.gpu_world_size,
        timeout=timedelta(seconds=120),
      )
      self._owns_process_group = True
    elif (torch.distributed.get_rank(), torch.distributed.get_world_size()) != (
      self.gpu_global_rank,
      self.gpu_world_size,
    ):
      raise ValueError("Existing process group differs from requested rank topology")

  def _check_rank_errors(self, error: str | None, operation: str) -> None:
    errors: list[str | None] = [error]
    if self.is_distributed:
      errors = [None] * self.gpu_world_size
      torch.distributed.all_gather_object(errors, error)
    failures = [f"rank {rank}: {value}" for rank, value in enumerate(errors) if value]
    if failures:
      raise ValueError(f"{operation} failed: {'; '.join(failures)}")

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    rng = None
    if self._stateful_resume:
      if self.is_distributed and self._coordinated_states is None:
        raise ValueError(
          "Distributed stateful save must use save_training_checkpoint on every rank"
        )
      states = self._coordinated_states or {
        self.gpu_global_rank: self._capture_rank_state()
      }
      saved_dict["training_state"] = self._training_envelope(states)
      rng = states[self.gpu_global_rank]["rng"]
    try:
      save_checkpoint(saved_dict, path)
      if self.cfg["upload_model"]:
        self.logger.save_model(path, self.current_learning_iteration)
    finally:
      if rng is not None:
        training_state.restore_rng(rng)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    if self._stateful_resume:
      if map_location is None:
        map_location = self.device
      return self._load_stateful(path, load_cfg, strict, map_location)
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def _load_stateful(self, path, load_cfg, strict, map_location) -> dict:
    checkpoint = state = None
    error = None
    try:
      if torch.device(map_location) != torch.device(self.device):
        raise ValueError("Stateful resume map_location must match the learner device")
      if load_cfg is not None or not strict or self.alg.storage.step != 0:
        raise ValueError(
          "Stateful resume requires a complete strict restore at an update boundary"
        )
      checkpoint = torch.load(path, map_location=map_location, weights_only=False)
      state = self._validate_training_state(checkpoint)
    except Exception as exc:
      error = f"{type(exc).__name__}: {exc}"
    # No rank may modify its learner/environment before every peer accepts input.
    self._check_rank_errors(error, "checkpoint read/preflight")
    assert checkpoint is not None and state is not None
    if self.is_distributed:
      identities = [None] * self.gpu_world_size
      torch.distributed.all_gather_object(identities, state["snapshot_id"])
      if len(set(identities)) != 1:
        raise ValueError("Distributed ranks selected different training snapshots")
    error = None
    try:
      learner = state["ranks"][self.gpu_global_rank]["learner"]
      self.alg.load(learner, None, True)
      self.current_learning_iteration = checkpoint["iter"]
      self.alg.transition.clear()
      self._restore_training_state(state)
    except Exception as exc:
      error = f"{type(exc).__name__}: {exc}"
    # A custom hook may fail after partial mutation: stop all peers. Rollback is
    # not claimed; callers must discard the failed runner rather than continue.
    self._check_rank_errors(error, "checkpoint restore")
    return checkpoint["infos"]

  def learn(self, num_learning_iterations: int, init_at_random_ep_len=False) -> None:
    """Close the native logging writer even if collection or learning fails."""
    completed = False
    try:
      if self._stateful_resume:
        self._learn_stateful(num_learning_iterations, init_at_random_ep_len)
      else:
        super().learn(num_learning_iterations, init_at_random_ep_len)
      completed = True
    finally:
      # TensorBoard is not a native LogWriter; stop_logging_writer does not close it.
      writer = self.logger.writer
      if writer is not None:
        close = getattr(writer, "close", None)
        if close is not None:
          close()
        elif not completed:
          self.logger.stop_logging_writer()
        self.logger.writer = None

  def _capture_rank_state(self) -> dict:
    if not self._at_update_boundary or self.alg.storage.step != 0:
      raise ValueError("Stateful snapshots require a completed PPO update boundary")
    assert self._continuation_observation is not None
    self._synchronize_training_state()
    logger = {
      name: getattr(self.logger, name)
      for name in (
        "tot_timesteps",
        "tot_time",
        "ep_extras",
        "rewbuffer",
        "lenbuffer",
        "cur_reward_sum",
        "cur_episode_length",
      )
    }
    return training_state.host_state(
      {
        "compatibility": self._state_compatibility,
        "learner": self.alg.save(),
        "simulation": training_state.capture_simulation(
          self.env.unwrapped, self._state_compatibility["simulation_constants"]
        ),
        "components": training_state.ComponentState(self.env.unwrapped).capture(),
        "observation": self._continuation_observation.cpu().clone().to_dict(),
        "recurrent": {
          name: getattr(self.alg, name).get_hidden_state()
          for name in ("actor", "critic")
        },
        "logger": logger,
        "rng": training_state.rng_state(),
      }
    )

  def _training_envelope(self, ranks: dict) -> dict:
    return {
      "version": training_state.FORMAT_VERSION,
      "snapshot_id": uuid.uuid4().hex,
      "world_size": self.gpu_world_size,
      "next_iteration": self._next_iteration,
      "ranks": ranks,
    }

  def save_training_checkpoint(self, path: str) -> None:
    """Capture on every rank, then publish once; never collect inside save().

    All ranks must call at the same update boundary. Capture failures are exchanged
    before transferring potentially large payloads; publication errors are broadcast
    so a nonzero rank does not continue into the next gradient collective alone.
    Process-group operations have a bounded timeout for a lost worker.
    """
    state = None
    error = None
    try:
      state = self._capture_rank_state()
    except Exception as exc:
      error = f"{type(exc).__name__}: {exc}"
    self._check_rank_errors(error, "training snapshot capture")
    assert state is not None
    gathered: list[dict] | None = (
      [{} for _ in range(self.gpu_world_size)] if self.gpu_global_rank == 0 else None
    )
    if self.is_distributed:
      torch.distributed.gather_object(state, gathered, dst=0)
    else:
      gathered = [state]
    outcome: list[str | None] = [None]
    try:
      if self.gpu_global_rank == 0:
        assert gathered is not None
        self._coordinated_states = dict(enumerate(gathered))
        try:
          self.save(path)
        except Exception as exc:
          outcome[0] = f"{type(exc).__name__}: {exc}"
        finally:
          self._coordinated_states = None
      if self.is_distributed:
        torch.distributed.broadcast_object_list(outcome, src=0)
      if outcome[0] is not None:
        raise OSError(f"Training checkpoint publication failed: {outcome[0]}")
    finally:
      training_state.restore_rng(state["rng"])

  def _validate_training_state(self, checkpoint: dict) -> dict:
    state = checkpoint.get("training_state")
    if (
      not isinstance(state, dict)
      or state.get("version") != training_state.FORMAT_VERSION
    ):
      raise ValueError(
        "Checkpoint lacks a supported full training state; use ordinary resume for legacy checkpoints"
      )
    rank = state.get("ranks", {}).get(self.gpu_global_rank)
    if not isinstance(state.get("snapshot_id"), str) or not state["snapshot_id"]:
      raise ValueError("Invalid stateful checkpoint identity")
    if (
      not isinstance(rank, dict)
      or rank.get("compatibility") != self._state_compatibility
    ):
      raise ValueError(
        "Stateful resume task, learner, device topology or software differs"
      )
    if (
      state.get("next_iteration") != checkpoint["iter"] + 1
      or set(state.get("ranks", {})) != set(range(self.gpu_world_size))
      or state.get("world_size") != self.gpu_world_size
    ):
      raise ValueError("Invalid stateful checkpoint iteration/rank metadata")
    training_state.validate_simulation(self.env.unwrapped, rank["simulation"])
    training_state.ComponentState(self.env.unwrapped).restore(
      rank["components"], validate=True
    )
    learner = rank["learner"]
    for name in ("actor", "critic"):
      expected = getattr(self.alg, f"_raw_{name}").state_dict()
      saved = learner[f"{name}_state_dict"]
      if saved.keys() != expected.keys() or any(
        saved[key].shape != value.shape or saved[key].dtype != value.dtype
        for key, value in expected.items()
      ):
        raise ValueError(f"Stateful resume {name} state layout differs")
    if "optimizer_state_dict" not in learner:
      raise ValueError("Stateful resume requires the complete optimizer state")
    return state

  def _restore_training_state(self, state: dict) -> None:
    rank = state["ranks"][self.gpu_global_rank]
    self._synchronize_training_state()
    with torch.inference_mode():
      training_state.restore_simulation(self.env.unwrapped, rank["simulation"])
      self._synchronize_training_state()
      training_state.ComponentState(self.env.unwrapped).restore(rank["components"])
      self._continuation_observation = TensorDict(
        rank["observation"], batch_size=[self.env.num_envs], device=self.device
      )
      for name, hidden in rank["recurrent"].items():
        getattr(self.alg, name).reset(hidden_state=hidden)
      for name, value in rank["logger"].items():
        setattr(self.logger, name, value)
    self._synchronize_training_state()
    self._next_iteration = state["next_iteration"]
    self._at_update_boundary = True
    self._pending_rng = rank["rng"]
    training_state.restore_rng(rank["rng"])

  def _synchronize_training_state(self) -> None:
    """Wait for every stream before crossing Torch/Warp snapshot boundaries."""
    for device in {torch.device(self.device), torch.device(self.env.device)}:
      if device.type == "cuda":
        torch.cuda.synchronize(device)
    wp.synchronize_device(self.env.unwrapped.sim.wp_device)

  def _learn_stateful(
    self, num_learning_iterations: int, init_at_random_ep_len: bool
  ) -> None:
    """RSL 5.5.1 loop with an explicit continuation observation/update boundary.

    Keep native collection/update/logging order. The opt-in adapter is necessary:
    native learn() reinitializes episode lengths and repeats its saved iteration.
    """
    if num_learning_iterations < 0:
      raise ValueError("num_learning_iterations must be nonnegative")
    if self._continuation_observation is None:
      if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
          self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )
      obs = self.env.get_observations().to(self.device)
    else:
      obs = self._continuation_observation
    self.alg.train_mode()
    if self.is_distributed and not self._at_update_boundary:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()
    if self._pending_rng is not None:
      training_state.restore_rng(self._pending_rng)
      self._pending_rng = None
    start_it = self._next_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      self._at_update_boundary = False
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras)
        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)
      loss_dict = self.alg.update()
      learn_time = time.time() - start
      self.current_learning_iteration = it
      self._next_iteration = it + 1
      self._continuation_observation = obs
      self._at_update_boundary = True
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )
      if self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
        assert self.logger.log_dir is not None
        self.save_training_checkpoint(
          os.path.join(self.logger.log_dir, f"model_{it}.pt")
        )
    if self.logger.log_dir is not None:
      assert self.logger.log_dir is not None
      if (
        self._at_update_boundary
        and self.current_learning_iteration % self.cfg["save_interval"] != 0
      ):
        self.save_training_checkpoint(
          os.path.join(
            self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
          )
        )
      self.logger.stop_logging_writer()

  def close(self) -> None:
    """Release a process group created by this runner after all training is done."""
    if self._owns_process_group and torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()
      self._owns_process_group = False
