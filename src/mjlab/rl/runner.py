import math
import os
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.mpopi.algorithm import MpopiPpo
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
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    mpopi_cfg = train_cfg.get("algorithm", {}).get("mpopi") or {}
    min_action_std = mpopi_cfg.get("min_action_std")
    _resolve_mpopi_mode(train_cfg)
    super().__init__(env, train_cfg, log_dir, device)
    if min_action_std is not None:
      set_min_action_std(self.alg.actor, min_action_std)
    self._attach_mpc_collector()

  def _attach_mpc_collector(self) -> None:
    """In mode ``"mpc_ppo"``, give the algorithm an MPC collector on this task."""
    alg = self.alg
    if not isinstance(alg, MpopiPpo) or alg.mpc_cfg is None:
      return
    if not isinstance(self.env, RslRlVecEnvWrapper):
      raise ValueError("mpc_ppo requires an mjlab environment.")
    # Local import: mjlab.mpc.collector imports mjlab.rl (circular).
    from mjlab.mpc.collector import MpcCollector

    cfg = alg.mpc_cfg
    collector = MpcCollector(
      self.env.unwrapped.cfg,
      num_envs=cfg.num_envs,
      num_steps=cfg.num_steps,
      planner_cfg=cfg.planner,
      execution_std=cfg.execution_std,
      clip_actions=self.env.clip_actions,
      device=self.device,
      seed=int(self.cfg.get("seed", 0)) + 1,  # Different starts from training.
      teacher_gap_every=cfg.teacher_gap_every,
    )
    alg.attach_mpc_collector(collector)

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
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

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


MPOPI_PPO_CLASS_NAME = "mjlab.rl.mpopi:MpopiPpo"


def set_min_action_std(actor, min_std: float) -> None:
  """Raise the lower bound of a Gaussian actor's std to ``min_std``."""
  dist = getattr(actor, "distribution", None)
  if not isinstance(dist, GaussianDistribution):
    raise ValueError("min_action_std requires a Gaussian actor distribution.")
  dist.std_range[0] = max(dist.std_range[0], min_std)
  dist.log_std_range[0] = math.log(dist.std_range[0])


def _resolve_mpopi_mode(train_cfg: dict) -> None:
  """Translate ``algorithm.mpopi.mode`` into the RSL-RL algorithm class.

  In ``"ppo"`` mode the ``mpopi`` key is removed so the upstream ``PPO`` class
  receives exactly the arguments it would without MPOPI. Otherwise the
  algorithm class becomes :class:`mjlab.rl.mpopi.MpopiPpo`.
  """
  alg_cfg = train_cfg.get("algorithm")
  if alg_cfg is None or "mpopi" not in alg_cfg:
    return
  mpopi_cfg = alg_cfg.pop("mpopi")
  if mpopi_cfg is None or mpopi_cfg.get("mode", "ppo") == "ppo":
    return
  if alg_cfg.get("class_name", "PPO") not in ("PPO", MPOPI_PPO_CLASS_NAME):
    raise ValueError(
      f"MPOPI mode '{mpopi_cfg['mode']}' requires the PPO algorithm, got "
      f"class_name='{alg_cfg['class_name']}'."
    )
  alg_cfg["class_name"] = MPOPI_PPO_CLASS_NAME
  alg_cfg["mpopi"] = mpopi_cfg
