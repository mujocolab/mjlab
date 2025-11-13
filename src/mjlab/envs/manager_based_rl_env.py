import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from mjlab.envs import types
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionManager
from mjlab.managers.command_manager import CommandManager, NullCommandManager
from mjlab.managers.curriculum_manager import CurriculumManager, NullCurriculumManager
from mjlab.managers.event_manager import EventManager
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.observation_manager import ObservationManager
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.spaces import Box
from mjlab.spaces import Dict as DictSpace
from mjlab.utils import random as random_utils
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
  """Configuration for a manager-based RL environment.

  Combines base environment settings with RL-specific configuration
  (rewards, terminations, commands, curriculum).
  """

  # Base environment configuration.
  decimation: int
  scene: SceneCfg
  observations: dict[str, ObservationGroupCfg]
  actions: dict[str, ActionTermCfg]
  events: dict[str, EventTermCfg] = field(
    default_factory=lambda: {
      "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
      )
    }
  )
  seed: int | None = None
  sim: SimulationCfg = field(default_factory=SimulationCfg)
  viewer: ViewerConfig = field(default_factory=ViewerConfig)

  # RL-specific configuration.
  episode_length_s: float = 0.0
  rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
  commands: dict[str, CommandTermCfg] | None = None
  curriculum: dict[str, CurriculumTermCfg] | None = None
  is_finite_horizon: bool = False


class ManagerBasedRlEnv:
  """Manager-based environment with RL support.

  Combines base environment management (scene, simulation, managers) with
  RL-specific features (rewards, terminations, commands, curriculum).
  """

  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: ManagerBasedRlEnvCfg

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    """Initialize the environment.

    Args:
        cfg: Environment configuration
        device: Device for computation (e.g., "cuda:0")
        render_mode: Rendering mode ("rgb_array" or None)
        **kwargs: Additional arguments (unused, for compatibility)
    """
    # Initialize base environment state.
    self.cfg = cfg
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}

    # Initialize scene and simulation.
    self.scene = Scene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      model=self.scene.compile(),
      device=device,
    )

    if "cuda" in self.device:
      torch.cuda.set_device(self.device)

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    # Print environment info.
    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    # Initialize RL-specific state.
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore

    # Load all managers.
    self.load_managers()
    self.setup_manager_visualizers()

  # Properties.

  @property
  def num_envs(self) -> int:
    """Number of parallel environments."""
    return self.scene.num_envs

  @property
  def physics_dt(self) -> float:
    """Physics simulation step size."""
    return self.cfg.sim.mujoco.timestep

  @property
  def step_dt(self) -> float:
    """Environment step size (physics_dt * decimation)."""
    return self.cfg.sim.mujoco.timestep * self.cfg.decimation

  @property
  def device(self) -> str:
    """Device for computation."""
    return self.sim.device

  @property
  def max_episode_length_s(self) -> float:
    """Maximum episode length in seconds."""
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> int:
    """Maximum episode length in steps."""
    return math.ceil(self.max_episode_length_s / self.step_dt)

  @property
  def unwrapped(self) -> "ManagerBasedRlEnv":
    """Get the unwrapped environment (base case for wrapper chains)."""
    return self

  # Methods.

  def setup_manager_visualizers(self) -> None:
    self.manager_visualizers = {}
    if getattr(self.command_manager, "active_terms", None):
      self.manager_visualizers["command_manager"] = self.command_manager

  def load_managers(self) -> None:
    """Load and initialize all managers.

    Order is important! Event and command managers must be loaded first,
    then action and observation managers, then other RL managers.
    """
    # Event manager (required before everything else for domain randomization).
    self.event_manager = EventManager(self.cfg.events, self)
    print_info(f"[INFO] {self.event_manager}")

    self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

    # Command manager (must be before observation manager since observations may reference commands).
    if self.cfg.commands is not None:
      self.command_manager = CommandManager(self.cfg.commands, self)
    else:
      self.command_manager = NullCommandManager()
    print_info(f"[INFO] {self.command_manager}")

    # Action and observation managers.
    self.action_manager = ActionManager(self.cfg.actions, self)
    print_info(f"[INFO] {self.action_manager}")
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    print_info(f"[INFO] {self.observation_manager}")

    # Other RL-specific managers.

    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    print_info(f"[INFO] {self.termination_manager}")
    self.reward_manager = RewardManager(self.cfg.rewards, self)
    print_info(f"[INFO] {self.reward_manager}")
    if self.cfg.curriculum is not None:
      self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
    else:
      self.curriculum_manager = NullCurriculumManager()
    print_info(f"[INFO] {self.curriculum_manager}")

    # Configure spaces for the environment.
    self._configure_gym_env_spaces()

    # Initialize startup events if defined.
    if "startup" in self.event_manager.available_modes:
      self.event_manager.apply(mode="startup")
      self.sim.create_graph()

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    """Reset the environment.

    Args:
        seed: Seed for random number generation
        env_ids: Environment indices to reset (default: all)
        options: Additional reset options (unused)

    Returns:
        Observation and extras dict
    """
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return self.obs_buf, self.extras

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Step the environment.

    Args:
        action: Action tensor to apply

    Returns:
        Tuple of (observations, rewards, terminated, truncated, extras)
    """
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.obs_buf = self.observation_manager.compute(update_history=True)

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def render(self) -> np.ndarray | None:
    if self.render_mode == "human" or self.render_mode is None:
      return None
    elif self.render_mode == "rgb_array":
      if self._offline_renderer is None:
        raise ValueError("Offline renderer not initialized")
      debug_callback = (
        self.update_visualizers if hasattr(self, "update_visualizers") else None
      )
      self._offline_renderer.update(self.sim.data, debug_vis_callback=debug_callback)
      return self._offline_renderer.render()
    else:
      raise NotImplementedError(
        f"Render mode {self.render_mode} is not supported. "
        f"Please use: {self.metadata['render_modes']}."
      )

  def close(self) -> None:
    """Close the environment and free resources."""
    if self._offline_renderer is not None:
      self._offline_renderer.close()

  @staticmethod
  def seed(seed: int = -1) -> int:
    """Set the random seed.

    Args:
        seed: Seed value (-1 = random)

    Returns:
        The actual seed used
    """
    if seed == -1:
      seed = np.random.randint(0, 10_000)
    print_info(f"Setting seed: {seed}")
    random_utils.seed_rng(seed)
    return seed

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    """Update debug visualizers.

    Args:
        visualizer: Debug visualizer instance
    """
    for mod in self.manager_visualizers.values():
      mod.debug_vis(visualizer)

  # Private methods.

  def _configure_gym_env_spaces(self) -> None:
    """Configure observation and action spaces.

    Uses mjlab.spaces instead of gymnasium.spaces.
    """
    from mjlab.spaces import batch_space

    self.single_observation_space = DictSpace()
    for group_name, group_term_names in self.observation_manager.active_terms.items():
      has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
      group_dim = self.observation_manager.group_obs_dim[group_name]
      if has_concatenated_obs:
        assert isinstance(group_dim, tuple)
        self.single_observation_space.spaces[group_name] = Box(
          shape=group_dim, low=-math.inf, high=math.inf
        )
      else:
        assert not isinstance(group_dim, tuple)
        group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
        # Create a nested dict for this group.
        group_space = DictSpace()
        for term_name, term_dim, _term_cfg in zip(
          group_term_names, group_dim, group_term_cfgs, strict=False
        ):
          group_space.spaces[term_name] = Box(
            shape=term_dim, low=-math.inf, high=math.inf
          )
        self.single_observation_space.spaces[group_name] = group_space

    action_dim = sum(self.action_manager.action_term_dim)
    self.single_action_space = Box(shape=(action_dim,), low=-math.inf, high=math.inf)

    self.observation_space = batch_space(self.single_observation_space, self.num_envs)
    self.action_space = batch_space(self.single_action_space, self.num_envs)

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.curriculum_manager.compute(env_ids=env_ids)
    # Reset the internal buffers of the scene elements.
    self.scene.reset(env_ids)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # NOTE: This is order sensitive.
    self.extras["log"] = dict()
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
