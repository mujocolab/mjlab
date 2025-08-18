"""Enhanced Native MuJoCo viewer implementation with dm_control-style timing."""

from typing import Any, Callable, Optional
import torch
import mujoco
import mujoco.viewer

from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel


class NativeMujocoViewer(BaseViewer):
  """Native MuJoCo viewer implementation using mujoco.viewer with enhanced timing."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    render_all_envs: bool = True,
    env_idx: int = 0,
    key_callback: Optional[Callable[[int], None]] = None,
    enable_perturbations: bool = True,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ):
    """Initialize the native MuJoCo viewer.

    Args:
      env: The environment to visualize.
      policy: The policy to use for action generation.
      frame_rate: Target frame rate for visualization.
      render_all_envs: Whether to render all environments or just one.
      env_idx: Index of the primary environment to render.
      key_callback: Optional callback for keyboard input.
      enable_perturbations: Whether to enable interactive perturbations.
    """
    super().__init__(env, policy, frame_rate, render_all_envs, env_idx, verbosity)

    self.key_callback = key_callback
    self.enable_perturbations = enable_perturbations

    self.mjm: Optional[mujoco.MjModel] = None
    self.mjd: Optional[mujoco.MjData] = None
    self.viewer: Optional[mujoco.viewer.Handle] = None
    self.vd: Optional[mujoco.MjData] = None
    self.pert: Optional[mujoco.MjvPerturb] = None
    self.vopt: Optional[mujoco.MjvOption] = None
    self.catmask: Optional[int] = None

    self._setup_default_callbacks()

  def _setup_default_callbacks(self) -> None:
    """Setup default key callbacks with enhanced controls."""
    from mjlab.viewer.keys import (
      KEY_ENTER,
      KEY_SPACE,
      KEY_COMMA,
      KEY_PERIOD,
      KEY_MINUS,
      KEY_EQUAL,
    )

    self.KEY_ENTER = KEY_ENTER
    self.KEY_SPACE = KEY_SPACE
    self.KEY_COMMA = KEY_COMMA
    self.KEY_PERIOD = KEY_PERIOD
    self.KEY_MINUS = KEY_MINUS
    self.KEY_EQUAL = KEY_EQUAL

    # Create wrapper callback that includes default handlers.
    original_callback = self.key_callback

    def combined_callback(key: int) -> None:
      if key == self.KEY_ENTER:
        self.log("RESET: Environment reset triggered")
        self.reset_environment()

      elif key == self.KEY_SPACE:
        self.toggle_pause()

      elif key == self.KEY_MINUS:
        # new_multiplier = self._time_multiplier * 0.5
        # self.set_time_multiplier(new_multiplier)
        self.decrease_speed()

      elif key == self.KEY_EQUAL:
        # new_multiplier = self._time_multiplier * 2.0
        # self.set_time_multiplier(new_multiplier)
        self.increase_speed()

      elif key == self.KEY_COMMA:
        # Previous environment (if multiple envs).
        if self.env.unwrapped.num_envs > 1:
          self.env_idx = (self.env_idx - 1) % self.env.unwrapped.num_envs
          self.log(f"[INFO] Switched to environment {self.env_idx}")

      elif key == self.KEY_PERIOD:
        # Next environment (if multiple envs).
        if self.env.unwrapped.num_envs > 1:
          self.env_idx = (self.env_idx + 1) % self.env.unwrapped.num_envs
          self.log(f"[INFO] Switched to environment {self.env_idx}")

      # User-provided callback.
      if original_callback:
        original_callback(key)

    self.key_callback = combined_callback

  def setup(self) -> None:
    """Setup the MuJoCo viewer resources."""
    self._is_running = True

    # Get MuJoCo model and data from environment.
    sim = self.env.unwrapped.sim
    self.mjm = sim.mj_model
    self.mjd = sim.mj_data

    # Create visualization data for rendering multiple environments.
    if self.render_all_envs and self.env.unwrapped.num_envs > 1:
      self.vd = mujoco.MjData(self.mjm)

    # Setup visualization options.
    self.pert = mujoco.MjvPerturb() if self.enable_perturbations else None
    self.vopt = mujoco.MjvOption()
    self.catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value

    self.viewer = mujoco.viewer.launch_passive(
      self.mjm, self.mjd, key_callback=self.key_callback
    )
    if self.viewer is None:
      raise RuntimeError("Failed to launch MuJoCo viewer")

    if self.enable_perturbations:
      self.log("[INFO] Interactive perturbations enabled")

    self._print_controls()

  def _print_controls(self) -> None:
    """Print control instructions to console."""
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Key", "Action"]
    table.align["Key"] = "l"
    table.align["Action"] = "l"

    table.add_row(["Space", "Pause/Resume simulation"])
    table.add_row(["R", "Reset environment"])
    table.add_row(["- (minus)", "Slow down (0.5x speed)"])
    table.add_row(["+ (plus)", "Speed up (2x speed)"])

    if self.env.unwrapped.num_envs > 1:
      table.add_row([",", "Previous environment"])
      table.add_row([".", "Next environment"])

    self.log("\n" + "VIEWER CONTROLS".center(50), VerbosityLevel.INFO)
    self.log(str(table), VerbosityLevel.INFO)  # Convert table to string

  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    sim_data = self.env.unwrapped.sim.data

    # Copy primary environment state to viewer.
    self.mjd.qpos[:] = sim_data.qpos[self.env_idx].cpu().numpy()
    self.mjd.qvel[:] = sim_data.qvel[self.env_idx].cpu().numpy()

    # Forward dynamics to update derived quantities.
    mujoco.mj_forward(self.mjm, self.mjd)

  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (perturbations)."""
    if not self.enable_perturbations or self._is_paused:
      return

    # Copy perturbation forces from viewer to environment.
    xfrc_applied = torch.tensor(
      self.mjd.xfrc_applied, dtype=torch.float, device=self.env.device
    )
    self.env.unwrapped.sim.data.xfrc_applied[:] = xfrc_applied[None]

  def update_visualizations(self) -> None:
    """Update additional visualizations."""
    user_scn = self.viewer.user_scn
    if user_scn is None:
      return

    user_scn.ngeom = 0

    if self._is_paused:
      self._add_pause_indicator(user_scn)

    # Update environment-specific visualizers.
    if hasattr(self.env.unwrapped, "update_visualizers"):
      self.env.unwrapped.update_visualizers(user_scn)

    # Render additional environments if requested.
    if self.render_all_envs and self.vd is not None:
      sim_data = self.env.unwrapped.sim.data

      for i in range(self.env.unwrapped.num_envs):
        if i == self.env_idx:
          continue  # Skip primary environment (already rendered).
        self.vd.qpos[:] = sim_data.qpos[i].cpu().numpy()
        self.vd.qvel[:] = sim_data.qvel[i].cpu().numpy()
        mujoco.mj_forward(self.mjm, self.vd)
        mujoco.mjv_addGeoms(
          self.mjm, self.vd, self.vopt, self.pert, self.catmask, user_scn
        )

  def _add_pause_indicator(self, user_scn: Any) -> None:
    """Add visual pause indicator to the scene."""
    # TODO
    pass

  def render_frame(self) -> None:
    """Render a single frame."""
    if self.viewer and self.viewer.is_running():
      self.viewer.sync(state_only=True)

  def is_running(self) -> bool:
    """Check if the viewer is still running."""
    return self.viewer is not None and self.viewer.is_running()

  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self.viewer:
      self.viewer.close()
      self.viewer = None
    self._is_running = False
    self.log("[INFO] MuJoCo viewer closed")


class NativeMujocoViewerBuilder:
  """Builder class for convenient viewer configuration."""

  def __init__(self, env: EnvProtocol, policy: PolicyProtocol):
    """Initialize the builder.

    Args:
      env: The environment to visualize.
      policy: The policy to use for action generation.
    """
    self.env = env
    self.policy = policy
    self.frame_rate = 60.0
    self.render_all_envs = True
    self.env_idx = 0
    self.key_callback = None
    self.enable_perturbations = True

  def with_frame_rate(self, frame_rate: float) -> "NativeMujocoViewerBuilder":
    """Set the frame rate."""
    self.frame_rate = frame_rate
    return self

  def with_single_env(self, env_idx: int = 0) -> "NativeMujocoViewerBuilder":
    """Configure to render only a single environment."""
    self.render_all_envs = False
    self.env_idx = env_idx
    return self

  def with_all_envs(self) -> "NativeMujocoViewerBuilder":
    """Configure to render all environments."""
    self.render_all_envs = True
    return self

  def with_key_callback(
    self, callback: Callable[[int], None]
  ) -> "NativeMujocoViewerBuilder":
    """Add a key callback."""
    self.key_callback = callback
    return self

  def without_perturbations(self) -> "NativeMujocoViewerBuilder":
    """Disable interactive perturbations."""
    self.enable_perturbations = False
    return self

  def build(self) -> NativeMujocoViewer:
    """Build the viewer with configured settings."""
    return NativeMujocoViewer(
      env=self.env,
      policy=self.policy,
      frame_rate=self.frame_rate,
      render_all_envs=self.render_all_envs,
      env_idx=self.env_idx,
      key_callback=self.key_callback,
      enable_perturbations=self.enable_perturbations,
    )
