"""Configuration of the sampling MPC planner (no simulation imports)."""

from dataclasses import dataclass


@dataclass
class SamplingMpcCfg:
  num_samples: int = 64
  """Action sequences per real env and iteration (K). Sample 0 is the
  noise-free nominal plan."""
  horizon: int = 20
  """Planning horizon in control steps (H)."""
  iterations: int = 1
  """Sampling batches per control step (L). 1 = MPPI, > 1 = MPOPI."""
  noise_std: float = 0.5
  """Initial std of the action perturbations, in policy action units."""
  target_ess: float | None = 0.1
  """Normalized effective sample size the MPPI temperature is tuned to. None
  uses the fixed ``temperature``."""
  temperature: float = 0.1
  """Fixed temperature when ``target_ess`` is None."""
  action_clip: float | None = 1.0
  """Clip sampled actions to ``[-x, x]``. None disables clipping."""
  std_smoothing: float = 0.7
  """MPOPI only: blend factor for the adapted std between iterations."""
  min_std_scale: float = 0.2
  """MPOPI only: lower bound on the adapted std, as a fraction of noise_std."""
  max_std_scale: float = 3.0
  """MPOPI only: upper bound on the adapted std, as a fraction of noise_std."""
  num_knots: int | None = None
  """Sample the noise at this many evenly spaced knots over the horizon and
  interpolate linearly between them (smooth perturbations, like the spline
  plans of MuJoCo MPC). None samples independent noise at every step, which
  action-rate penalties punish heavily on high-dimensional robots."""
  seed: int = 0
  """Seed of the planner's private random generator."""
