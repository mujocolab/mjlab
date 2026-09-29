"""Batched 14-node lumped-parameter thermal network (LPTN) for a quadruped.

Star topology: each of the 12 actuator nodes conducts heat into a single
shared chassis node through a fixed resistance (``Rth``); the chassis node
convects to ambient air through a resistance that shrinks with base linear
velocity (faster motion -> more airflow -> better cooling). Ambient is
modeled as a Dirichlet boundary condition rather than a 14th dynamic state:
real outside air has no principled finite thermal capacitance to assign it,
whereas a fixed boundary temperature is exact and standard practice for
LPTN models. It occupies node index 13 purely so the state vector, and the
``A``/``B`` matrices of the discrete state-space update, stay a uniform
14-wide shape.

Discretized with explicit forward Euler at ``dt = env.step_dt``. Thermal
time constants here are ``Rth * Cth ~= 800 s``, far larger than a control
step (~0.02 s), so Euler integration is not a stability concern and there's
no need for an exact matrix-exponential discretization.

Cross-checked against published thermal-aware quadruped locomotion work
(Qian et al., "Learning Thermal-Aware Locomotion Policies for an
Electrically-Actuated Quadruped Robot," arXiv:2603.01631; Wan et al.,
"Learning to Balance Motor Thermal Safety and Quadrupedal Locomotion
Performance with Residual Policy," arXiv:2605.27046). Both use the same
14-node LPTN topology (12 motors + 1 non-actuator node + ambient) updated
at 50 Hz synchronized with a 200 Hz PD/physics loop -- an exact match for
this module's node count and the `step_dt`/`physics_dt` split used
elsewhere in this package. Their heat input is RMS torque over the 200 Hz
samples inside each 50 Hz interval; since heat ~ I^2 and RMS(x)^2 =
mean(x^2), that's the same operation as this package's per-physics-substep
mean-of-I^2 accumulation (see `ThermalEnergyObservation._accumulate_substep`
in `heat_bench/envs_mjlab/eval_observations.py`). The one difference is
discretization method: both papers use zero-order-hold (exact,
matrix-exponential) discretization, kept here as forward Euler instead --
ZOH was benchmarked at ~140x the per-step cost of the current `bmm`-based
Euler update (batched `torch.linalg.matrix_exp` on an augmented `(N, 26,
26)` system, required since the ambient row makes `A` singular) for no
accuracy benefit given the time-constant margin above.
"""

from __future__ import annotations

import torch

NUM_JOINTS = 12
CHASSIS_IDX = 12
AMBIENT_IDX = 13
NUM_NODES = 14


class BatchedLPTNEngine:
  """Vectorized 14-node LPTN thermal state, stepped once per env control step."""

  def __init__(self, cfg: dict, num_envs: int, device: str):
    self.num_envs = num_envs
    self.device = device

    self._cth_joint = float(cfg["joint_thermal_capacitance_Cth"])
    self._cth_chassis = float(cfg["chassis_thermal_capacitance_Cth_chassis"])
    self._g_joint = 1.0 / float(cfg["joint_thermal_resistance_Rth"])
    self._conv_base = float(cfg["chassis_convection_base_conductance"])
    self._conv_gain = float(cfg["chassis_convection_velocity_gain"])
    self.ambient_temperature = float(cfg["ambient_temperature_c"])
    self._init_joint_temp = float(cfg["initial_joint_temperature_c"])
    self._init_chassis_temp = float(cfg["initial_chassis_temperature_c"])

    # Per-node thermal capacitance. Ambient's entry is never used (its row
    # of K is all zero, so its dT/dt is always 0) but must be nonzero to
    # avoid a division by zero when building C^-1.
    c = torch.full((NUM_NODES,), self._cth_joint, device=device)
    c[CHASSIS_IDX] = self._cth_chassis
    c[AMBIENT_IDX] = self._cth_chassis
    self._c_inv = 1.0 / c

    # Env-independent conductance template: joint<->chassis links and the
    # chassis self-term contribution from those links. The chassis<->ambient
    # convective term is velocity-dependent and added per step in `step()`.
    k_base = torch.zeros(NUM_NODES, NUM_NODES, device=device)
    joint_idx = torch.arange(NUM_JOINTS, device=device)
    k_base[joint_idx, joint_idx] = self._g_joint
    k_base[joint_idx, CHASSIS_IDX] = -self._g_joint
    k_base[CHASSIS_IDX, joint_idx] = -self._g_joint
    k_base[CHASSIS_IDX, CHASSIS_IDX] = self._g_joint * NUM_JOINTS
    self._k_base = k_base

    # Heat input matrix: Joule heat for actuator j is injected directly into
    # node j. Chassis and ambient receive no direct electrical input.
    b_raw = torch.zeros(NUM_NODES, NUM_JOINTS, device=device)
    b_raw[joint_idx, joint_idx] = 1.0
    self._b_raw = b_raw

    self.T = torch.zeros(num_envs, NUM_NODES, device=device)
    self.reset(env_ids=None)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.T[env_ids, :NUM_JOINTS] = self._init_joint_temp
    self.T[env_ids, CHASSIS_IDX] = self._init_chassis_temp
    self.T[env_ids, AMBIENT_IDX] = self.ambient_temperature

  def step(
    self, joule_heat: torch.Tensor, base_lin_vel_xy: torch.Tensor, dt: float
  ) -> torch.Tensor:
    """Advance thermal state by one control step.

    Args:
      joule_heat: Per-actuator I^2*Rd heat input, shape (N, 12), watts.
      base_lin_vel_xy: Base linear velocity in the xy-plane, shape (N, 2).
      dt: Control step duration (env.step_dt).

    Returns:
      Updated node temperatures, shape (N, 14).
    """
    n = self.T.shape[0]
    v_xy = torch.linalg.norm(base_lin_vel_xy, dim=-1)
    g_conv = self._conv_base + self._conv_gain * v_xy  # (N,)

    k_t = self._k_base.unsqueeze(0).expand(n, -1, -1).clone()
    k_t[:, CHASSIS_IDX, CHASSIS_IDX] += g_conv
    k_t[:, CHASSIS_IDX, AMBIENT_IDX] = -g_conv

    eye = torch.eye(NUM_NODES, device=self.device).unsqueeze(0)
    a_t = eye - dt * self._c_inv.view(1, NUM_NODES, 1) * k_t
    b_scaled = dt * self._c_inv.unsqueeze(-1) * self._b_raw  # (14, 12)

    t_next = torch.bmm(a_t, self.T.unsqueeze(-1)).squeeze(-1)
    t_next = t_next + joule_heat @ b_scaled.T
    t_next[:, AMBIENT_IDX] = self.ambient_temperature

    self.T = t_next
    return self.T
