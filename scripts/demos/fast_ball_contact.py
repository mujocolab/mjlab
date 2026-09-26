"""Demo: latching contact sensors catch fast ball hits missed by decimation.

A ball is launched at 10 m/s between two static paddles and bounces back and
forth like a rally. Each hit enters and leaves the paddle's contact zone
within a few physics substeps of one control step (decimation = 20), so by
the final substep the ball is already gone and the instantaneous ``found``
read reports no contact.

Three detection modes are compared over the same rollout:

- ``instant``: ``found`` read at the end of the control step (what you get
  without any substep-aware mechanism).
- ``history``: ``found_history`` (or ``force_history``) reduced over the
  decimation window (PR #699 style).
- ``latch``: ``found_any`` accumulated by ``catch_substep_contacts=True``
  and cleared at every control-step boundary.

Run with:
  uv run python scripts/demos/fast_ball_contact.py
  uv run python scripts/demos/fast_ball_contact.py --viewer
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

# One entity holds both paddles and the ball so the sensor can scope geoms.
RALLY_XML = """
<mujoco>
  <worldbody>
    <body name="paddle_left" pos="-0.3 0 0.15">
      <geom name="paddle_left_geom" type="box" size="0.01 0.15 0.15" mass="100"/>
    </body>
    <body name="paddle_right" pos="0.3 0 0.15">
      <geom name="paddle_right_geom" type="box" size="0.01 0.15 0.15" mass="100"/>
    </body>
    <body name="ball" pos="0 0 0.15">
      <freejoint/>
      <!-- Stiff near-elastic contact so a 10 m/s ball genuinely rebounds
           off the paddles instead of tunneling through them. -->
      <geom name="ball_geom" type="sphere" size="0.033" mass="0.058"
            solref="-100000 0"/>
    </body>
  </worldbody>
</mujoco>
"""

DECIMATION = 20
NUM_ENVS = 1
NUM_POLICY_STEPS = 60
PHYSICS_DT = 0.002
BALL_SPEED = 10.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build() -> tuple[Scene, Simulation]:
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(RALLY_XML))
  sensor_cfg = ContactSensorCfg(
    name="ball_contact",
    primary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    fields=("found", "force"),
    history_length=DECIMATION,
    catch_substep_contacts=True,
  )
  scene_cfg = SceneCfg(
    num_envs=NUM_ENVS,
    env_spacing=3.0,
    entities={"ball": entity_cfg},
    sensors=(sensor_cfg,),
  )
  scene = Scene(scene_cfg, DEVICE)
  model = scene.compile()
  sim = Simulation(
    num_envs=NUM_ENVS,
    cfg=SimulationCfg(njmax=50, mujoco=MujocoCfg(gravity=(0.0, 0.0, 0.0))),
    model=model,
    device=DEVICE,
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def launch_ball(scene: Scene):
  """Serve the ball from the center toward the right paddle."""
  root_state = torch.zeros((NUM_ENVS, 13), device=DEVICE)
  root_state[:, 2] = 0.15
  root_state[:, 3] = 1.0
  root_state[:, 7] = BALL_SPEED
  scene["ball"].write_root_state_to_sim(root_state)


def run_analysis():
  scene, sim = build()
  sensor = scene["ball_contact"]
  launch_ball(scene)

  print("=" * 70)
  print("Fast Ball Contact Demo")
  print(f"  Ball speed {BALL_SPEED} m/s between paddles at x = +/-0.3")
  print(f"  Physics dt = {PHYSICS_DT}s, decimation = {DECIMATION}")
  print(f"  Policy dt = {DECIMATION * PHYSICS_DT}s, steps = {NUM_POLICY_STEPS}")
  print("=" * 70)

  instant_hits = []
  history_hits = []
  latched_hits = []
  ball_x_substep = []

  for _ in range(NUM_POLICY_STEPS):
    sensor.begin_control_step()
    for _ in range(DECIMATION):
      sim.step()
      scene.update(dt=PHYSICS_DT)
      ball_x_substep.append(sim.data.qpos[0, 0].item())

    data = sensor.data
    instant_hits.append(bool(data.found[0, 0].item() > 0))
    force_mag = data.force_history[0, 0].norm(dim=-1)  # [H]
    history_hits.append(bool((force_mag > 1.0e-6).any().item()))
    latched_hits.append(bool(data.found_any[0, 0].item()))

  missed = [
    i for i in range(NUM_POLICY_STEPS) if latched_hits[i] and not instant_hits[i]
  ]
  print()
  print(f"Policy steps with contact (instant read):  {sum(instant_hits)}")
  print(f"Policy steps with contact (history):       {sum(history_hits)}")
  print(f"Policy steps with contact (latch):         {sum(latched_hits)}")
  print()
  if missed:
    print(f"Hit(s) MISSED by the instantaneous read but caught: {len(missed)}")
    print(f"  Policy steps: {missed}")
  else:
    print("No missed hits (try a faster ball or a larger decimation).")

  print()
  print("Step-by-step (first 60 policy steps):")
  print(f"{'step':>6}  {'instant':>8}  {'history':>8}  {'latch':>8}  {'missed':>8}")
  print("-" * 50)
  for i in range(min(60, NUM_POLICY_STEPS)):
    flag = " <<<" if (latched_hits[i] and not instant_hits[i]) else ""
    print(
      f"{i:>6}  {instant_hits[i]!s:>8}  {history_hits[i]!s:>8}  "
      f"{latched_hits[i]!s:>8}  {flag}"
    )

  _plot(ball_x_substep, instant_hits, history_hits, latched_hits, missed)


def _plot(
  ball_x_substep: list[float],
  instant_hits: list[bool],
  history_hits: list[bool],
  latched_hits: list[bool],
  missed: list[int],
):
  """Plot the ball trajectory with per-window detection outcomes."""
  t_substep = np.arange(len(ball_x_substep)) * PHYSICS_DT

  # Sample the trajectory at the END of each policy step for markers.
  step_end_x = [
    ball_x_substep[(i + 1) * DECIMATION - 1] for i in range(NUM_POLICY_STEPS)
  ]
  t_windows = np.arange(NUM_POLICY_STEPS) * DECIMATION * PHYSICS_DT + (
    DECIMATION * PHYSICS_DT
  )

  idx_both = [i for i in range(NUM_POLICY_STEPS) if instant_hits[i] and latched_hits[i]]
  idx_history_only = [
    i
    for i in range(NUM_POLICY_STEPS)
    if history_hits[i] and not instant_hits[i] and i not in idx_both
  ]

  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(t_substep, ball_x_substep, color="0.4", linewidth=0.8, label="Ball x")
  # Shade the paddle contact zones (ball-center x range that touches a paddle).
  for x_paddle in (-0.3, 0.3):
    ax.axvspan(x_paddle - 0.043, x_paddle + 0.043, color="tab:blue", alpha=0.08)

  if idx_both:
    ax.scatter(
      [t_windows[i] for i in idx_both],
      [step_end_x[i] for i in idx_both],
      color="tab:green",
      s=40,
      zorder=3,
      label="Detected by instant + latch",
    )
  if missed:
    ax.scatter(
      [t_windows[i] for i in missed],
      [step_end_x[i] for i in missed],
      color="tab:red",
      s=60,
      marker="x",
      linewidths=2,
      zorder=4,
      label="Caught by latch/history only",
    )
  if idx_history_only:
    ax.scatter(
      [t_windows[i] for i in idx_history_only],
      [step_end_x[i] for i in idx_history_only],
      color="tab:orange",
      s=40,
      zorder=3,
      label="Caught by history only",
    )

  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Ball x (m)")
  ax.set_title(
    f"Fast ball rally with decimation = {DECIMATION}: "
    f"{len(missed)} hits missed without substep-aware detection"
  )
  ax.legend(loc="upper right")
  fig.tight_layout()
  fig.savefig("scripts/demos/fast_ball_contact.png", dpi=150)
  print("\nPlot saved to scripts/demos/fast_ball_contact.png")
  plt.close(fig)


def run_viewer():
  """Launch a Viser viewer showing the rally with contact forces."""
  import viser

  from mjlab.viewer.viser import ViserMujocoScene

  scene, sim = build()
  launch_ball(scene)

  server = viser.ViserServer(label="Fast Ball Rally")
  viz = ViserMujocoScene(server, sim.mj_model, num_envs=NUM_ENVS)
  viz.show_contact_forces = True
  viz.show_contact_points = True
  viz.create_scene_gui(
    camera_distance=2.5,
    camera_azimuth=90.0,
    camera_elevation=15.0,
  )

  print("Open the Viser URL above to watch the rally.")
  print("Contact forces and points are enabled by default.")
  print("Press Ctrl+C to stop.\n")

  sensor = scene["ball_contact"]
  try:
    while True:
      sensor.begin_control_step()
      for _ in range(DECIMATION):
        sim.step()
        scene.update(dt=PHYSICS_DT)
      viz.update(sim.data)
      if viz.needs_update:
        viz.refresh_visualization()
      time.sleep(DECIMATION * PHYSICS_DT)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--viewer",
    action="store_true",
    help="Launch a Viser viewer instead of running the analysis.",
  )
  args = parser.parse_args()

  if args.viewer:
    run_viewer()
  else:
    run_analysis()


if __name__ == "__main__":
  main()
