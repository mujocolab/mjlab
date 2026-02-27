"""Demo: contact sensor history catches collisions missed by decimation.

A bouncing ball with high restitution contacts the ground briefly during each
bounce. With large decimation, the contact may start and end within
intermediate substeps, so by the final substep there is no active contact and
instantaneous sensor reads miss it. Setting ``history_length = decimation``
captures every substep.

Run with:
  uv run python scripts/demos/contact_sensor_decimation.py
  uv run python scripts/demos/contact_sensor_decimation.py --viewer
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
from mjlab.sim.sim import Simulation, SimulationCfg

BOUNCING_BALL_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="ground" pos="0 0 0">
      <geom name="ground_geom" type="plane" size="5 5 0.1"/>
    </body>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.1"
            solref="-1000 0"/>
    </body>
  </worldbody>
</mujoco>
"""

DECIMATION = 20
NUM_ENVS = 1
NUM_POLICY_STEPS = 200
PHYSICS_DT = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(history_length: int) -> tuple[Scene, Simulation]:
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(BOUNCING_BALL_XML))
  sensor_cfg = ContactSensorCfg(
    name="ball_contact",
    primary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    secondary=None,
    fields=("found", "force"),
    history_length=history_length,
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
    cfg=SimulationCfg(njmax=50),
    model=model,
    device=DEVICE,
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def run_no_history():
  """Read instantaneous contact at the end of each policy step."""
  scene, sim = build(history_length=0)
  sensor = scene["ball_contact"]

  contact_detected = []
  for _ in range(NUM_POLICY_STEPS):
    for _ in range(DECIMATION):
      sim.step()
      scene.update(dt=PHYSICS_DT)
    found = sensor.data.found[0, 0].item() > 0
    contact_detected.append(found)

  return contact_detected


def run_with_history():
  """Read full substep history to catch mid-decimation contacts."""
  scene, sim = build(history_length=DECIMATION)
  sensor = scene["ball_contact"]

  contact_detected_instant = []
  contact_detected_history = []
  ball_height_substep = []
  for _ in range(NUM_POLICY_STEPS):
    for _ in range(DECIMATION):
      sim.step()
      scene.update(dt=PHYSICS_DT)
      # qpos is always current after step; qpos[2] is z for a freejoint.
      ball_height_substep.append(sim.data.qpos[0, 2].item())
    data = sensor.data
    found_instant = data.found[0, 0].item() > 0
    # Check whether any substep in the decimation window had contact.
    force_hist = data.force_history  # [B, N, H, 3]
    found_history = (force_hist[0, 0].norm(dim=-1) > 1e-6).any().item()
    contact_detected_instant.append(found_instant)
    contact_detected_history.append(found_history)

  return contact_detected_instant, contact_detected_history, ball_height_substep


def run_viewer():
  """Launch a Viser viewer showing the bouncing ball with contact forces."""
  import viser

  from mjlab.viewer.viser import ViserMujocoScene

  scene, sim = build(history_length=0)

  server = viser.ViserServer(label="Bouncing Ball")
  viz = ViserMujocoScene.create(server, sim.mj_model, num_envs=NUM_ENVS)
  viz.show_contact_forces = True
  viz.show_contact_points = True
  viz.create_visualization_gui(
    camera_distance=2.0,
    camera_azimuth=90.0,
    camera_elevation=20.0,
  )

  print("Open the Viser URL above to watch the bouncing ball.")
  print("Contact forces and points are enabled by default.")
  print("Press Ctrl+C to stop.\n")

  try:
    while True:
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


def run_analysis():
  """Run the analysis comparing instantaneous vs history contact detection."""
  print("=" * 70)
  print("Contact Sensor Decimation Demo")
  print(f"  Ball dropped from 1m, restitution ~ 1, decimation = {DECIMATION}")
  print(f"  Physics dt = {PHYSICS_DT}s, policy dt = {DECIMATION * PHYSICS_DT}s")
  print("=" * 70)

  no_hist = run_no_history()
  instant, history, ball_height = run_with_history()

  # Policy steps where history caught a contact that instant missed.
  missed = []
  for i in range(NUM_POLICY_STEPS):
    if history[i] and not instant[i]:
      missed.append(i)

  total_contacts_instant = sum(instant)
  total_contacts_history = sum(history)
  total_contacts_no_hist = sum(no_hist)

  print()
  print(f"Total policy steps with contact (no history):   {total_contacts_no_hist}")
  print(f"Total policy steps with contact (instant only): {total_contacts_instant}")
  print(f"Total policy steps with contact (with history): {total_contacts_history}")
  print()

  if missed:
    print(f"Contacts MISSED by instantaneous read but CAUGHT by history: {len(missed)}")
    print(f"  Policy steps: {missed}")
  else:
    print("No missed contacts (try increasing decimation or adjusting drop height)")

  print()
  print("Step-by-step (showing first 60 policy steps):")
  print(f"{'step':>6}  {'no_hist':>8}  {'instant':>8}  {'history':>8}  {'missed':>8}")
  print("-" * 50)
  for i in range(min(60, NUM_POLICY_STEPS)):
    flag = " <<<" if (history[i] and not instant[i]) else ""
    print(f"{i:>6}  {no_hist[i]!s:>8}  {instant[i]!s:>8}  {history[i]!s:>8}  {flag}")

  # --- Plot ---
  total_substeps = NUM_POLICY_STEPS * DECIMATION
  t_substep = np.arange(total_substeps) * PHYSICS_DT

  # Place markers at the minimum height within each policy step window.
  min_height = []
  min_time = []
  for i in range(NUM_POLICY_STEPS):
    start = i * DECIMATION
    end = (i + 1) * DECIMATION
    window = ball_height[start:end]
    j = int(np.argmin(window))
    min_height.append(window[j])
    min_time.append(t_substep[start + j])

  # Separate history detections into: caught by both, caught only by history.
  idx_both = [i for i in range(NUM_POLICY_STEPS) if instant[i] and history[i]]
  idx_history_only = missed  # history=True, instant=False

  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(t_substep, ball_height, color="0.4", linewidth=0.8, label="Ball height")

  if idx_both:
    ax.scatter(
      [min_time[i] for i in idx_both],
      [min_height[i] for i in idx_both],
      color="tab:green",
      s=40,
      zorder=3,
      label="Detected by both",
    )

  if idx_history_only:
    ax.scatter(
      [min_time[i] for i in idx_history_only],
      [min_height[i] for i in idx_history_only],
      color="tab:red",
      s=60,
      marker="x",
      linewidths=2,
      zorder=4,
      label="Caught only by history",
    )

  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Ball height (m)")
  ax.set_title(
    f"Contact sensor with decimation = {DECIMATION}: "
    f"{len(missed)} collisions missed without history"
  )
  ax.legend(loc="upper right")
  ax.set_ylim(bottom=-0.05)
  fig.tight_layout()
  fig.savefig("scripts/demos/contact_sensor_decimation.png", dpi=150)
  print("\nPlot saved to scripts/demos/contact_sensor_decimation.png")
  plt.close(fig)


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
