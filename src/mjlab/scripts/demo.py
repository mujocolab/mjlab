"""Script to run a tracking demo with a pretrained policy.

This demo downloads a pretrained checkpoint and motion file from cloud storage
and launches an interactive viewer with a humanoid robot performing a cartwheel.
"""

import tyro

from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import PlayConfig, run_play


def main() -> None:
  """Run demo with pretrained tracking policy."""
  print("🎮 Setting up MJLab demo with pretrained tracking policy...")

  try:
    checkpoint_path = ensure_default_checkpoint()
    motion_path = ensure_default_motion()
  except RuntimeError as e:
    print(f"❌ Failed to download demo assets: {e}")
    print("Please check your internet connection and try again.")
    return

  play_config = PlayConfig(
    checkpoint_file=checkpoint_path,
    motion_file=motion_path,
    num_envs=8,
    viewer="viser",
  )

  args = tyro.cli(PlayConfig, default=play_config)

  run_play("Mjlab-Tracking-Flat-Unitree-G1-Demo", args)


if __name__ == "__main__":
  main()
