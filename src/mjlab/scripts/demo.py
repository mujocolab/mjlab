"""Script to run a tracking demo with a pretrained policy.

This demo downloads a pretrained checkpoint and motion file from cloud storage
and launches an interactive viewer with a humanoid robot performing a cartwheel.
"""

from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import run_play

_TASK_NAME = "Mjlab-Tracking-Flat-Unitree-G1-Play"
_N_ENVS = 8


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

  print(f"🚀 Launching {_N_ENVS} environments with cartwheel motion...")

  run_play(
    task=_TASK_NAME,
    checkpoint_file=checkpoint_path,
    motion_file=motion_path,
    num_envs=_N_ENVS,
    render_all_envs=True,
    viewer="viser",
  )


if __name__ == "__main__":
  main()
