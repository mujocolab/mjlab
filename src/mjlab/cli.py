# src/mjlab/cli.py
"""MJLab command-line interface."""

import argparse
import subprocess
import sys
from pathlib import Path


def _get_script_path(*parts: str) -> Path:
  """Get path to script relative to project root."""
  # Assume we're in src/mjlab/cli.py, so go up to project root
  project_root = Path(__file__).parent.parent.parent
  return project_root / "scripts" / Path(*parts)


def tracking_play():
  """Play trained motion tracking policy with auto-downloaded assets."""
  parser = argparse.ArgumentParser(description="Play motion tracking policy")
  parser.add_argument(
    "--task",
    default="Mjlab-Tracking-Flat-G1-Play",
    help="Task name (default: Mjlab-Tracking-Flat-G1-Play)",
  )
  parser.add_argument(
    "--device",
    default="cuda:0",
    help="Device to use (default: cuda:0)",
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=8,
    help="Number of environments",
  )
  parser.add_argument(
    "--viewer",
    choices=["native", "viser"],
    default="native",
    help="Viewer type (default: native)",
  )
  parser.add_argument(
    "--render-all-envs",
    action="store_true",
    default=True,
    help="Render all environments",
  )

  args, unknown = parser.parse_known_args()

  # Auto-download assets from GCS.
  from mjlab.utils.download import ensure_default_checkpoint, ensure_default_motion

  print("🚀 Setting up MJLab demo...")
  checkpoint_file = ensure_default_checkpoint()
  motion_file = ensure_default_motion()

  cmd = [
    sys.executable,
    str(_get_script_path("tracking", "rl", "play.py")),
    "--task",
    args.task,
    "--checkpoint-file",
    checkpoint_file,
    "--motion-file",
    motion_file,
    "--device",
    args.device,
  ]

  # Add optional arguments.
  if args.num_envs:
    cmd.extend(["--num-envs", str(args.num_envs)])
  if args.viewer != "native":
    cmd.extend(["--viewer", args.viewer])
  if args.render_all_envs:
    cmd.append("--render-all-envs")

  cmd.extend(unknown)

  print("▶️  Starting demo...")
  subprocess.run(cmd)


def main():
  """Main CLI entry point."""
  import argparse

  parser = argparse.ArgumentParser(
    description="🤖 Welcome to MJLab",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Quick demo (downloads assets automatically)
  mjlab tracking-play
  
  # With custom viewer
  mjlab tracking-play --viewer viser
  
  # Multiple environments
  mjlab tracking-play --num-envs 4 --render-all-envs
        """,
  )
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  tracking_parser = subparsers.add_parser(
    "tracking-play",
    help="🏃 Play trained motion tracking policy (auto-downloads demo assets)",
  )
  tracking_parser.set_defaults(func=tracking_play)

  args, remaining = parser.parse_known_args()

  if args.command == "tracking-play":
    sys.argv = [sys.argv[0]] + remaining
    tracking_play()
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
