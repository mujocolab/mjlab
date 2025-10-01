# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to run a tracking demo with a pretrained policy.

This demo downloads a pretrained checkpoint and motion file from cloud storage
and launches an interactive viewer with a humanoid robot performing a cartwheel.
"""

from functools import partial

import tyro

from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import run_play


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

  tyro.cli(
    partial(
      run_play,
      task="Mjlab-Tracking-Flat-Unitree-G1-Play",
      checkpoint_file=checkpoint_path,
      motion_file=motion_path,
      num_envs=8,
      render_all_envs=True,
      viewer="viser",
    )
  )


if __name__ == "__main__":
  main()
