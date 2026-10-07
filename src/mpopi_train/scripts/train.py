"""``mpopi-train``: mjlab's ``train`` command with the MPOPI tasks registered.

Example::

  uv run mpopi-train Mpopi-G1-2k-Replay-IS-DAgger --agent.seed 1
"""

import mpopi_train.tasks  # noqa: F401  (registers the Mpopi-* tasks)
from mjlab.scripts.train import main as mjlab_train


def main() -> None:
  mjlab_train()


if __name__ == "__main__":
  main()
