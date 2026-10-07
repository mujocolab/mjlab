"""``mpopi-play``: mjlab's ``play`` command with the MPOPI tasks registered."""

import mpopi_train.tasks  # noqa: F401  (registers the Mpopi-* tasks)
from mjlab.scripts.play import main as mjlab_play


def main() -> None:
  mjlab_play()


if __name__ == "__main__":
  main()
