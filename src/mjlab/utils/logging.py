"""Logging utilities for colored terminal output."""

import sys


def print_info(message: str, color: str = "green") -> None:
  """Print information message with color.

  Args:
    message: The message to print.
    color: Color name ('green', 'red', 'yellow', 'blue', 'cyan', 'magenta').
  """
  colors = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
  }

  if sys.stdout.isatty() and color in colors:
    print(f"{colors[color]}{message}\033[0m")
  else:
    print(message)
