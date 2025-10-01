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
