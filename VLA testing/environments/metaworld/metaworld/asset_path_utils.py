"""Set of utilities for retrieving asset paths for the environments."""

from __future__ import annotations

from pathlib import Path

_CURRENT_FILE_DIR = Path(__file__).parent.absolute()

ENV_ASSET_DIR_V3 = _CURRENT_FILE_DIR / "assets"


def full_V3_path_for(file_name: str) -> str:
    """Retrieves the full, absolute path for a given V3 asset

    Args:
        file_name: Name of the asset file. Can include subdirectories.

    Returns:
        The full path to the asset file.
    """
    return str(ENV_ASSET_DIR_V3 / file_name)
