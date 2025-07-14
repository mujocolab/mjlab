from typing import Any, Dict, Union
from pathlib import Path

def update_assets(
  assets: Dict[str, Any],
  path: Union[str, Path],
  glob: str = "*",
  recursive: bool = False,
):
  for f in Path(path).glob(glob):
    if f.is_file():
      assets[f.name] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, glob, recursive)