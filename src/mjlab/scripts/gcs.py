import hashlib
import tempfile
from pathlib import Path
from typing import TypedDict

import requests
from tqdm import tqdm

CACHE_DIR = Path(tempfile.gettempdir()) / "mjlab_cache"


class AssetInfo(TypedDict):
  url: str
  sha256: str
  path: Path


ASSETS: dict[str, AssetInfo] = {
  "demo_ckpt.pt": {
    "url": "https://storage.googleapis.com/mjlab_beta/model_49999.pt",
    "sha256": "f1bc404f5165b833a3933ac98ff223450392b78df3eb76b0c36cd1360ff22319",
    "path": CACHE_DIR / "demo_ckpt.pt",
  },
  "demo_motion.npz": {
    "url": "https://storage.googleapis.com/mjlab_beta/lafan_dance1_subject1.npz",
    "sha256": "f08d15d4b5bb605e17b6928ccdb44ae6ce7bf2038111e8b145f12a176cd096d4",
    "path": CACHE_DIR / "lafan1_dance1_subject1_demo_motion.npz",
  },
}


def download_with_progress(url: str, path: Path) -> None:
  """Download file with progress bar."""
  response = requests.get(url, stream=True)
  response.raise_for_status()

  total_size = int(response.headers.get("content-length", 0))

  with (
    open(path, "wb") as f,
    tqdm(
      desc=path.name,
      total=total_size,
      unit="B",
      unit_scale=True,
      unit_divisor=1024,
    ) as pbar,
  ):
    for chunk in response.iter_content(chunk_size=8192):
      size = f.write(chunk)
      pbar.update(size)


def verify_file_hash(path: Path, expected_hash: str) -> bool:
  """Verify file integrity using SHA256."""
  if not path.exists():
    return False

  sha256_hash = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      sha256_hash.update(chunk)

  return sha256_hash.hexdigest() == expected_hash


def ensure_asset_downloaded(asset_name: str, force_download: bool = False) -> Path:
  """Download and verify an asset if needed."""
  if asset_name not in ASSETS:
    raise ValueError(f"Unknown asset: {asset_name}")

  asset_info = ASSETS[asset_name]
  local_path = Path(asset_info["path"])

  if not force_download and verify_file_hash(local_path, asset_info["sha256"]):
    print(f"✓ {asset_name} already cached at {local_path}")
    return local_path

  print(f"📥 Downloading {asset_name}...")
  local_path.parent.mkdir(parents=True, exist_ok=True)

  try:
    download_with_progress(asset_info["url"], local_path)

    if not verify_file_hash(local_path, asset_info["sha256"]):
      local_path.unlink()  # Delete corrupted file
      raise RuntimeError(f"Downloaded {asset_name} failed hash verification")

    print(f"✅ {asset_name} cached at {local_path}")
    return local_path

  except Exception as e:
    if local_path.exists():
      local_path.unlink()
    raise RuntimeError(f"Failed to download {asset_name}: {e}") from e


def ensure_default_checkpoint() -> str:
  """Ensure default checkpoint is available and return its absolute path."""
  checkpoint_path = ensure_asset_downloaded("demo_ckpt.pt")
  return str(checkpoint_path.resolve())


def ensure_default_motion() -> str:
  """Ensure default motion file is available and return its absolute path."""
  motion_path = ensure_asset_downloaded("demo_motion.npz")
  return str(motion_path.resolve())
