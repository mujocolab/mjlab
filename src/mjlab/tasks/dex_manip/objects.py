from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

DEFAULT_DEX_OBJECTS: Final[tuple[str, ...]] = (
  "water-bottle",
  "orange",
  "tuna-fish-can",
)

_OBJECT_TO_MESH_NAME: Final[dict[str, str]] = {
  "water-bottle": "water_bottle_mesh",
  "orange": "orange_mesh",
  "tuna-fish-can": "tuna_fish_can_mesh",
}

_ASSETS_ROOT = Path(__file__).resolve().parent / "assets" / "objects" / "meshes"
_OBJECT_TO_MESH_FILE: Final[dict[str, Path]] = {
  "water-bottle": _ASSETS_ROOT / "contactdb_objects" / "water_bottle.stl",
  "orange": _ASSETS_ROOT / "ycb_objects" / "orange.stl",
  "tuna-fish-can": _ASSETS_ROOT / "ycb_objects" / "tuna_fish_can.stl",
}


def available_object_names() -> tuple[str, ...]:
  return tuple(_OBJECT_TO_MESH_NAME.keys())


def parse_object_selection(raw: str | None) -> tuple[str, ...]:
  if raw is None or raw.strip() == "" or raw.strip().lower() == "all":
    return DEFAULT_DEX_OBJECTS

  tokens = [t.strip().lower() for t in re.split(r"[;,\s]+", raw) if t.strip()]
  if not tokens:
    raise ValueError("Object selection is empty.")

  normalized = tuple(token.replace("_", "-") for token in tokens)
  unknown = [name for name in normalized if name not in _OBJECT_TO_MESH_NAME]
  if unknown:
    raise ValueError(
      f"Unknown objects: {unknown}. Available: {sorted(_OBJECT_TO_MESH_NAME.keys())}"
    )
  return normalized


def object_names_to_mesh_names(object_names: tuple[str, ...]) -> tuple[str, ...]:
  unknown = [name for name in object_names if name not in _OBJECT_TO_MESH_NAME]
  if unknown:
    raise ValueError(
      f"Unknown objects: {unknown}. Available: {sorted(_OBJECT_TO_MESH_NAME.keys())}"
    )
  return tuple(_OBJECT_TO_MESH_NAME[name] for name in object_names)


def object_names_to_mesh_files(object_names: tuple[str, ...]) -> tuple[Path, ...]:
  unknown = [name for name in object_names if name not in _OBJECT_TO_MESH_FILE]
  if unknown:
    raise ValueError(
      f"Unknown objects: {unknown}. Available: {sorted(_OBJECT_TO_MESH_FILE.keys())}"
    )
  missing = [name for name in object_names if not _OBJECT_TO_MESH_FILE[name].is_file()]
  if missing:
    raise FileNotFoundError(
      "Missing mesh files for objects: "
      f"{missing}. Expected under {_ASSETS_ROOT}."
    )
  return tuple(_OBJECT_TO_MESH_FILE[name].resolve() for name in object_names)


def resolve_mesh_id(env: "ManagerBasedRlEnv", mesh_name: str) -> int:
  """Resolve a mesh id by full name or unique basename."""
  cache_key = f"_dex_manip_mesh_id::{mesh_name}"
  cached = getattr(env, cache_key, None)
  if cached is not None:
    return int(cached)

  available_names = [mesh.name for mesh in env.scene.spec.meshes if mesh.name is not None]
  exact_name_to_id = {name: idx for idx, name in enumerate(available_names)}

  if mesh_name in exact_name_to_id:
    mesh_id = exact_name_to_id[mesh_name]
  else:
    suffix_matches = [
      idx
      for idx, full_name in enumerate(available_names)
      if full_name.rsplit("/", 1)[-1] == mesh_name
    ]
    if len(suffix_matches) == 1:
      mesh_id = suffix_matches[0]
    elif len(suffix_matches) > 1:
      matched = [available_names[idx] for idx in suffix_matches]
      raise ValueError(
        f"Mesh name {mesh_name!r} is ambiguous. Matches: {matched}. "
        "Use a fully qualified mesh name."
      )
    else:
      raise ValueError(
        f"Unknown mesh name {mesh_name!r}. Available: {sorted(available_names)}"
      )

  setattr(env, cache_key, mesh_id)
  return mesh_id
