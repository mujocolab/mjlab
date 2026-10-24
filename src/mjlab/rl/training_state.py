"""Versioned, opt-in continuation state for native PPO.

Only data is serialized. Runtime objects are supplied by the newly constructed
environment; arrays are restored in place to retain Warp/CUDA graph pointers.
Custom components must implement the explicit protocol below or declare themselves
stateless. Unsupported resources fail at preflight, before collecting experience.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import platform
import random
from collections import deque
from dataclasses import fields, is_dataclass
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv

FORMAT_VERSION = 1


class TrainingStateful(Protocol):
  """State protocol for custom terms, commands, actuators and sensors.

  Return a tree of tensors, NumPy arrays, scalars and ordinary containers. Include
  every mutable value that influences future outputs, including private generators.
  Restore into the existing object and preserve aliases to simulation arrays.
  A component with no such state may instead set ``training_state_stateless=True``.
  Functions outside mjlab must also explicitly declare that stateless contract.
  """

  def training_state_dict(self) -> dict: ...

  def load_training_state_dict(self, state: dict) -> None: ...


def _type_name(value: Any) -> str:
  cls = value if isinstance(value, type) else type(value)
  return f"{cls.__module__}.{cls.__qualname__}"


def _function_name(value: Any) -> str:
  return f"{value.__module__}.{value.__qualname__}"


def _precision_settings() -> dict:
  # Torch 2.9 introduced per-backend fp32_precision. Mixing its legacy getters
  # with that API raises on newer Torch, so choose one complete API family.
  matmul = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
  if matmul is not None:
    return {
      "api": "per_backend",
      "matmul": str(matmul),
      "cudnn": str(torch.backends.cudnn.fp32_precision),
      "global": str(getattr(torch.backends, "fp32_precision", None)),
    }
  return {
    "api": "legacy",
    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    "matmul_precision": torch.get_float32_matmul_precision(),
  }


def _callable_constant(value: Any) -> Any:
  """Only immutable captured inputs belong to a stateless custom function."""
  if value is None or isinstance(
    value, (str, int, float, bool, np.generic, Enum, Path, slice)
  ):
    return _configuration(value)
  if isinstance(value, tuple):
    return [_callable_constant(item) for item in value]
  raise ValueError(
    "Custom stateless callable defaults/closures must contain immutable data; "
    "use an explicit training-state component for mutable or opaque captures"
  )


def _configuration(value: Any) -> Any:
  """Stable identity of construction inputs; never serialize callable objects."""
  if is_dataclass(value) and not isinstance(value, type):
    return {f.name: _configuration(getattr(value, f.name)) for f in fields(value)}
  if isinstance(value, dict):
    return {str(k): _configuration(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_configuration(v) for v in value]
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, Enum):
    return [_type_name(value), value.name]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, slice):
    return ["slice", value.start, value.stop, value.step]
  if isinstance(value, torch.Tensor):
    return [str(value.dtype), value.detach().cpu().tolist()]
  if isinstance(value, np.ndarray):
    return [str(value.dtype), value.tolist()]
  if inspect.isfunction(value) or isinstance(value, type):
    identity: dict[str, Any] = {"callable": _function_name(value)}
    if not value.__module__.startswith("mjlab."):
      try:
        identity["source_sha256"] = hashlib.sha256(
          inspect.getsource(value).encode()
        ).hexdigest()
      except (OSError, TypeError) as exc:
        raise ValueError(
          f"Cannot fingerprint custom callable {identity['callable']}"
        ) from exc
      if inspect.isfunction(value):
        identity["defaults"] = _callable_constant(value.__defaults__)
        identity["kwdefaults"] = {
          key: _callable_constant(item)
          for key, item in (value.__kwdefaults__ or {}).items()
        }
        identity["closure"] = {
          name: _callable_constant(cell.cell_contents)
          for name, cell in zip(
            value.__code__.co_freevars, value.__closure__ or (), strict=True
          )
        }
    return identity
  raise ValueError(f"Unsupported stateful-resume configuration: {_type_name(value)}")


def compatibility(env: ManagerBasedRlEnv, cfg: dict, learner_device: str) -> dict:
  """Require the same task, learner, batch topology and software versions."""
  learner = {
    k: cfg.get(k)
    for k in (
      "actor",
      "critic",
      "algorithm",
      "obs_groups",
      "num_steps_per_env",
      "clip_actions",
    )
  }
  construction = _configuration({"environment": env.cfg, "learner": learner})
  digest = hashlib.sha256(json.dumps(construction, sort_keys=True).encode()).hexdigest()
  binary = np.empty(mujoco.mj_sizeModel(env.sim.mj_model), dtype=np.uint8)
  mujoco.mj_saveModel(env.sim.mj_model, buffer=binary)
  source_hash = hashlib.sha256()
  source_root = Path(__file__).resolve().parents[1]
  for source in sorted(source_root.rglob("*.py")):
    source_hash.update(source.relative_to(source_root).as_posix().encode())
    source_hash.update(source.read_bytes())
  return {
    "configuration_sha256": digest,
    "model_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
    "implementation_sha256": source_hash.hexdigest(),
    "simulation_constants": simulation_constants(env),
    "num_envs": env.num_envs,
    "device_type": torch.device(env.device).type,
    "learner_device": str(torch.device(learner_device)),
    "execution": {
      "python": platform.python_version(),
      "machine": platform.machine(),
      "default_dtype": str(torch.get_default_dtype()),
      "default_device": str(torch.get_default_device()),
      "threads": torch.get_num_threads(),
      "interop_threads": torch.get_num_interop_threads(),
      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
      "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
      "fp32_precision": _precision_settings(),
      "mkldnn": torch.backends.mkldnn.enabled,
      "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
      "cudnn_deterministic": torch.backends.cudnn.deterministic,
      "cudnn_benchmark": torch.backends.cudnn.benchmark,
      "cuda_devices": [
        str(torch.cuda.get_device_properties(i))
        for i in range(torch.cuda.device_count())
      ]
      if "cuda" in (torch.device(env.device).type, torch.device(learner_device).type)
      else [],
    },
    "versions": {
      name: version(name)
      for name in (
        "mjlab",
        "rsl-rl-lib",
        "mujoco",
        "mujoco-warp",
        "torch",
        "warp-lang",
        "numpy",
      )
    },
  }


def rng_state() -> dict:
  return {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
  }


def restore_rng(state: dict) -> None:
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"].cpu())
  if state["cuda"]:
    if len(state["cuda"]) != torch.cuda.device_count():
      raise ValueError("Stateful resume requires the same visible CUDA devices")
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def host_state(value: Any) -> Any:
  """Detach learner/runtime tensors before transferring snapshots between ranks."""
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().clone()
  if isinstance(value, dict):
    return {key: host_state(child) for key, child in value.items()}
  if isinstance(value, tuple):
    return tuple(host_state(child) for child in value)
  if isinstance(value, list):
    return [host_state(child) for child in value]
  if isinstance(value, deque):
    return deque((host_state(child) for child in value), maxlen=value.maxlen)
  return value


def _warp_arrays(value: Any, prefix: str = "") -> dict[str, wp.array]:
  if isinstance(value, wp.array):
    return {prefix: value}
  if isinstance(value, dict):
    children = value.items()
  elif isinstance(value, (list, tuple)):
    children = enumerate(value)
  elif type(value).__module__.startswith("mujoco_warp") and hasattr(value, "__dict__"):
    children = vars(value).items()
  else:
    return {}
  result = {}
  for name, child in children:
    path = f"{prefix}.{name}" if prefix else str(name)
    result.update(_warp_arrays(child, path))
  return result


def _simulation_arrays(env: ManagerBasedRlEnv) -> dict[str, wp.array]:
  return {
    **{f"model.{k}": v for k, v in _warp_arrays(env.sim.wp_model).items()},
    **{f"data.{k}": v for k, v in _warp_arrays(env.sim.wp_data).items()},
  }


def simulation_constants(env: ManagerBasedRlEnv) -> dict:
  """Immutable host-side flags/layout that are baked into native Warp graphs.

  Restoring a changed solver/feature flag without recapturing every affected graph
  is unsafe. Require equality and fail if a component changes these at runtime.
  Mutable numeric options such as timestep/gravity are Warp arrays and are saved.
  """

  def constants(value: Any) -> dict:
    if isinstance(value, wp.array):
      return {"array": True}
    if isinstance(value, dict):
      return {str(k): constants(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
      return {str(i): constants(v) for i, v in enumerate(value)}
    if type(value).__module__.startswith("mujoco_warp") and hasattr(value, "__dict__"):
      return {k: constants(v) for k, v in vars(value).items()}
    return _configuration(value)

  return {"model": constants(env.sim.wp_model), "data": constants(env.sim.wp_data)}


def capture_simulation(env: ManagerBasedRlEnv, expected_constants: dict) -> dict:
  # Include all derived arrays too: a forward()/set_const() would change the exact
  # state at this update boundary, and would discard live per-world DR parameters.
  current_constants = simulation_constants(env)
  if current_constants != expected_constants:
    raise ValueError(
      "Stateful resume does not support runtime changes to host simulation flags/layout"
    )
  return {
    "constants": current_constants,
    "arrays": {
      name: array.numpy().copy() for name, array in _simulation_arrays(env).items()
    },
  }


def validate_simulation(env: ManagerBasedRlEnv, state: dict) -> None:
  arrays = _simulation_arrays(env)
  if simulation_constants(env) != state["constants"]:
    raise ValueError("Stateful resume simulation flags/layout differ")
  if arrays.keys() != state["arrays"].keys():
    raise ValueError("Stateful resume simulation field layout differs")
  for name, array in arrays.items():
    saved = state["arrays"][name]
    current = array.numpy()
    if current.shape != saved.shape or current.dtype != saved.dtype:
      raise ValueError(f"Stateful resume simulation array differs: {name}")


def restore_simulation(env: ManagerBasedRlEnv, state: dict) -> None:
  for name, array in _simulation_arrays(env).items():
    array.assign(state["arrays"][name])


# Native compiled/spec objects are immutable construction resources. Their
# simulation arrays are captured separately; do not replace bridges or pointers.
_STATIC_MODULES = ("mujoco.", "warp.")
_BUILTIN_MODULES = (
  "mjlab.managers.",
  "mjlab.envs.mdp.",
  "mjlab.tasks.",
  "mjlab.entity.",
  "mjlab.actuator.",
  "mjlab.sensor.",
  "mjlab.scene.",
  "mjlab.terrains.",
  "mjlab.utils.buffers.",
  "mjlab.utils.noise.",
  "mjlab.utils.spaces",
)


class ComponentState:
  """Data-only graph codec for native components and explicit custom hooks.

  Object identities, tensor storage aliases and Warp pointers belong to the fresh
  runtime and are retained. Config objects are included because curricula may
  modify term weights/parameters during a run. Unknown object types are rejected.
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.references = {
      id(env): "environment",
      id(env.sim): "simulation",
      id(env.sim.wp_model): "warp_model",
      id(env.sim.wp_data): "warp_data",
      id(env.sim.model): "model_bridge",
      id(env.sim.data): "data_bridge",
    }
    context = getattr(getattr(env, "scene", None), "sensor_context", None)
    if context is not None:
      # Raycast-only BVH resources are freshly built from the same model and
      # refitted before every native sense() query. Never copy opaque handles.
      self.references[id(context)] = "raycast_context"
    self.nodes: dict[int, dict] = {}
    self.seen: dict[int, int] = {}
    self.tensor_storages: dict[tuple, int] = {}
    self.active_tuples: set[int] = set()

  def capture(self) -> dict:
    roots = {
      "environment": {
        k: v
        for k, v in vars(self.env).items()
        if k not in ("sim", "_offline_renderer", "manager_visualizers")
      },
      "simulation_baselines": {
        k: getattr(self.env.sim, k)
        for k in ("_default_model_fields", "_world_to_variant")
      },
    }
    self.references[id(self.env)] = "environment"
    root = self._capture(roots, "environment")
    return {"root": root, "nodes": self.nodes}

  def _capture(self, value: Any, path: str) -> dict:
    if value is None or type(value) in (str, int, float, bool):
      return {"kind": "value", "value": value}
    if isinstance(value, np.generic):
      return {"kind": "value", "value": value.item()}
    if id(value) in self.references:
      return {"kind": "reference", "name": self.references[id(value)]}
    if isinstance(value, (Enum, Path, slice, torch.device, torch.dtype)):
      return {"kind": "static", "identity": repr(value)}
    if inspect.isfunction(value) or isinstance(value, type):
      stateful_type = isinstance(value, type) and all(
        callable(getattr(value, name, None))
        for name in ("training_state_dict", "load_training_state_dict")
      )
      if not value.__module__.startswith("mjlab.") and not (
        getattr(value, "training_state_stateless", False) or stateful_type
      ):
        raise ValueError(
          f"{path}: custom callable must declare training_state_stateless"
        )
      return {"kind": "static", "identity": _configuration(value)}
    module = type(value).__module__
    if module.startswith(_STATIC_MODULES) and not isinstance(value, wp.array):
      return {"kind": "static", "identity": _type_name(value)}
    if id(value) in self.seen:
      if self.active_tuples:
        # Immutable tuple cycles cannot be rebuilt without replacing aliases in
        # already restored mutable nodes. Fail rather than silently change them.
        if id(value) in self.active_tuples:
          raise ValueError(f"{path}: cyclic tuples are unsupported for stateful resume")
      return {"kind": "node", "id": self.seen[id(value)]}
    index = len(self.nodes)
    self.seen[id(value)] = index
    node: dict[str, Any] = {"type": _type_name(value)}
    self.nodes[index] = node
    if isinstance(value, torch.Tensor):
      storage = value.untyped_storage()
      key = (str(value.device), storage.data_ptr(), storage.nbytes())
      if not storage.nbytes():
        key = (*key, id(value))
      storage_id = self.tensor_storages.setdefault(key, len(self.tensor_storages))
      node.update(
        kind="tensor",
        value=value.detach().cpu().clone(),
        device=str(value.device),
        storage=storage_id,
        stride=tuple(value.stride()),
        offset=value.storage_offset(),
        view=(
          not value.is_contiguous()
          or value.storage_offset() != 0
          or storage.nbytes() != value.numel() * value.element_size()
        ),
        storage_bytes=storage.nbytes(),
      )
    elif isinstance(value, np.ndarray):
      if value.dtype.hasobject:
        raise ValueError(f"{path}: object arrays cannot be checkpointed")
      node.update(kind="numpy", value=value.copy())
    elif isinstance(value, ctypes.Array):
      node.update(kind="ctypes_array", value=list(value))
    elif isinstance(value, torch.Generator):
      node.update(kind="generator", value=value.get_state().cpu())
    elif isinstance(value, wp.array):
      node.update(kind="warp", value=value.numpy().copy())
    elif isinstance(value, dict):
      node.update(
        kind="dict",
        items=[
          (self._key(key), self._capture(child, f"{path}[{key}]"))
          for key, child in value.items()
        ],
      )
    elif isinstance(value, (list, tuple, deque)):
      if isinstance(value, tuple):
        self.active_tuples.add(id(value))
      node.update(
        kind="sequence",
        items=[self._capture(child, f"{path}[{i}]") for i, child in enumerate(value)],
      )
      self.active_tuples.discard(id(value))
    elif callable(getattr(value, "training_state_dict", None)) and callable(
      getattr(value, "load_training_state_dict", None)
    ):
      node.update(kind="hook", state=self._capture(value.training_state_dict(), path))
    elif getattr(value, "training_state_stateless", False):
      node.update(kind="stateless")
    elif module.startswith(_BUILTIN_MODULES) or (
      is_dataclass(value) and not callable(value)
    ):
      node.update(
        kind="object",
        attrs={
          key: self._capture(child, f"{path}.{key}")
          for key, child in vars(value).items()
        },
      )
    else:
      raise ValueError(
        f"{path}: {_type_name(value)} does not support stateful resume. Implement "
        "training_state_dict/load_training_state_dict or declare "
        "training_state_stateless=True."
      )
    return {"kind": "node", "id": index}

  @staticmethod
  def _key(key: Any) -> tuple:
    if type(key) in (str, int, float, bool) or key is None:
      return ("value", key)
    if isinstance(key, tuple):
      return ("tuple", tuple(ComponentState._key(v) for v in key))
    return ("static", _function_name(key) if isinstance(key, type) else repr(key))

  @staticmethod
  def _decode_key(key: tuple) -> Any:
    if key[0] == "value":
      return key[1]
    if key[0] == "tuple":
      return tuple(ComponentState._decode_key(child) for child in key[1])
    raise ValueError("Stateful resume cannot reconstruct a new non-data mapping key")

  def restore(self, snapshot: dict, *, validate: bool = False) -> None:
    """Validate structure before mutation, then restore into the existing graph."""
    targets = {
      "environment": {
        k: v
        for k, v in vars(self.env).items()
        if k not in ("sim", "_offline_renderer", "manager_visualizers")
      },
      "simulation_baselines": {
        k: getattr(self.env.sim, k)
        for k in ("_default_model_fields", "_world_to_variant")
      },
    }
    memo: dict[int, Any] = {}
    storages: dict[int, tuple] = {}
    target_storages: dict[tuple, int] = {}
    storage_counts: dict[int, int] = {}
    for record in snapshot["nodes"].values():
      if record["kind"] == "tensor":
        storage_counts[record["storage"]] = storage_counts.get(record["storage"], 0) + 1

    def restore(node: dict, target: Any, path: str) -> Any:
      kind = node["kind"]
      if kind == "value":
        return node["value"]
      if kind in ("reference", "static"):
        current = self._capture(target, path)
        if current != node:
          raise ValueError(f"Stateful resume static component differs: {path}")
        return target
      index = node["id"]
      if index in memo:
        return memo[index]
      record = snapshot["nodes"][index]
      kind = record["kind"]
      if kind == "tensor":
        saved = record["value"]
        if isinstance(target, (bool, int, float, np.generic)):
          # Native Logger.log converts scalar diagnostics to owned tensors.
          target = None
        if target is not None and not isinstance(target, torch.Tensor):
          raise ValueError(f"Stateful resume tensor type differs: {path}")
        if target is None:
          if (
            record["view"] or record["offset"] or storage_counts[record["storage"]] > 1
          ):
            raise ValueError(
              f"{path}: lazy aliased/view tensors require an explicit training-state hook"
            )
          result = saved.to(record["device"]).clone()
        else:
          if target.dtype != saved.dtype or str(target.device) != record["device"]:
            raise ValueError(f"Stateful resume tensor shape/dtype differs: {path}")
          storage = target.untyped_storage()
          if storage.nbytes() != record["storage_bytes"]:
            raise ValueError(
              f"Stateful resume tensor backing-storage size differs: {path}"
            )
          identity = (str(target.device), storage.data_ptr(), storage.nbytes())
          group = record["storage"]
          if storage.nbytes() and group in storages and storages[group] != identity:
            raise ValueError(f"Stateful resume tensor storage aliases differ: {path}")
          if (
            storage.nbytes()
            and identity in target_storages
            and target_storages[identity] != group
          ):
            raise ValueError(f"Stateful resume tensor storage aliases differ: {path}")
          storages[group] = identity
          target_storages[identity] = group
          result = target
          if (
            target.shape != saved.shape
            or tuple(target.stride()) != record["stride"]
            or target.storage_offset() != record["offset"]
          ):
            # Native logging turns scalar aggregates into 1D views. Rebuild the
            # saved view on this allocation, never by resizing backing storage.
            result = target.as_strided(saved.shape, record["stride"], record["offset"])
          if not validate:
            result.copy_(saved)
      elif kind == "numpy":
        saved = record["value"]
        result = saved.copy()
        if target is not None:
          if (
            not isinstance(target, np.ndarray)
            or target.shape != saved.shape
            or target.dtype != saved.dtype
          ):
            raise ValueError(f"Stateful resume NumPy shape/dtype differs: {path}")
          result = target
          if not validate:
            result[...] = saved
      elif kind == "ctypes_array":
        if not isinstance(target, ctypes.Array) or len(target) != len(record["value"]):
          raise ValueError(f"Stateful resume ctypes array differs: {path}")
        result = target
        if not validate:
          result[:] = record["value"]
      elif kind == "generator":
        if not isinstance(target, torch.Generator):
          raise ValueError(f"Stateful resume generator differs: {path}")
        result = target
        if not validate:
          result.set_state(record["value"].cpu())
      elif kind == "warp":
        saved = record["value"]
        if not isinstance(target, wp.array):
          raise ValueError(f"Stateful resume Warp array differs: {path}")
        current = target.numpy()
        if current.shape != saved.shape or current.dtype != saved.dtype:
          raise ValueError(f"Stateful resume Warp array shape/dtype differs: {path}")
        result = target
        if not validate:
          result.assign(saved)
      elif kind == "dict":
        if target is None:
          target = {}
        if not isinstance(target, dict):
          raise ValueError(f"Stateful resume mapping differs: {path}")
        result = target
        memo[index] = result
        keys = {self._key(k): k for k in target}
        restored = {}
        for key, child in record["items"]:
          actual_key = keys[key] if key in keys else self._decode_key(key)
          restored[actual_key] = restore(
            child, target.get(actual_key), f"{path}[{actual_key}]"
          )
        if not validate:
          result.clear()
          result.update(restored)
      elif kind == "sequence":
        if target is None:
          target = () if record["type"] == "builtins.tuple" else []
        elif _type_name(target) != record["type"]:
          raise ValueError(f"Stateful resume sequence type differs: {path}")
        result = []
        memo[index] = target
        for i, child in enumerate(record["items"]):
          result.append(
            restore(child, target[i] if i < len(target) else None, f"{path}[{i}]")
          )
        if isinstance(target, tuple):
          result = tuple(result)
        elif isinstance(target, deque):
          if not validate:
            target.clear()
            target.extend(result)
          result = target
        elif not validate:
          target[:] = result
          result = target
        else:
          result = target
      elif kind in ("object", "hook", "stateless"):
        if _type_name(target) != record["type"]:
          raise ValueError(f"Stateful resume component type differs: {path}")
        result = target
        memo[index] = result
        if kind == "object":
          if vars(target).keys() != record["attrs"].keys():
            raise ValueError(f"Stateful resume component fields differ: {path}")
          for key, child in record["attrs"].items():
            value = restore(child, getattr(target, key), f"{path}.{key}")
            if not validate:
              object.__setattr__(target, key, value)
        elif kind == "hook":
          value = restore(record["state"], target.training_state_dict(), path)
          if not validate:
            target.load_training_state_dict(value)
      else:
        raise ValueError(f"Invalid stateful resume component kind: {kind}")
      memo[index] = result
      return result

    restored = restore(snapshot["root"], targets, "environment")
    if not validate:
      for key, value in restored["environment"].items():
        setattr(self.env, key, value)
      for key, value in restored["simulation_baselines"].items():
        setattr(self.env.sim, key, value)


def preflight(env: Any, cfg: dict, learner_device: str) -> dict:
  """Reject unsupported state sources before starting a stateful training run."""
  if type(env.env) is not ManagerBasedRlEnv:
    raise ValueError(
      "Stateful resume requires the native environment without extra wrappers"
    )
  native = env.unwrapped
  if version("rsl-rl-lib") != "5.5.1":
    raise ValueError("Stateful resume requires the audited RSL-RL 5.5.1 learn loop")
  if any(value is not None for value in vars(native.sim.wp_model.callback).values()):
    raise ValueError("Stateful resume does not yet support custom physics callbacks")
  if (
    native.scene.sensor_context is not None and native.scene.sensor_context.has_cameras
  ) or native.render_mode is not None:
    raise ValueError("Stateful resume does not yet support camera/render contexts")
  if native.recorder_manager.active_terms:
    raise ValueError("Stateful resume does not support external recorder side effects")
  ComponentState(native).capture()
  return compatibility(native, cfg, learner_device)
