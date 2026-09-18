"""
Roll out open-loop trajectories from initial states, get subsequent states and sensor values.
"""
from __future__ import annotations
import atexit as atexit
from collections.abc import Sequence
import mujoco as mujoco
from mujoco import _rollout
import numpy as np
from numpy import typing as npt
__all__: list[str] = ['Rollout', 'Sequence', 'atexit', 'mujoco', 'np', 'npt', 'persistent_rollout', 'rollout', 'shutdown_persistent_pool']
class Rollout:
    """
    Rollout object containing a thread pool for parallel rollouts.
    """
    def __enter__(self):
        ...
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
    def __init__(self, *, nthread: typing.Optional[int] = None):
        """
        Construct a rollout object containing a thread pool for parallel rollouts.
        
        Args:
          nthread: Number of threads in pool.
            If zero, this pool is not started and rollouts run on the calling thread.
        """
    def close(self):
        ...
    def rollout(self, model: typing.Union[mujoco._structs.MjModel, collections.abc.Sequence[mujoco._structs.MjModel]], data: typing.Union[mujoco._structs.MjData, collections.abc.Sequence[mujoco._structs.MjData]], initial_state: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str]], control: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, *, control_spec: int = 64, skip_checks: bool = False, nstep: typing.Optional[int] = None, initial_warmstart: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, state: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, sensordata: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, chunk_size: typing.Optional[int] = None):
        """
        Rolls out open-loop trajectories from initial states, get subsequent state and sensor values.
        
        Python wrapper for rollout.cc, see documentation therein.
        Infers nbatch and nstep.
        Tiles inputs with singleton dimensions.
        Allocates outputs if none are given.
        
        Args:
          model: An instance or length nbatch sequence of MjModel with the same size signature.
          data: Associated mjData instance or sequence of instances with length nthread.
          initial_state: Array of initial states from which to roll out trajectories.
            ([nbatch or 1] x nstate)
          control: Open-loop controls array to apply during the rollouts.
            ([nbatch or 1] x [nstep or 1] x ncontrol)
          control_spec: mjtState specification of control vectors.
          skip_checks: Whether to skip internal shape and type checks.
          nstep: Number of steps in rollouts (inferred if unspecified).
          initial_warmstart: Initial qfrc_warmstart array (optional).
            ([nbatch or 1] x nv)
          state: State output array (optional).
            (nbatch x nstep x nstate)
          sensordata: Sensor data output array (optional).
            (nbatch x nstep x nsensordata)
          chunk_size: Determines threadpool chunk size. If unspecified,
                      chunk_size = max(1, nbatch / (nthread * 10))
        
        Returns:
          state:
            State output array, (nbatch x nstep x nstate).
          sensordata:
            Sensor data output array, (nbatch x nstep x nsensordata).
        
        Raises:
          RuntimeError: rollout requested after thread pool shutdown.
          ValueError: bad shapes or sizes.
        """
def _check_must_be_numeric(**kwargs):
    ...
def _check_number_of_dimensions(ndim, **kwargs):
    ...
def _check_trailing_dimension(dim, **kwargs):
    ...
def _ensure_2d(arg):
    ...
def _ensure_3d(arg):
    ...
def _infer_dimension(dim, value, **kwargs):
    """
    Infers dimension `dim` given guess `value` from set of arrays.
    
    Args:
      dim: Dimension to be inferred.
      value: Initial guess of inferred value (1: unknown).
      **kwargs: List of arrays which should all have the same size (or 1) along
        dimension dim.
    
    Returns:
      Inferred dimension.
    
    Raises:
      ValueError: If mismatch between array shapes or initial guess.
    """
def _tile_if_required(array, dim0, dim1 = None):
    ...
def rollout(model: typing.Union[mujoco._structs.MjModel, collections.abc.Sequence[mujoco._structs.MjModel]], data: typing.Union[mujoco._structs.MjData, collections.abc.Sequence[mujoco._structs.MjData]], initial_state: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str]], control: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, *, control_spec: int = 64, skip_checks: bool = False, nstep: typing.Optional[int] = None, initial_warmstart: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, state: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, sensordata: typing.Union[collections.abc.Buffer, numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], complex, bytes, str, numpy._typing._nested_sequence._NestedSequence[complex | bytes | str], NoneType] = None, chunk_size: typing.Optional[int] = None, persistent_pool: bool = False):
    """
    Rolls out open-loop trajectories from initial states, get subsequent states and sensor values.
    
    Python wrapper for rollout.cc, see documentation therein.
    Infers nbatch and nstep.
    Tiles inputs with singleton dimensions.
    Allocates outputs if none are given.
    
    Args:
      model: An instance or length nbatch sequence of MjModel with the same size signature.
      data: Associated mjData instance or sequence of instances with length nthread.
      initial_state: Array of initial states from which to roll out trajectories.
        ([nbatch or 1] x nstate)
      control: Open-loop controls array to apply during the rollouts.
        ([nbatch or 1] x [nstep or 1] x ncontrol)
      control_spec: mjtState specification of control vectors.
      skip_checks: Whether to skip internal shape and type checks.
      nstep: Number of steps in rollouts (inferred if unspecified).
      initial_warmstart: Initial qfrc_warmstart array (optional).
        ([nbatch or 1] x nv)
      state: State output array (optional).
        (nbatch x nstep x nstate)
      sensordata: Sensor data output array (optional).
        (nbatch x nstep x nsensordata)
      chunk_size: Determines threadpool chunk size. If unspecified,
                  chunk_size = max(1, nbatch / (nthread * 10))
      persistent_pool: Determines if a persistent thread pool is created or reused.
    
    Returns:
      state:
        State output array, (nbatch x nstep x nstate).
      sensordata:
        Sensor data output array, (nbatch x nstep x nsensordata).
    
    Raises:
      ValueError: bad shapes or sizes.
    """
def shutdown_persistent_pool():
    """
    Shutdown the persistent thread pool that is optionally created by rollout.
    
    This is called automatically interpreter shutdown, but can also be called manually.
    """
persistent_rollout = None
