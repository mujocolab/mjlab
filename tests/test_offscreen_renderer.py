"""Tests for offscreen renderer environment selection."""

import numpy as np
import pytest

from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def test_env_ids_clamps_when_fewer_envs_than_requested():
  # Default max_extra_envs=2 with a single env must not raise (csv_to_npz and
  # single-env video recording rely on this).
  cfg = ViewerConfig(max_extra_envs=2)
  ids = OffscreenRenderer._get_env_ids(cfg, np.zeros((1, 3)))
  assert ids == (0,)


def test_env_ids_primary_is_env_idx_despite_identical_origins():
  # With identical origins, distance ties must not evict env_idx from the set.
  cfg = ViewerConfig(env_idx=3, max_extra_envs=2)
  ids = OffscreenRenderer._get_env_ids(cfg, np.zeros((8, 3)))
  assert ids[0] == 3
  assert len(ids) == 3
  assert len(set(ids)) == 3


def test_env_ids_selects_nearest_neighbors():
  cfg = ViewerConfig(env_idx=0, max_extra_envs=2)
  origins = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
  )
  ids = OffscreenRenderer._get_env_ids(cfg, origins)
  assert ids == (0, 2, 3)


def test_env_ids_out_of_range_env_idx_is_clamped():
  cfg = ViewerConfig(env_idx=99, max_extra_envs=0)
  ids = OffscreenRenderer._get_env_ids(cfg, np.zeros((4, 3)))
  assert ids == (3,)


def test_env_ids_negative_max_extra_envs_raises():
  cfg = ViewerConfig(max_extra_envs=-1)
  with pytest.raises(ValueError, match="max_extra_envs"):
    OffscreenRenderer._get_env_ids(cfg, np.zeros((4, 3)))
