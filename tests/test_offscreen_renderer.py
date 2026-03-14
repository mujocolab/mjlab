from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mjlab.viewer.offscreen_renderer import _copy_per_env_render_fields


def test_copy_per_env_render_fields_updates_geom_dataid() -> None:
  mj_model = SimpleNamespace(geom_dataid=np.array([10, 11, 12], dtype=np.int32))
  batched_model = SimpleNamespace(
    geom_dataid=torch.tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dtype=torch.int32,
    )
  )

  _copy_per_env_render_fields(mj_model, batched_model, env_idx=1)

  assert np.array_equal(mj_model.geom_dataid, np.array([4, 5, 6], dtype=np.int32))


def test_copy_per_env_render_fields_noop_without_batched_field() -> None:
  mj_model = SimpleNamespace(geom_dataid=np.array([7, 8], dtype=np.int32))

  _copy_per_env_render_fields(mj_model, batched_model=None, env_idx=0)

  assert np.array_equal(mj_model.geom_dataid, np.array([7, 8], dtype=np.int32))
