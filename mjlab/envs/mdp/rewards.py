"""Useful methods for MPD rewards."""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

"""
joint_torques_l2(
  env=env,
  # Penalize all joints.
  asset_cfg=AssetCfg(joint_names=[".*"]),
)

joint_torques_l2(
  env=env,
  # Penalize all joints whose name starts with g1/.
  # Note the scene attachment process prefixes entities
  # with a name so for example if I attach a g1 to
  # the scene and specify its name as g1, then all
  # elements of the g1 such as its joints will start
  # with `g1/`.
  asset_cfg=AssetCfg(joint_names=[r"^g1/.*"])
)

# They might not all have names but I can specify it from
# an entity.
joint_torques_l2(
  env=env,
  asset_cfg=AssetCfg(name="g1")
)
"""


def joint_torques_l2(
  env, entity_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  # entity = env.scene[asset_cfg.name]
  arr = env.data.qfrc_actuator[entity_cfg.dof_ids]
  torques = torch.from_numpy(arr)[None]
  return torch.sum(torch.square(torques), dim=-1)
