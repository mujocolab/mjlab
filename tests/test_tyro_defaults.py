import tyro

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.scripts.train import TrainConfig
from mjlab.terrains import TerrainEntityCfg


def test_tyro_defaults():
  """
  Test for regression of https://github.com/mujocolab/mjlab/issues/677 due to incompatibilities between the declared types and default values of the env configs.
  """
  tyro.cli(
    TrainConfig,
    default=TrainConfig(
      env=ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
          terrain=TerrainEntityCfg(),
        ),
      ),
      agent=RslRlOnPolicyRunnerCfg(),
    ),
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
    args=[],
  )
