from mjlab.terrains.heightfield_terrains import (
  HfDiscreteObstaclesTerrainCfg as HfDiscreteObstaclesTerrainCfg,
)
from mjlab.terrains.heightfield_terrains import (
  HfPerlinNoiseTerrainCfg as HfPerlinNoiseTerrainCfg,
)
from mjlab.terrains.heightfield_terrains import (
  HfPyramidSlopedTerrainCfg as HfPyramidSlopedTerrainCfg,
)
from mjlab.terrains.heightfield_terrains import (
  HfRandomUniformTerrainCfg as HfRandomUniformTerrainCfg,
)
from mjlab.terrains.heightfield_terrains import HfWaveTerrainCfg as HfWaveTerrainCfg
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg as BoxFlatTerrainCfg
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg as BoxInvertedPyramidStairsTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxNarrowBeamsTerrainCfg as BoxNarrowBeamsTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxNestedRingsTerrainCfg as BoxNestedRingsTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxOpenStairsTerrainCfg as BoxOpenStairsTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxPyramidStairsTerrainCfg as BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxRandomGridTerrainCfg as BoxRandomGridTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxRandomSpreadTerrainCfg as BoxRandomSpreadTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxRandomStairsTerrainCfg as BoxRandomStairsTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxSteppingStonesTerrainCfg as BoxSteppingStonesTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxTiltedGridTerrainCfg as BoxTiltedGridTerrainCfg,
)
from mjlab.terrains.terrain_entity import TerrainEntity as TerrainEntity
from mjlab.terrains.terrain_entity import TerrainEntityCfg as TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg as FlatPatchSamplingCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg as SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGenerator as TerrainGenerator
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg as TerrainGeneratorCfg

# TODO(kevin): remove these aliases, see https://github.com/mujocolab/mjlab/issues/667
# Backwards compatibility aliases (deprecated).
_DEPRECATED_ALIASES = {
  "TerrainImporter": "TerrainEntity",
  "TerrainImporterCfg": "TerrainEntityCfg",
}


def __getattr__(name: str):
  if name in _DEPRECATED_ALIASES:
    import warnings

    new_name = _DEPRECATED_ALIASES[name]
    warnings.warn(
      f"{name} is deprecated and will be removed in a future version. Use {new_name} instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    return globals()[new_name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
