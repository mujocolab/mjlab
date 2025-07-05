from typing import Dict
from mjlab.entities.t1.t1 import BoosterT1
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

T1_XML = MJLAB_SRC_PATH / "entities" / "t1" / "xmls" / "t1.xml"


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "booster_t1"
  update_assets(assets, path / "assets")
  return assets


__all__ = (
  "BoosterT1",
  "T1_XML",
  "get_assets",
)
