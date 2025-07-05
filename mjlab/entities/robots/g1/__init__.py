from typing import Dict
from mjlab.entities.g1.g1 import UnitreeG1
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

G1_XML = MJLAB_SRC_PATH / "entities" / "g1" / "xmls" / "g1.xml"


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "unitree_g1"
  update_assets(assets, path / "assets")
  return assets


__all__ = (
  "UnitreeG1",
  "G1_XML",
  "get_assets",
)
