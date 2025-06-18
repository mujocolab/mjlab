from typing import Dict
from mjlab.entities.go1.go1 import UnitreeGo1
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

GO1_XML = MJLAB_SRC_PATH / "entities" / "go1" / "xmls" / "go1.xml"


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "unitree_go1"
  update_assets(assets, path, "*.xml")
  update_assets(assets, path / "assets")
  return assets


__all__ = (
  "UnitreeGo1",
  "GO1_XML",
  "get_assets",
)
