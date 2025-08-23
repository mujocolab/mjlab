from __future__ import annotations

from mjlab.third_party.isaaclab.isaaclab_tasks.utils.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
