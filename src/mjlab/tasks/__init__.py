from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", "velocity_football_env_cfg"]

import_packages(__name__, _BLACKLIST_PKGS)
