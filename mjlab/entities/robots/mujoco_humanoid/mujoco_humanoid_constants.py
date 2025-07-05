"""MuJoCo humanoid constants."""

# fmt: off

from mjlab import MJLAB_SRC_PATH

##
# MJCF and assets.
##

RAGDOLL_XML = MJLAB_SRC_PATH / "entities" / "robots" / "mujoco_humanoid" / "xmls" / "humanoid.xml"

##
# Constants.
##

NU = 3 + 6*2 + 3 * 2
NQ = NU + 7
NV = NU + 6

TORSO_BODY = "torso"
HEAD_BODY = "head"
