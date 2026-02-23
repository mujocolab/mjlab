"""Domain randomization functions."""

# Types and built-in instances.
# isort: split
from ._types import Distribution as Distribution
from ._types import Operation as Operation
from ._types import abs as abs
from ._types import add as add
from ._types import gaussian as gaussian
from ._types import log_uniform as log_uniform
from ._types import scale as scale
from ._types import uniform as uniform

# Geom.
# isort: split
from .geom import geom_friction as geom_friction
from .geom import geom_pos as geom_pos
from .geom import geom_quat as geom_quat
from .geom import geom_rgba as geom_rgba
from .geom import geom_size as geom_size

# Body.
# isort: split
from .body import body_com_offset as body_com_offset
from .body import body_ipos as body_ipos
from .body import body_mass as body_mass
from .body import body_pos as body_pos
from .body import body_quat as body_quat
from .body import pseudo_inertia as pseudo_inertia

# Joint / DOF.
# isort: split
from .joint import dof_armature as dof_armature
from .joint import dof_damping as dof_damping
from .joint import dof_frictionloss as dof_frictionloss
from .joint import encoder_bias as encoder_bias
from .joint import jnt_range as jnt_range
from .joint import jnt_stiffness as jnt_stiffness
from .joint import joint_armature as joint_armature
from .joint import joint_damping as joint_damping
from .joint import joint_default_pos as joint_default_pos
from .joint import joint_friction as joint_friction
from .joint import joint_limits as joint_limits
from .joint import joint_stiffness as joint_stiffness
from .joint import qpos0 as qpos0

# Site.
# isort: split
from .site import site_pos as site_pos
from .site import site_quat as site_quat

# Tendon.
# isort: split
from .tendon import tendon_armature as tendon_armature
from .tendon import tendon_damping as tendon_damping
from .tendon import tendon_friction as tendon_friction
from .tendon import tendon_frictionloss as tendon_frictionloss
from .tendon import tendon_stiffness as tendon_stiffness

# Camera.
# isort: split
from .camera import cam_fovy as cam_fovy
from .camera import cam_intrinsic as cam_intrinsic
from .camera import cam_pos as cam_pos
from .camera import cam_quat as cam_quat

# Light.
# isort: split
from .light import light_dir as light_dir
from .light import light_pos as light_pos

# Actuator.
# isort: split
from .actuator import effort_limits as effort_limits
from .actuator import pd_gains as pd_gains
from .actuator import sync_actuator_delays as sync_actuator_delays
