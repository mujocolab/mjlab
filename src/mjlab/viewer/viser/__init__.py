"""Viser web-based viewer implementation."""

from mjviser.conversions import create_primitive_mesh as create_primitive_mesh
from mjviser.conversions import create_site_mesh as create_site_mesh
from mjviser.conversions import get_body_name as get_body_name
from mjviser.conversions import get_geom_texture_id as get_geom_texture_id
from mjviser.conversions import get_site_name as get_site_name
from mjviser.conversions import (
  group_geoms_by_visual_compat as group_geoms_by_visual_compat,
)
from mjviser.conversions import is_fixed_body as is_fixed_body
from mjviser.conversions import merge_geoms as merge_geoms
from mjviser.conversions import merge_sites as merge_sites
from mjviser.conversions import mujoco_mesh_to_trimesh as mujoco_mesh_to_trimesh
from mjviser.conversions import (
  rotation_matrix_from_vectors as rotation_matrix_from_vectors,
)

from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel as RewardBarPanel
from mjlab.viewer.viser.scene import MjlabViserScene as MjlabViserScene
from mjlab.viewer.viser.term_plotter import ViserTermPlotter as ViserTermPlotter
from mjlab.viewer.viser.viewer import ViserPlayViewer as ViserPlayViewer

# Backwards compatibility alias.
ViserMujocoScene = MjlabViserScene
