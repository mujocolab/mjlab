"""Test viser conversion functions with various robot models."""

from pathlib import Path

import mujoco

from mjlab.viewer.viser import (
  create_site_mesh,
  get_geom_texture_id,
  group_geoms_by_visual_compat,
  merge_geoms,
  merge_sites,
  mujoco_mesh_to_trimesh,
)


def load_robot_model(robot_name: str) -> mujoco.MjModel:
  """Load a robot model from the asset zoo."""
  base_path = Path(__file__).parent.parent / "src/mjlab/asset_zoo/robots"

  # Map robot names to their XML files.
  robot_paths = {
    "unitree_g1": base_path / "unitree_g1/xmls/g1.xml",
    "unitree_go1": base_path / "unitree_go1/xmls/go1.xml",
  }

  if robot_name not in robot_paths:
    raise ValueError(f"Unknown robot: {robot_name}")

  xml_path = robot_paths[robot_name]
  if not xml_path.exists():
    raise FileNotFoundError(f"Robot XML not found: {xml_path}")

  return mujoco.MjModel.from_xml_path(str(xml_path))


def test_unitree_g1_conversion():
  """Test conversion with Unitree G1 robot."""
  model = load_robot_model("unitree_g1")

  mesh_geom_count = 0
  has_textures = False

  for geom_idx in range(model.ngeom):
    if model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_geom_count += 1

      # Convert to trimesh.
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      # Basic checks.
      assert mesh is not None, f"Failed to convert geom {geom_idx}"
      assert len(mesh.vertices) > 0, f"Mesh {geom_idx} has no vertices"
      assert len(mesh.faces) > 0, f"Mesh {geom_idx} has no faces"

      # Check for textures.
      if hasattr(mesh.visual, "uv"):
        has_textures = True

  assert mesh_geom_count > 0, "No mesh geometries found in Unitree G1"
  print(f"✓ Unitree G1: Successfully converted {mesh_geom_count} mesh geometries")
  if has_textures:
    print("  - Found textured meshes")


def test_unitree_go1_conversion():
  """Test conversion with Unitree Go1 robot."""
  model = load_robot_model("unitree_go1")

  mesh_geom_count = 0
  primitive_geom_count = 0

  for geom_idx in range(model.ngeom):
    geom_type = model.geom_type[geom_idx]

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_geom_count += 1

      # Convert to trimesh.
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      # Basic checks.
      assert mesh is not None, f"Failed to convert geom {geom_idx}"
      assert len(mesh.vertices) > 0, f"Mesh {geom_idx} has no vertices"
      assert len(mesh.faces) > 0, f"Mesh {geom_idx} has no faces"
    else:
      # Count primitive geometries (box, sphere, capsule, etc.).
      primitive_geom_count += 1

  print(f"✓ Unitree Go1: Successfully converted {mesh_geom_count} mesh geometries")
  print(f"  - Also has {primitive_geom_count} primitive geometries")


def test_texture_extraction():
  """Test texture extraction with a simple textured model."""
  # Create a model with procedural texture.
  xml_string = """
    <mujoco>
        <asset>
            <!-- Procedural checker texture -->
            <texture name="checker" type="2d" builtin="checker" width="64" height="64"
                     rgb1="1 0 0" rgb2="0 0 1"/>
            <material name="checker_mat" texture="checker" rgba="1 1 1 1"/>

            <!-- Simple box mesh with 8 vertices -->
            <mesh name="box"
                  vertex="0 0 0  1 0 0  1 1 0  0 1 0  0 0 1  1 0 1  1 1 1  0 1 1"
                  face="0 1 2  0 2 3  4 5 6  4 6 7"/>
        </asset>

        <worldbody>
            <geom name="textured_mesh" type="mesh" mesh="box" material="checker_mat"/>
        </worldbody>
    </mujoco>
    """

  model = mujoco.MjModel.from_xml_string(xml_string)

  # Find the mesh geom.
  for geom_idx in range(model.ngeom):
    if model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      assert mesh is not None, "Failed to convert textured mesh"
      assert mesh.visual is not None, "Mesh has no visual"

      # Check if texture was extracted (procedural textures should work).
      matid = model.geom_matid[geom_idx]
      if matid >= 0 and matid < model.nmat:
        texid = model.mat_texid[matid]
        # texid might be an array, get the first element if so.
        if hasattr(texid, "__len__"):
          texid = texid[0] if len(texid) > 0 else -1
        if texid >= 0:
          # Should have extracted the checker texture.
          print("✓ Texture extraction: Successfully extracted procedural texture")
          return

  print("✓ Texture extraction: Tested (no textured meshes in simple model)")


def test_material_colors():
  """Test that material colors are properly applied."""
  xml_string = """
    <mujoco>
        <asset>
            <material name="red" rgba="1 0 0 1"/>
            <material name="green" rgba="0 1 0 0.5"/>
            <!-- Tetrahedron with 4 vertices (minimum for MuJoCo mesh) -->
            <mesh name="tetra"
                  vertex="0 0 0  1 0 0  0.5 0.866 0  0.5 0.289 0.816"
                  face="0 1 2  0 1 3  0 2 3  1 2 3"/>
        </asset>

        <worldbody>
            <geom name="red_mesh" type="mesh" mesh="tetra" material="red"/>
            <geom name="green_mesh" type="mesh" mesh="tetra" material="green" pos="2 0 0"/>
            <geom name="default_mesh" type="mesh" mesh="tetra" pos="4 0 0"/>
        </worldbody>
    </mujoco>
    """

  model = mujoco.MjModel.from_xml_string(xml_string)

  colors_found = []
  for geom_idx in range(model.ngeom):
    if model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      # Check visual colors.
      visual = mesh.visual
      if visual and hasattr(visual, "vertex_colors"):
        # Get the first vertex color (they should all be the same).
        color = visual.vertex_colors[0]
        colors_found.append(tuple(color))

  assert len(colors_found) == 3, "Should have 3 meshes"

  # Check that we got different colors.
  assert colors_found[0][:3] == (255, 0, 0), "First mesh should be red"
  assert colors_found[1][:3] == (0, 255, 0), "Second mesh should be green"
  # Third mesh should have default color.

  print("✓ Material colors: Correctly applied to meshes")


def test_performance():
  """Test conversion performance with a complex model."""
  import time

  model = load_robot_model("unitree_g1")

  mesh_geoms = []
  for geom_idx in range(model.ngeom):
    if model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_geoms.append(geom_idx)

  start_time = time.time()
  for geom_idx in mesh_geoms:
    mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)
  elapsed = time.time() - start_time

  avg_time = elapsed / len(mesh_geoms) * 1000  # Convert to ms.
  print(f"✓ Performance: Converted {len(mesh_geoms)} meshes in {elapsed:.3f}s")
  print(f"  - Average: {avg_time:.2f}ms per mesh")

  # Warn if it's too slow.
  if avg_time > 50:
    print("  ⚠ Warning: Conversion is slow (>50ms per mesh)")


def test_verbose_mode():
  """Test that verbose mode produces output."""
  xml_string = """
    <mujoco>
        <asset>
            <material name="test" rgba="1 0 0 1"/>
            <!-- Tetrahedron with 4 vertices -->
            <mesh name="tetra"
                  vertex="0 0 0  1 0 0  0.5 0.866 0  0.5 0.289 0.816"
                  face="0 1 2  0 1 3  0 2 3  1 2 3"/>
        </asset>
        <worldbody>
            <geom type="mesh" mesh="tetra" material="test"/>
        </worldbody>
    </mujoco>
    """

  model = mujoco.MjModel.from_xml_string(xml_string)

  # Test with verbose=True.
  import io
  import sys

  captured_output = io.StringIO()
  sys.stdout = captured_output

  mujoco_mesh_to_trimesh(model, 0, verbose=True)

  sys.stdout = sys.__stdout__
  output = captured_output.getvalue()

  # Should have printed something.
  assert len(output) > 0, "Verbose mode should produce output"
  assert "vertices" in output or "color" in output, "Should mention vertices or color"

  print("✓ Verbose mode: Produces debug output when enabled")


def test_mesh_with_texture_coordinates():
  """Test meshes with texture coordinates (Issue #225)."""
  xml_string = """
    <mujoco>
        <asset>
            <!-- Simple tetrahedron WITH texture coordinates -->
            <mesh name="textured_tetra"
                  vertex="0 0 0  1 0 0  0.5 0.866 0  0.5 0.289 0.816"
                  texcoord="0 0  1 0  0.5 1  0.5 0.5"
                  face="0 1 2  0 1 3  0 2 3  1 2 3"/>
        </asset>
        <worldbody>
            <geom type="mesh" mesh="textured_tetra"/>
        </worldbody>
    </mujoco>
    """

  model = mujoco.MjModel.from_xml_string(xml_string)

  # Verify the model has texture coordinates.
  assert model.mesh_texcoord.shape == (4, 2), "Model should have 4 texture coordinates"
  assert model.mesh_texcoordnum[0] == 4, "Mesh should have 4 texture coordinates"

  # Convert the mesh.
  mesh = mujoco_mesh_to_trimesh(model, 0, verbose=False)

  # Verify the mesh was created successfully.
  assert mesh is not None, "Failed to convert textured mesh"
  assert len(mesh.vertices) > 0, "Mesh has no vertices"
  assert len(mesh.faces) > 0, "Mesh has no faces"

  # With texture coordinates, vertices are duplicated per face.
  # 4 faces * 3 vertices per face = 12 vertices.
  assert len(mesh.vertices) == 12, (
    f"Expected 12 duplicated vertices, got {len(mesh.vertices)}"
  )
  assert len(mesh.faces) == 4, f"Expected 4 faces, got {len(mesh.faces)}"

  print(
    "✓ Mesh with texture coordinates: Successfully converted mesh with texcoords (Issue #225 fixed)"
  )


def _make_mixed_visual_model() -> mujoco.MjModel:
  """Create a model with both textured and color-only geoms on the same body."""
  xml = """
    <mujoco>
      <asset>
        <texture name="checker" type="2d" builtin="checker" width="64" height="64"
                 rgb1="1 0 0" rgb2="0 0 1"/>
        <material name="tex_mat" texture="checker"/>
        <material name="color_mat" rgba="0 1 0 1"/>
        <mesh name="textured_tetra"
              vertex="0 0 0  1 0 0  0.5 0.866 0  0.5 0.289 0.816"
              texcoord="0 0  1 0  0.5 1  0.5 0.5"
              face="0 1 2  0 1 3  0 2 3  1 2 3"/>
        <mesh name="plain_tetra"
              vertex="2 0 0  3 0 0  2.5 0.866 0  2.5 0.289 0.816"
              face="0 1 2  0 1 3  0 2 3  1 2 3"/>
      </asset>
      <worldbody>
        <body name="mixed">
          <geom name="tex_geom" type="mesh" mesh="textured_tetra" material="tex_mat"/>
          <geom name="color_geom" type="mesh" mesh="plain_tetra" material="color_mat"/>
        </body>
      </worldbody>
    </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


def test_get_geom_texture_id():
  """Verify correct texture ID detection for textured vs untextured geoms."""
  model = _make_mixed_visual_model()

  # Find geom indices by name.
  tex_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tex_geom")
  color_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "color_geom")

  tex_id = get_geom_texture_id(model, tex_geom_id)
  assert tex_id >= 0, "Textured geom should have a non-negative texture ID"

  color_id = get_geom_texture_id(model, color_geom_id)
  assert color_id == -1, "Color-only geom should have texture ID -1"


def test_get_geom_texture_id_primitive():
  """Primitive (non-mesh) geoms should always return -1."""
  xml = """
    <mujoco>
      <worldbody>
        <geom name="sphere" type="sphere" size="0.1"/>
      </worldbody>
    </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
  assert get_geom_texture_id(model, geom_id) == -1


def test_group_geoms_by_visual_compat():
  """Verify geoms are correctly partitioned by visual compatibility."""
  model = _make_mixed_visual_model()

  tex_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tex_geom")
  color_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "color_geom")

  groups = group_geoms_by_visual_compat(model, [tex_geom_id, color_geom_id])

  # Should produce two groups: one textured, one untextured.
  assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"

  # Each group should have exactly one geom.
  assert all(len(g) == 1 for g in groups)

  # Flatten to a set and verify all geoms are present.
  all_ids = {gid for g in groups for gid in g}
  assert all_ids == {tex_geom_id, color_geom_id}


def test_group_geoms_single_type():
  """All untextured geoms should land in a single group."""
  xml = """
    <mujoco>
      <asset>
        <material name="red" rgba="1 0 0 1"/>
        <material name="blue" rgba="0 0 1 1"/>
        <mesh name="t1"
              vertex="0 0 0  1 0 0  0.5 0.866 0  0.5 0.289 0.816"
              face="0 1 2  0 1 3  0 2 3  1 2 3"/>
        <mesh name="t2"
              vertex="2 0 0  3 0 0  2.5 0.866 0  2.5 0.289 0.816"
              face="0 1 2  0 1 3  0 2 3  1 2 3"/>
      </asset>
      <worldbody>
        <body name="b">
          <geom name="g1" type="mesh" mesh="t1" material="red"/>
          <geom name="g2" type="mesh" mesh="t2" material="blue"/>
        </body>
      </worldbody>
    </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g1")
  g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g2")

  groups = group_geoms_by_visual_compat(model, [g1, g2])
  assert len(groups) == 1, "Two untextured geoms should form a single group"
  assert set(groups[0]) == {g1, g2}


def test_merge_mixed_visual_geoms():
  """Regression: merging each visual-compat subgroup must preserve visuals."""
  import trimesh.visual

  model = _make_mixed_visual_model()

  tex_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tex_geom")
  color_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "color_geom")

  groups = group_geoms_by_visual_compat(model, [tex_geom_id, color_geom_id])

  for group in groups:
    mesh = merge_geoms(model, group)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0

    tex_id = get_geom_texture_id(model, group[0])
    if tex_id >= 0:
      # Textured group should have TextureVisuals, not default gray.
      assert isinstance(mesh.visual, trimesh.visual.TextureVisuals), (
        f"Expected TextureVisuals for textured group, got {type(mesh.visual)}"
      )
    else:
      # Color group should have ColorVisuals.
      assert isinstance(mesh.visual, trimesh.visual.ColorVisuals), (
        f"Expected ColorVisuals for color group, got {type(mesh.visual)}"
      )
      # Verify the color is green (not default gray).
      vertex_colors = mesh.visual.vertex_colors
      assert vertex_colors[0][1] == 255, (
        f"Expected green channel 255, got {vertex_colors[0][1]}"
      )


def _make_site_model() -> mujoco.MjModel:
  """Create a model with all 5 site types."""
  xml = """
    <mujoco>
      <worldbody>
        <body name="test_body" pos="0 0 1">
          <joint type="free"/>
          <geom type="sphere" size="0.01"/>
          <site name="sphere_site" type="sphere" size="0.1" rgba="1 0 0 1"/>
          <site name="capsule_site" type="capsule" size="0.05 0.1" rgba="0 1 0 1"
                pos="1 0 0"/>
          <site name="ellipsoid_site" type="ellipsoid" size="0.1 0.05 0.03"
                rgba="0 0 1 1" pos="2 0 0"/>
          <site name="cylinder_site" type="cylinder" size="0.05 0.1" rgba="1 1 0 1"
                pos="3 0 0"/>
          <site name="box_site" type="box" size="0.1 0.05 0.03" rgba="0 1 1 1"
                pos="4 0 0"/>
        </body>
      </worldbody>
    </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


def test_create_site_mesh_sphere():
  """Test sphere site mesh creation."""
  model = _make_site_model()
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "sphere_site")
  mesh = create_site_mesh(model, site_id)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  # Check color is red.
  assert mesh.visual is not None
  assert mesh.visual.vertex_colors[0][0] == 255
  assert mesh.visual.vertex_colors[0][1] == 0


def test_create_site_mesh_capsule():
  """Test capsule site mesh creation."""
  model = _make_site_model()
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "capsule_site")
  mesh = create_site_mesh(model, site_id)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  # Check color is green.
  assert mesh.visual is not None
  assert mesh.visual.vertex_colors[0][1] == 255


def test_create_site_mesh_ellipsoid():
  """Test ellipsoid site mesh creation."""
  model = _make_site_model()
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ellipsoid_site")
  mesh = create_site_mesh(model, site_id)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  # Check color is blue.
  assert mesh.visual is not None
  assert mesh.visual.vertex_colors[0][2] == 255


def test_create_site_mesh_cylinder():
  """Test cylinder site mesh creation."""
  model = _make_site_model()
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cylinder_site")
  mesh = create_site_mesh(model, site_id)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0


def test_create_site_mesh_box():
  """Test box site mesh creation."""
  model = _make_site_model()
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "box_site")
  mesh = create_site_mesh(model, site_id)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0


def test_create_site_mesh_rgba_fallback():
  """Test that all-zero RGBA falls back to gray."""
  xml = """
    <mujoco>
      <worldbody>
        <site name="zero_rgba" type="sphere" size="0.1" rgba="0 0 0 0"/>
      </worldbody>
    </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "zero_rgba")
  mesh = create_site_mesh(model, site_id)
  # Should be gray (0.5 * 255 = 127), not transparent black.
  assert mesh.visual is not None
  color = mesh.visual.vertex_colors[0]
  assert color[0] == 127, f"Expected R=127, got {color[0]}"
  assert color[1] == 127, f"Expected G=127, got {color[1]}"
  assert color[2] == 127, f"Expected B=127, got {color[2]}"
  assert color[3] == 255, f"Expected A=255, got {color[3]}"


def test_merge_sites_single():
  """Test merging a single site returns the site mesh."""
  model = _make_site_model()
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "sphere_site")
  mesh = merge_sites(model, [site_id])
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0


def test_merge_sites_multiple():
  """Test merging multiple sites combines them correctly."""
  model = _make_site_model()
  sphere_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "sphere_site")
  capsule_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "capsule_site")
  box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "box_site")

  single_sphere = create_site_mesh(model, sphere_id)
  single_capsule = create_site_mesh(model, capsule_id)
  single_box = create_site_mesh(model, box_id)

  merged = merge_sites(model, [sphere_id, capsule_id, box_id])

  # Merged mesh should have at least the sum of individual vertices.
  expected_min_verts = (
    len(single_sphere.vertices)
    + len(single_capsule.vertices)
    + len(single_box.vertices)
  )
  assert len(merged.vertices) >= expected_min_verts, (
    f"Expected >= {expected_min_verts} vertices, got {len(merged.vertices)}"
  )
  assert len(merged.faces) > 0


if __name__ == "__main__":
  # Run all tests
  print("=" * 60)
  print("Testing Viser Conversions")
  print("=" * 60)

  tests = [
    test_unitree_g1_conversion,
    test_unitree_go1_conversion,
    test_texture_extraction,
    test_material_colors,
    test_performance,
    test_verbose_mode,
    test_mesh_with_texture_coordinates,
    test_get_geom_texture_id,
    test_get_geom_texture_id_primitive,
    test_group_geoms_by_visual_compat,
    test_group_geoms_single_type,
    test_merge_mixed_visual_geoms,
  ]

  failed = []
  for test in tests:
    try:
      test()
    except Exception as e:
      test_name = test.__name__ if hasattr(test, "__name__") else str(test)
      print(f"✗ {test_name}: {e}")
      import traceback

      traceback.print_exc()
      failed.append(test_name)

  print("\n" + "=" * 60)
  if failed:
    print(f"Failed tests: {', '.join(failed)}")
    import sys

    sys.exit(1)
  else:
    print("All tests passed!")
    import sys

    sys.exit(0)
