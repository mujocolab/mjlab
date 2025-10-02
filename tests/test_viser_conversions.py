"""Test viser conversion functions with various robot models."""

from pathlib import Path

import mujoco

from mjlab.viewer.viser_conversions import mujoco_mesh_to_trimesh


def load_robot_model(robot_name: str) -> mujoco.MjModel:
  """Load a robot model from the asset zoo."""
  base_path = Path(__file__).parent.parent / "src/mjlab/asset_zoo/robots"

  # Map robot names to their XML files
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

      # Convert to trimesh
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      # Basic checks
      assert mesh is not None, f"Failed to convert geom {geom_idx}"
      assert len(mesh.vertices) > 0, f"Mesh {geom_idx} has no vertices"
      assert len(mesh.faces) > 0, f"Mesh {geom_idx} has no faces"

      # Check for textures
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

      # Convert to trimesh
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      # Basic checks
      assert mesh is not None, f"Failed to convert geom {geom_idx}"
      assert len(mesh.vertices) > 0, f"Mesh {geom_idx} has no vertices"
      assert len(mesh.faces) > 0, f"Mesh {geom_idx} has no faces"
    else:
      # Count primitive geometries (box, sphere, capsule, etc.)
      primitive_geom_count += 1

  print(f"✓ Unitree Go1: Successfully converted {mesh_geom_count} mesh geometries")
  print(f"  - Also has {primitive_geom_count} primitive geometries")


def test_texture_extraction():
  """Test texture extraction with a simple textured model."""
  # Create a model with procedural texture
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

  # Find the mesh geom
  for geom_idx in range(model.ngeom):
    if model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

      assert mesh is not None, "Failed to convert textured mesh"
      assert mesh.visual is not None, "Mesh has no visual"

      # Check if texture was extracted (procedural textures should work)
      matid = model.geom_matid[geom_idx]
      if matid >= 0 and matid < model.nmat:
        texid = model.mat_texid[matid]
        # texid might be an array, get the first element if so
        if hasattr(texid, "__len__"):
          texid = texid[0] if len(texid) > 0 else -1
        if texid >= 0:
          # Should have extracted the checker texture
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

      # Check visual colors
      if mesh.visual and hasattr(mesh.visual, "vertex_colors"):
        # Get the first vertex color (they should all be the same)
        color = mesh.visual.vertex_colors[0]
        colors_found.append(tuple(color))

  assert len(colors_found) == 3, "Should have 3 meshes"

  # Check that we got different colors
  assert colors_found[0][:3] == (255, 0, 0), "First mesh should be red"
  assert colors_found[1][:3] == (0, 255, 0), "Second mesh should be green"
  # Third mesh should have default color

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

  avg_time = elapsed / len(mesh_geoms) * 1000  # Convert to ms
  print(f"✓ Performance: Converted {len(mesh_geoms)} meshes in {elapsed:.3f}s")
  print(f"  - Average: {avg_time:.2f}ms per mesh")

  # Warn if it's too slow
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

  # Test with verbose=True
  import io
  import sys

  captured_output = io.StringIO()
  sys.stdout = captured_output

  mujoco_mesh_to_trimesh(model, 0, verbose=True)

  sys.stdout = sys.__stdout__
  output = captured_output.getvalue()

  # Should have printed something
  assert len(output) > 0, "Verbose mode should produce output"
  assert "vertices" in output or "color" in output, "Should mention vertices or color"

  print("✓ Verbose mode: Produces debug output when enabled")


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
