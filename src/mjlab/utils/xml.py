"""XML fixup utilities for MjSpec.to_xml() output."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco


def strip_buffer_textures(spec: mujoco.MjSpec) -> None:
  """Remove buffer textures and their materials from a spec.

  MuJoCo's ``to_xml()`` cannot serialize textures created from raw pixel data
  (``texture.data = ...``). This function deletes those textures and the materials that
  reference them, and clears the material reference on any geom that used one.
  """
  buffer_names: set[str] = set()
  for tex in list(spec.textures):
    if len(tex.data) > 0:
      buffer_names.add(tex.name)
      spec.delete(tex)
  if not buffer_names:
    return
  mat_names: set[str] = set()
  for mat in list(spec.materials):
    tex_name = mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]
    if tex_name in buffer_names:
      mat_names.add(mat.name)
      spec.delete(mat)
  if not mat_names:
    return
  for geom in spec.geoms:
    if geom.material in mat_names:
      geom.material = ""


def _collapse_defaults(elem: ET.Element) -> None:
  """Collapse duplicate nested defaults with the same class name.

  MjSpec.to_xml() emits ``<default class="X"><default class="X">...`` after
  ``attach(prefix=...)``. This hoists the children of the inner duplicate into the
  outer element and removes the inner wrapper.
  """
  for child in list(elem):
    if child.tag == "default":
      _collapse_defaults(child)
  for child in list(elem):
    if (
      child.tag == "default"
      and child.get("class") is not None
      and len(child) == 1
      and child[0].tag == "default"
      and child[0].get("class") == child.get("class")
    ):
      inner = child[0]
      child.remove(inner)
      for grandchild in list(inner):
        child.append(grandchild)
      child.text = inner.text
      if inner.tail:
        last = list(child)[-1] if len(child) else None
        if last is not None:
          last.tail = (last.tail or "") + inner.tail


def _remove_empty_defaults(elem: ET.Element) -> None:
  """Remove empty ``<default/>`` and ``<default class="..."/>`` elements."""
  for child in list(elem):
    if child.tag == "default":
      _remove_empty_defaults(child)
      if len(child) == 0 and not (child.text and child.text.strip()):
        elem.remove(child)


def fix_spec_xml(xml: str, meshdir: str | None = None) -> str:
  """Fix known issues in MjSpec.to_xml() output.

  1. Remove empty ``<default/>`` and ``<default class="..."/>`` tags.
  2. Collapse duplicate nested defaults with the same class name.
  3. Optionally set ``meshdir`` in the ``<compiler>`` element.

  Args:
    xml: Raw XML string from ``MjSpec.to_xml()``.
    meshdir: If provided, set ``meshdir`` on the ``<compiler>`` element.
  """
  root = ET.fromstring(xml)

  for default_root in root.findall("default"):
    _remove_empty_defaults(default_root)
    _collapse_defaults(default_root)
  # Remove top-level <default> if it ended up empty.
  for default_root in list(root):
    if default_root.tag == "default" and len(default_root) == 0:
      root.remove(default_root)

  if meshdir is not None:
    compiler = root.find("compiler")
    if compiler is not None:
      compiler.set("meshdir", meshdir)

  ET.indent(root, space="  ")
  return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"
