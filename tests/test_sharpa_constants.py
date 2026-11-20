"""Tests for sharpa_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.sharpa_wave import sharpa_constants
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg


@pytest.fixture(scope="module", params=["left", "right"])
def side(request) -> sharpa_constants.Side:
  return request.param


@pytest.fixture(scope="module")
def model(side: sharpa_constants.Side) -> mujoco.MjModel:
  return Entity(sharpa_constants.get_sharpa_robot_cfg(side)).spec.compile()


@pytest.mark.parametrize(
  "cfg", sharpa_constants.SHARPA_ACTUATORS, ids=lambda c: c.target_names_expr[0]
)
def test_actuator_parameters(model, cfg):
  joints = [
    model.joint(i)
    for i in range(model.njnt)
    if re.fullmatch(cfg.target_names_expr[0], model.joint(i).name)
  ]
  assert joints
  for joint in joints:
    act = model.actuator(joint.name)
    dof = joint.dofadr[0]
    np.testing.assert_allclose(
      [
        act.gainprm[0],
        -act.biasprm[2],
        act.forcerange[1],
        model.dof_armature[dof],
        model.dof_frictionloss[dof],
        model.dof_damping[dof],
      ],
      [
        cfg.stiffness,
        cfg.damping,
        cfg.effort_limit,
        cfg.armature,
        cfg.frictionloss,
        cfg.viscous_damping,
      ],
    )


def test_every_joint_is_actuated_once(model):
  assert model.nu == model.njnt == 22
  np.testing.assert_array_equal(
    np.sort(model.actuator_trnid[:, 0]), np.arange(model.njnt)
  )


def test_collision_hulls_are_capped(model):
  for g in range(model.ngeom):
    if model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      graphadr = model.mesh_graphadr[model.geom_dataid[g]]
      assert model.mesh_graph[graphadr] <= 64, model.geom(g).name


def test_fingertips_collide_through_pad_and_shell(model, side):
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    pad = model.geom(f"{side}_{finger}_pad_collision")
    shell = model.geom(f"{side}_{finger}_DP_collision")
    assert pad.type == shell.type == mujoco.mjtGeom.mjGEOM_MESH
    assert pad.contype == shell.contype == 1
    np.testing.assert_allclose(pad.solref, (0.06, 0.9))
    np.testing.assert_allclose(shell.solref, (0.01, 1.0))


def test_fingertip_tactile_grids_land_on_the_pads(side):
  robot = sharpa_constants.get_sharpa_robot_cfg(side)
  sensors = sharpa_constants.get_fingertip_tactile_cfgs(side)
  model = Scene(SceneCfg(entities={"robot": robot}, sensors=sensors), "cpu").compile()
  assert [model.sensor(i).dim[0] for i in range(model.nsensor)] == [3 * 70] * 5


def test_mass_matches_menagerie(model):
  assert model.body_mass.sum() == pytest.approx(1.3151, abs=1e-4)
