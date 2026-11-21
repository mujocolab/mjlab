"""Tests for multi-motion tracking functionality."""

from typing import cast

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts import pack_motions
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import (
  MOTION_FIELDS,
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
  resolve_motion_file,
)
from mjlab.utils.forward_kinematics import ForwardKinematics


@pytest.fixture(scope="module")
def fk() -> ForwardKinematics:
  """Forward kinematics of the G1 to its first three bodies."""
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  return ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.from_numpy(model.jnt_qposadr[1:]),
    torch.tensor([0, 1, 2]),
  )


@pytest.fixture
def create_motion_file(tmp_path, fk):
  """Factory for motion NPZ files of random root and joint states."""

  def _create(name: str, num_frames: int, fps: float = 50.0) -> str:
    root_pos, root_lin_vel, root_ang_vel = torch.randn(3, num_frames, 3)
    root_quat = torch.nn.functional.normalize(torch.randn(num_frames, 4), dim=-1)
    joint_pos, joint_vel = torch.randn(2, num_frames, len(fk.joint_names))
    path = tmp_path / f"{name}.npz"
    np.savez(
      path,
      joint_pos=joint_pos.numpy(),
      joint_vel=joint_vel.numpy(),
      root_pos_w=root_pos.numpy(),
      root_quat_w=root_quat.numpy(),
      root_lin_vel_w=root_lin_vel.numpy(),
      root_ang_vel_w=root_ang_vel.numpy(),
      joint_names=fk.joint_names,
      fps=np.array([fps]),
    )
    return str(path)

  return _create


def test_resolve_motion_file(create_motion_file, tmp_path):
  """A list, a glob in sorted order and an existing path with glob characters."""
  paths = [create_motion_file(name, num_frames=2) for name in ("a", "b", "c[1]")]
  assert resolve_motion_file(paths) == paths
  assert resolve_motion_file(str(tmp_path / "?.npz")) == paths[:2]
  assert resolve_motion_file(paths[2]) == [paths[2]]
  with pytest.raises(ValueError, match="No files matched"):
    resolve_motion_file(str(tmp_path / "missing_*.npz"))


def test_loader_rejects_other_joints(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=20)
  data = dict(np.load(path))
  data["joint_names"] = data["joint_names"][::-1]
  np.savez(path, **data)
  with pytest.raises(ValueError, match="motion.npz: motion joints"):
    MotionLoader(path, fk)


def test_loader_rejects_file_without_motion_keys(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=20)
  data = dict(np.load(path))
  del data["root_pos_w"], data["joint_names"]
  np.savez(path, **data)
  with pytest.raises(ValueError, match="csv_to_npz") as error:
    MotionLoader(path, fk)
  assert str(error.value).startswith(f"{path}: the motion file lacks")
  assert "['root_pos_w', 'joint_names']" in str(error.value)


@pytest.mark.parametrize(
  ("clip", "match"),
  [
    ({"num_frames": 1}, "clip b has 1 frame"),
    ({"num_frames": 20, "fps": 30.0}, "fps 30.0 is not the 50.0"),
  ],
  ids=["one_frame", "mixed_fps"],
)
def test_loader_rejects_clip(create_motion_file, fk, clip, match):
  paths = [create_motion_file("a", num_frames=20), create_motion_file("b", **clip)]
  with pytest.raises(ValueError, match=match):
    MotionLoader(paths, fk)


def test_loader_body_state_is_forward_kinematics(create_motion_file, fk):
  loader = MotionLoader(create_motion_file("motion", num_frames=20), fk)
  frames = torch.tensor([3, 0, 19])
  expected = fk(
    loader.root_pos_w[frames],
    loader.root_quat_w[frames],
    loader.joint_pos[frames],
    loader.root_lin_vel_w[frames],
    loader.root_ang_vel_w[frames],
    loader.joint_vel[frames],
  )
  for got, want in zip(loader.body_state(frames), expected, strict=True):
    assert torch.equal(got, want)


def test_loader_concatenates_clips(create_motion_file, fk, tmp_path):
  """A list of clips and the directory that pack_motions writes from them load as
  the same concatenated motion; pack_motions rejects a clip of another fps."""
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
  ]
  packed_dir = tmp_path / "packed"
  pack_motions.main(str(tmp_path / "*.npz"), str(packed_dir))

  packed, clips = MotionLoader(str(packed_dir), fk), MotionLoader(paths, fk)
  joint_pos = np.concatenate([np.load(path)["joint_pos"] for path in paths])
  assert torch.equal(clips.joint_pos, torch.from_numpy(joint_pos))
  assert packed.motion_ends.tolist() == clips.motion_ends.tolist() == [50, 150]
  assert packed.motion_names == clips.motion_names == ["a", "b"]
  assert packed.fps == clips.fps == 50.0
  for name in MOTION_FIELDS:
    assert torch.equal(getattr(packed, name), getattr(clips, name))

  create_motion_file("c", num_frames=20, fps=30.0)
  with pytest.raises(ValueError, match="different fps"):
    pack_motions.main(str(tmp_path / "*.npz"), str(packed_dir))


def test_command_bodies_follow_time_steps(create_motion_file, monkeypatch):
  """The command's tracked bodies are the forward kinematics of the motion at
  ``time_steps`` after resets, steps, clip ends and ``set_time_steps``, ``start``
  mode resets to the first frame of a clip, and the command resamples at a clip
  end."""
  cfg = unitree_g1_flat_tracking_env_cfg()
  cfg.scene.num_envs = 4
  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = [
    create_motion_file("a", num_frames=20),
    create_motion_file("b", num_frames=30),
  ]
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  last_frames = command.motion.motion_ends - 1

  def check() -> None:
    pos, quat, lin_vel, ang_vel = command.motion.body_state(command.time_steps)
    origins = env.scene.env_origins[:, None, :]
    torch.testing.assert_close(command.body_pos_w, pos + origins)
    torch.testing.assert_close(command.body_quat_w, quat)
    torch.testing.assert_close(command.body_lin_vel_w, lin_vel)
    torch.testing.assert_close(command.body_ang_vel_w, ang_vel)

  check()
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  for _ in range(60):
    was_last = torch.isin(command.time_steps, last_frames)
    _, _, _, truncated, _ = env.step(action)
    # An episode on its clip's last frame ends as a time-out, and no env moves
    # from one clip into the next.
    assert truncated[was_last].all()
    assert not torch.isin(command.time_steps, command.motion.motion_ends).any()
    check()

  command.set_time_steps(torch.tensor([1]), torch.tensor([7]))
  assert command.time_steps[1] == 7
  check()

  command.reset_to_frame(torch.tensor([0, 2]), 25)
  assert command.time_steps.tolist()[0::2] == [25, 25]
  check()

  # With several clips, ``start`` resets an env to the first frame of a clip.
  command.cfg.sampling_mode = "start"
  reset_frames: set[int] = set()
  for _ in range(8):
    command.reset(torch.arange(env.num_envs))
    reset_frames.update(command.time_steps.tolist())
  assert reset_frames == {0, 20}

  # The command resamples an env whose frame steps past the last frame of its
  # clip, which holds the clip boundary without the ``motion_end`` termination.
  resampled: list[torch.Tensor] = []
  monkeypatch.setattr(command, "_resample_command", resampled.append)
  command.set_time_steps(torch.arange(4), torch.tensor([19, 5, 30, 20]))
  env.command_manager.compute(dt=env.step_dt)
  assert [ids.tolist() for ids in resampled] == [[0]]
  env.close()
