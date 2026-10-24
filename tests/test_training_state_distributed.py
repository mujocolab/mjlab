"""Two real CPU/gloo learners: rank-local continuation and collective failures."""

import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
from test_training_state import assert_equal, continuation_state, make_runner


def _worker(rank: int, directory: Path) -> None:
  os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
  torch.set_num_threads(1)
  torch.distributed.init_process_group(
    "gloo",
    init_method=(directory / "rendezvous").as_uri(),
    rank=rank,
    world_size=2,
    timeout=timedelta(seconds=90),
  )
  runner = None
  try:
    for phase, updates in (("baseline", 4), ("split", 2), ("resumed", 2)):
      runner = make_runner(recurrent=True, rank=rank, log_dir=str(directory / phase))
      try:
        if phase == "resumed":
          runner.load(str(directory / "split" / "model_1.pt"))
        runner.learn(updates, init_at_random_ep_len=True)
        torch.distributed.barrier()
      finally:
        runner.env.close()
    assert runner is not None
    # Rank zero has a valid older checkpoint; rank one cannot read its input.
    # Neither learner may be modified before this read/preflight agreement.
    before = {
      key: value.clone() for key, value in runner.alg.actor.state_dict().items()
    }
    selected = directory / "split" / ("model_1.pt" if rank == 0 else "missing.pt")
    with pytest.raises(ValueError, match="read/preflight.*rank 1.*FileNotFoundError"):
      runner.load(str(selected))
    for key, value in runner.alg.actor.state_dict().items():
      assert torch.equal(value, before[key])
    # Hook/runtime failures after validation cannot be rolled back, but peers
    # must stop before entering another normalizer/gradient collective.
    original_restore = runner._restore_training_state
    if rank == 1:

      def failed_restore(state):
        raise RuntimeError("simulated custom hook failure")

      runner._restore_training_state = failed_restore
    with pytest.raises(ValueError, match="checkpoint restore.*rank 1.*custom hook"):
      runner.load(str(directory / "resumed" / "model_3.pt"))
    runner._restore_training_state = original_restore
    # A rank-local serializer error must reach both workers before any payload
    # gather or subsequent PPO gradient collective can strand its peer.
    if rank == 1:
      runner.env.unwrapped.__dict__["unsupported_memory"] = object()
    with pytest.raises(ValueError, match="rank 1.*training_state_dict"):
      runner.save_training_checkpoint(str(directory / "must_not_exist.pt"))
    if rank == 1:
      del runner.env.unwrapped.__dict__["unsupported_memory"]
    assert not (directory / "must_not_exist.pt").exists()
    torch.distributed.barrier()
    if rank == 0:
      with pytest.raises(ValueError, match="on every rank"):
        runner.save(str(directory / "rank_zero_only.pt"))
    torch.distributed.barrier()
    # Publication errors also reach all ranks; this leaves the previous file intact.
    if rank == 0:

      def failed_save(path, infos=None):
        raise OSError("simulated disk full")

      runner.save = failed_save
    with pytest.raises(OSError, match="simulated disk full"):
      runner.save_training_checkpoint(str(directory / "publication_failure.pt"))
    (directory / f"rank_{rank}.json").write_text(
      json.dumps({"failures_propagated": True})
    )
  finally:
    if runner is not None:
      runner.env.close()
    torch.distributed.destroy_process_group()


@pytest.mark.slow
def test_two_rank_cpu_stateful_resume_and_rank_failure_propagation(tmp_path):
  if not torch.distributed.is_gloo_available():
    pytest.skip("PyTorch was built without gloo")
  children = []
  streams = []
  try:
    for rank in range(2):
      stream = (tmp_path / f"rank_{rank}.log").open("w", encoding="utf-8")
      streams.append(stream)
      children.append(
        subprocess.Popen(
          [sys.executable, __file__, "worker", str(rank), str(tmp_path)],
          env={
            **os.environ,
            "FORCE_CPU": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
          },
          stdout=stream,
          stderr=subprocess.STDOUT,
        )
      )
    for child in children:
      child.wait(timeout=240)
    for rank, child in enumerate(children):
      assert child.returncode == 0, (tmp_path / f"rank_{rank}.log").read_text(
        encoding="utf-8"
      )[-10000:]
      assert json.loads((tmp_path / f"rank_{rank}.json").read_text())[
        "failures_propagated"
      ]
  finally:
    for child in children:
      if child.poll() is None:
        child.kill()
        child.wait(timeout=20)
    for stream in streams:
      stream.close()
  expected = torch.load(tmp_path / "baseline" / "model_3.pt", weights_only=False)
  actual = torch.load(tmp_path / "resumed" / "model_3.pt", weights_only=False)
  assert_equal(continuation_state(actual), continuation_state(expected))
  ranks = actual["training_state"]["ranks"]
  assert set(ranks) == {0, 1}
  first, second = (ranks[i]["learner"]["actor_state_dict"] for i in range(2))
  assert all(
    torch.equal(first[key], second[key])
    for key in first
    if key.startswith("obs_normalizer.")
  ), "Native RSL normalizers combine moments across ranks"
  assert not torch.equal(ranks[0]["rng"]["torch"], ranks[1]["rng"]["torch"])
  assert not np.array_equal(
    ranks[0]["simulation"]["arrays"]["data.qpos"],
    ranks[1]["simulation"]["arrays"]["data.qpos"],
  )
  assert any(
    not torch.equal(a, b)
    for a, b in zip(
      ranks[0]["recurrent"]["actor"],
      ranks[1]["recurrent"]["actor"],
      strict=True,
    )
  ), "Recurrent rollout memory must remain rank-local"


if __name__ == "__main__":
  _worker(int(sys.argv[2]), Path(sys.argv[3]))
