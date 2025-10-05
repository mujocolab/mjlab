from dataclasses import asdict, dataclass

import torch
import yaml

from mjlab.tasks.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from mjlab.utils.os import dump_yaml


def test_dump_yaml_torch_scalars(tmp_path):
  """Torch scalar tensors should serialize as numbers, not binary."""
  out = tmp_path / "scalars.yaml"

  data = {
    "float_scalar": torch.tensor(42.5),
    "int_scalar": torch.tensor(100),
    "zero_dim": torch.tensor(3.14),
  }

  dump_yaml(out, data)
  text = out.read_text(encoding="utf-8")

  # Must not have binary tags.
  assert "!!binary" not in text
  assert "42.5" in text
  assert "100" in text

  loaded = yaml.safe_load(text)
  assert loaded["float_scalar"] == 42.5
  assert loaded["int_scalar"] == 100
  assert abs(loaded["zero_dim"] - 3.14) < 0.01  # Tolerance for float32 precision.


def test_dump_yaml_nested_dataclasses(tmp_path):
  """Nested dataclasses should serialize cleanly."""

  @dataclass
  class RewardScales:
    tracking: float = 1.0
    action_rate: float = -0.01

  @dataclass
  class Config:
    rewards: RewardScales
    max_steps: int = 1000

  cfg = Config(rewards=RewardScales())
  out = tmp_path / "nested.yaml"

  dump_yaml(out, asdict(cfg))
  text = out.read_text(encoding="utf-8")

  assert "!!python" not in text
  loaded = yaml.safe_load(text)
  assert loaded["rewards"]["tracking"] == 1.0
  assert loaded["rewards"]["action_rate"] == -0.01


def test_dump_yaml_velocity_env_config(tmp_path):
  """Test that LocomotionVelocityEnvCfg serializes without binary tags."""
  env_cfg = LocomotionVelocityEnvCfg()
  env_cfg_dict = asdict(env_cfg)

  out = tmp_path / "params" / "env.yaml"
  dump_yaml(out, env_cfg_dict)

  text = out.read_text(encoding="utf-8")

  # Critical: no binary or python object tags.
  assert "!!binary" not in text, "Config should not contain binary data"
  assert "!!python" not in text, "Config should not contain Python object tags"

  # Verify it's valid YAML and can be loaded.
  loaded = yaml.safe_load(text)
  assert isinstance(loaded, dict)
