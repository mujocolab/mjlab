# mjlab

This is extremely bad code and very much work in progress. I'm just exploring different design patterns for physics simulation + RL workflows.

## Installation

```bash
# For 5090 only.
uv pip install -U "jax[cuda12]<0.6.1"
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

```bash
uv pip install -e .
python -c "import mjlab"  # Trigger menagerie download.
```

## Benchmark

```bash
python mjlab/rl/train.py --num-envs 128 --agent-cfg.experiment-name cartpole_experiment --agent-cfg.max-iterations 1500
```

For cpu, add ` --agent-cfg.device "cpu"`.
