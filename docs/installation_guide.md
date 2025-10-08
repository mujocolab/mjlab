# mjlab Installation Guide

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**:
  - Linux (recommended)
  - macOS (limited support – see note below)
  - Windows (untested)
- **GPU**: NVIDIA GPU strongly recommended
  - **CUDA Compatibility**: Not all CUDA versions are supported by MuJoCo Warp
    - Check [mujoco_warp#101](https://github.com/google-deepmind/mujoco_warp/issues/101) for CUDA version compatibility
    - **Recommended**: CUDA 12.4+ (for [conditional control flow](https://nvidia.github.io/warp/modules/runtime.html#conditional-execution) in CUDA graphs)

> ⚠️ **Important Note on macOS**: mjlab is designed for large-scale training in
> GPU-accelerated simulations. Since macOS does not support GPU acceleration, it
> is **not recommended** for training. Even policy evaluation runs significantly
> slower on macOS. We are working on improving this with a C-based MuJoCo
> backend for evaluation — stay tuned for updates.

---

## ⚠️ Beta Status

mjlab is currently in **beta**. Expect frequent breaking changes in the coming weeks.
There is **no stable release yet**.

- The first beta snapshot is available on PyPI.
- **Recommended**: install from source (or Git) to stay up-to-date with fixes and improvements.

---

## Prerequisites

### Install uv

If you haven’t already installed [uv](https://docs.astral.sh/uv/), run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation Methods

### Method 1: From Source (Recommended)

Use this method if you want the latest beta updates.

#### Option A: Local Editable Install

1. Clone the repository:
```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
```

2. Add as an editable dependency to your project:
```bash
uv add --editable /path/to/cloned/mjlab
```

#### Option B: Direct Git Install

Install directly from GitHub without cloning:

```bash
uv add "mjlab @ git+https://github.com/mujocolab/mjlab" "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp"
```

> **Note**: `mujoco-warp` must be installed from Git since it’s not available on PyPI.

---

### Method 2: From PyPI (Beta Snapshot)

You can install the latest beta snapshot from PyPI, but note:
- It is **not stable**
- You still need to install `mujoco-warp` from Git

```bash
uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"
```

---

## Verification

After installation, verify that mjlab is working by running the demo:

```bash
# If working inside the mjlab directory
uv run demo

# If mjlab is installed as a dependency in your project
uv run python -m mjlab.scripts.demo
```

---

## Troubleshooting

If you run into problems:

1. **Check the FAQ**: [faq.md](faq.md) may have answers to common issues.
2. **CUDA Issues**: Verify your CUDA version is supported by MuJoCo Warp ([see compatibility list](https://github.com/google-deepmind/mujoco_warp/issues/101)).
3. **macOS Slowness**: Training is not supported; evaluation may still be slow (see macOS note above).
4. **Still stuck?** Open an issue on [GitHub Issues](https://github.com/mujocolab/mjlab/issues).
