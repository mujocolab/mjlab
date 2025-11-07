# mjlab Installation Guide

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**:
  - Linux (recommended)
  - macOS (limited support – see note below)
  - Windows (compatible - see note below)
- **GPU**: NVIDIA GPU strongly recommended
  - **CUDA Compatibility**: Not all CUDA versions are supported by MuJoCo Warp
    - Check
      [mujoco_warp#101](https://github.com/google-deepmind/mujoco_warp/issues/101)
      for CUDA version compatibility
    - **Recommended**: CUDA 12.4+ (for
      [conditional control flow](https://nvidia.github.io/warp/modules/runtime.html#conditional-execution)
      in CUDA graphs)

> ⚠️ **Important Note on macOS**: mjlab is designed for large-scale training in
> GPU-accelerated simulations. Since macOS does not support GPU acceleration, it
> is **not recommended** for training. Even policy evaluation runs significantly
> slower on macOS. We are working on improving this with a C-based MuJoCo
> backend for evaluation — stay tuned for updates.

> **Note on Windows**: Since warp/mujoco-warp are supported on Windows, Windows
> compatibility is available with mjlab. Core functionality (e.g. the demo and train
> scripts) has been tested, but support for Windows should be considered
> experimental. Windows users may also consider WSL as an option.
---

## ⚠️ Beta Status

mjlab is currently in **beta**. Expect frequent breaking changes in the coming weeks.
There is **no stable release yet**.

- The first beta snapshot is available on PyPI.
- **Recommended**: install from source (or Git) to stay up-to-date with fixes
  and improvements.

---

## Prerequisites

### Install uv

If you haven't already installed [uv](https://docs.astral.sh/uv/), run:

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
uv add "mjlab @ git+https://github.com/mujocolab/mjlab" "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@fb9bf88399796f161a4a2b129d846484da8a4ad0"
```

> **Note**: `mujoco-warp` must be installed from Git since it's not available on PyPI.

---

### Method 2: From PyPI (Beta Snapshot)

You can install the latest beta snapshot from PyPI, but note:
- It is **not stable**
- You still need to install `mujoco-warp` from Git

```bash
uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"
```

---

### Method 3: Using pip (venv, conda, virtualenv, etc.)

While mjlab is designed to work with [uv](https://docs.astral.sh/uv/), you can
also use it with any pip-based virtual environment (venv, conda, virtualenv, etc.).

1. **Create and activate your virtual environment**:

   **Using venv** (built-in):
   ```bash
   python -m venv mjlab-env
   source mjlab-env/bin/activate
   ```

   **Using conda**:
   ```bash
   conda create -n mjlab python=3.13
   conda activate mjlab
   ```

2. **Install mjlab and dependencies via pip**:

   **From Source (Recommended)**:
   ```bash
   pip install git+https://github.com/google-deepmind/mujoco_warp@fb9bf88399796f161a4a2b129d846484da8a4ad0
   git clone https://github.com/mujocolab/mjlab.git && cd mjlab
   pip install -e .
   ```

   > **Note**: You must install `mujoco-warp` from Git before running
   > `pip install -e .` since it's not available on PyPI and pip cannot resolve
   > the Git dependency specified in `pyproject.toml` (which uses uv-specific
   > syntax).

   **From PyPI**:
   ```bash
   pip install git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9
   pip install mjlab
   ```

---

### Method 4: Docker

- Install [Docker](https://docs.docker.com/engine/install/)
- Install an appropriate NVIDIA driver for your system and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - Be sure to register the container runtime with Docker and restart, as described in [the Docker section of the install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)
- `make docker-build`
- Use the included helper script to run an `mjlab` docker container with many useful arguments included: `./scripts/run_docker.sh`
  - Demo with viewer: `./scripts/run_docker.sh uv run demo`
  - Training example:`./scripts/run_docker.sh uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096`

## Verification

After installation, verify that mjlab is working by running the demo:

```bash
# If working inside the mjlab directory with uv.
uv run demo

# If mjlab is installed as a dependency in your project with uv.
uv run python -m mjlab.scripts.demo

# If installed via pip (conda, venv, etc.), use the CLI command directly.
demo

# Or use the module syntax (works anywhere mjlab is installed).
python -m mjlab.scripts.demo
```

---

## Troubleshooting

If you run into problems:

1. **Check the FAQ**: [faq.md](faq.md) may have answers to common issues.
2. **CUDA Issues**: Verify your CUDA version is supported by MuJoCo Warp
   ([see compatibility list](https://github.com/google-deepmind/mujoco_warp/issues/101)).
3. **macOS Slowness**: Training is not supported; evaluation may still be slow
   (see macOS note above).
4. **Still stuck?** Open an issue on
   [GitHub Issues](https://github.com/mujocolab/mjlab/issues).
