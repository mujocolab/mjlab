# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

# heat_bench research references

`heat_bench/` (passive thermal/battery/torque benchmark for trained
policies) bases several design decisions on published research rather than
guesswork — the 14-node thermal topology, the 50Hz/200Hz update split,
both battery models, and the Rd(T)/Kt(T) motor coefficients. Full context
and citations live in `heat_bench/README.md`'s "References" section; the
papers themselves:

1. Qian et al., *"Learning Thermal-Aware Locomotion Policies for an
   Electrically-Actuated Quadruped Robot,"* arXiv:2603.01631.
2. Wan et al., *"Learning to Balance Motor Thermal Safety and Quadrupedal
   Locomotion Performance with Residual Policy,"* arXiv:2605.27046.
3. Lin, Qian, Luo, Liang, *"Temperature Distribution Prediction of the
   Quadruped Robot Based on the Lumped-parameter Thermal Networks,"* ROBOT
   journal, 2025 (not on arXiv).
4. Shu, Huang, Ren, Wu, Li, *"Learning-Based Model Predictive Control for
   Legged Robots with Battery–Supercapacitor Hybrid Energy Storage
   System,"* Appl. Sci. 2025, 15, 382, 10.3390/app15010382.
5. Petit, Prada, Sauvant-Moynot, *"Development of an empirical aging model
   for Li-ion batteries and application to assess the impact of
   Vehicle-to-Grid strategies on battery lifetime,"* Appl. Energy 2016,
   172, 398–407.
6. Unitree, *Go2 battery specification* (BT2-05), unitree.com/go2/battery
   — a data source, not a paper.
7. U.S. Bureau of Standards, *Copper Wire Card*, Misc. Pub. No. 17, 1919
   — annealed-copper resistance temperature coefficient (data source).
8. Arnold Magnetic Technologies, *N42 NdFeB* datasheet — reversible
   α(Br) of NdFeB magnets (data source).

When adding a new heat_bench design decision backed by research, cite it
in both places: inline in the relevant module/config comment, and in
`heat_bench/README.md`'s References list (add here too if it's a new
source).
