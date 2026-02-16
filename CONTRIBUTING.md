# Contributing

Bug fixes and documentation improvements are always welcome. For new features, please open an issue first so we can discuss whether it fits and work out the design, as we're intentional about keeping the scope focused.

## Workflow

1. Fork the repository and create a feature branch.
2. Make your changes.
3. Ensure formatting, type checking, and tests pass: `make test-all`.
4. Submit a pull request.

Type checking (`make type`) is required, PRs that don't pass will be blocked. You can optionally install pre-commit hooks (`pre-commit install`) to catch issues early.

## Changelog

Add entries to the "Upcoming version" section in `docs/source/changelog.rst` under the appropriate category (Added / Changed / Fixed), following [Keep a Changelog](https://keepachangelog.com/) conventions.

## Getting Help

- **Issues**: https://github.com/mujocolab/mjlab/issues
- **Discussions**: https://github.com/mujocolab/mjlab/discussions

## License

By contributing, you agree your contributions will be licensed under Apache 2.0.