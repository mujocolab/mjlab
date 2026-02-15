# Releasing

## Pre-release checklist

1. Bump `version` in `pyproject.toml`.
2. Update `version` and `date-released` in `CITATION.cff`.
3. Update the "Upcoming version (not yet released)" heading in `docs/source/changelog.rst` to the new version number and date.
4. Commit the version bump, then create an annotated tag:

```sh
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## Build and verify

Clean previous build artifacts, then build:

```sh
rm -rf dist/
make build
```

This runs `uv build` to produce a wheel and sdist in `dist/`, then smoke-tests
both artifacts in isolated environments.

## Test on TestPyPI (optional but recommended)

Upload to TestPyPI first to catch packaging issues before the real release:

```sh
UV_PUBLISH_TOKEN=<your-testpypi-token> make publish-test
```

Then verify the upload works end-to-end. Use `--index-strategy unsafe-best-match`
because TestPyPI won't have all dependencies and uv needs to fall back to real
PyPI for them:

```sh
uvx --extra-index-url https://test.pypi.org/simple/ \
    --index-strategy unsafe-best-match \
    --from mjlab \
    demo
```

Note: TestPyPI requires a separate account and token from real PyPI.
Generate one at https://test.pypi.org/manage/account/token/.

## Publish to PyPI

```sh
UV_PUBLISH_TOKEN=<your-pypi-token> make publish
```

Generate a token at https://pypi.org/manage/account/token/.

## Post-release

Verify the release installs and runs correctly. Use `--refresh` to bypass
the `uvx` cache (which may still hold the TestPyPI version):

```sh
uvx --refresh --from mjlab demo
```

## Releasing from a past tag

If the tag has already been created and HEAD has moved ahead, check out the
tag before building:

```sh
git checkout vX.Y.Z
make build
make publish
git checkout main
```
