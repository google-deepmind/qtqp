# How to Contribute

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Development

### Environment

The test suite needs a handful of optional sparse backends that are easiest to
get from conda-forge. From a clone of the repository:

```bash
conda create -n qtqp python=3.12
conda activate qtqp
conda install -y -c conda-forge suitesparse scikit-umfpack nanoeigenpy
python -m pip install 'scikit-sparse>=0.5' qdldl
python -m pip install -e '.[test]'
```

Two backends are platform specific and are installed by the runtime
dependencies where they are available: `py-mkl-pardiso` on Linux and Windows
`x86_64`, and `macldlt` on macOS `arm64`. `petsc4py` (`conda install -y -c
conda-forge petsc4py`) is optional and unavailable on Windows. Tests for a
linear solver whose dependency is missing are skipped rather than failed, so a
partial environment still gives a useful run.

### Checks

These are the same two gates CI runs, so a green run locally is a green run on
the pull request:

```bash
ruff check --select E9,F src/   # syntax errors, undefined names, unused imports
pytest                          # the full suite, including doctests in src/
```

Style rules are deliberately excluded from the lint gate; only correctness
rules (`E9`, `F`) are enforced.

To see what the suite reaches, run it with coverage. CI fails below the
threshold committed in `pyproject.toml`:

```bash
pytest --cov=qtqp --cov-report=term-missing
```

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
