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

These are the same gates CI runs, so a green run locally is a green run on
the pull request:

```bash
ruff check --select E9,F src/   # syntax errors, undefined names, unused imports
pytest                          # the full suite, including doctests in src/
pip-audit                       # known vulnerabilities in the resolved env
bandit -r src/ --severity-level high   # risky code patterns
```

Style rules are deliberately excluded from the lint gate; only correctness
rules (`E9`, `F`) are enforced.

`bandit` is gated at high severity. `src/` reports twelve low findings, all
`B101` (`assert_used`), and each one is a deliberate programming-error check on
an argument the caller chose; `pip-audit` audits whatever is installed, so run
it from the environment you built above rather than an isolated one.

To see what the suite reaches, run it with coverage:

```bash
pytest --cov --cov-report=term-missing
```

Coverage is reported, not gated. Every CI leg installs a different subset of
the optional linear solvers, so the figure differs per leg and a single
threshold would move with whichever packages happen to resolve.
`test_required_solvers_are_available` guards the case a threshold was meant to
catch: a backend that CI installs dropping out of the run instead of its
parametrized tests silently disappearing.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
