# Contributing to contingency-py

Thank you for your interest in contingency-py. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributing changes

1. Open an issue first for anything larger than a small fix, so the change can be discussed before you write code.
2. Fork the repository and create a branch from `main`.
3. Set up a development environment (Python 3.11 or later):

   ```bash
   python -m venv .venv
   .venv/bin/python -m pip install -e ".[dev]"
   .venv/bin/pytest
   ```

4. Add tests for every behavior change. Schedule logic is not merged without tests.
5. If you change the outcome of a schedule for a known event sequence, regenerate the affected fixture under `conformance/` and explain why in the pull request; the Rust implementation [contingency-rs](https://github.com/OperantKit/contingency-rs) replays the same fixtures.
6. Algorithms taken from the literature (for example, Fleshler & Hoffman, 1962) must cite the original publication in the docstring. Do not change such an algorithm without stating the reason.
7. Open a pull request against `main` and describe what changed and how you tested it.

## Reporting issues

Report bugs and request features in the [issue tracker](https://github.com/OperantKit/contingency-py/issues). For a bug, include:

- the schedule expression or the code that builds the schedule,
- the response/event sequence (times and operanda) that reproduces the problem,
- the outcome you expected and the outcome you observed,
- the versions of Python, contingency-py, and contingency-dsl-py.

## Getting support

Ask usage questions in the [issue tracker](https://github.com/OperantKit/contingency-py/issues) as well. The project is maintained by a single developer on a best-effort basis; see the Support section of the [README](README.md) for what to expect.
