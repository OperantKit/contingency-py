# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions are taken from git tags.

## [Unreleased]

### Added

- JOSS paper draft (`paper/paper.md`, `paper/paper.bib`) and a workflow that builds its PDF.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and a Support section in the README.

## [0.1.1]

### Fixed

- The DSL bridge passes `@algorithm(seed=..., n=...)` to stochastic schedules, so seeded VI/VR/RI/RR/VT/RT schedules built from DSL source are reproducible.

### Changed

- Depend on `contingency-dsl-py` from PyPI instead of a direct git URL.
- Add a trusted-publishing workflow for PyPI.

## [0.1.0]

### Added

- Atomic schedules: FR, VR, RR, FI, VI, RI, FT, VT, RT, CRF, EXT.
- Compound schedules: Concurrent (with changeover delay), Multiple, Chained, Tandem, Alternative.
- Differential reinforcement: DRO (resetting, momentary), DRL, DRH.
- Progressive-ratio schedules (arithmetic, geometric, Hantula step functions) and Limited Hold.
- Trial-based and aversive schedule families.
- Fleshler-Hoffman VI/VR value generator, including the Hantula (1991) program variant.
- `ScheduleBuilder` and the DSL bridge `contingency.bridge.from_dsl`.
- Hardware abstraction layer `contingency.hw` with virtual, serial, and HIL bridge backends.
- Conformance fixtures under `conformance/` shared with the Rust implementation.

### Changed

- The concurrent-schedule changeover requirement is named Fixed-Ratio Changeover (`frco`).

[Unreleased]: https://github.com/OperantKit/contingency-py/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/OperantKit/contingency-py/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/OperantKit/contingency-py/releases/tag/v0.1.0
