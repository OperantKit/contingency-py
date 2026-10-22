# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions are taken from git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and a Support section in the README.
- `contingency.envs` (extra `gym`): `ScheduleEnv`, a Gymnasium environment that runs any `Schedule` as a free-operant session, and the registered IDs `contingency/FR-v0` and `contingency/FI-v0` (available after `import contingency.envs`).
- `contingency/ConcVIVI-v0`: concurrent VI VI with changeover delay on operanda `left` and `right` (`mean_left`, `mean_right`, `cod`), with the `conc_vi_vi` factory in `contingency.envs.presets`.

### Changed

- `Concurrent` COD now holds a reinforcer that a component sets up during the changeover delay and delivers it on the first response on that operandum after the COD, instead of discarding it; the component's next interval or ratio starts from that delivery (Herrnstein, 1961). Results of concurrent schedules with `cod > 0` differ from earlier releases. The conformance fixture `compound/concurrent_cod.json` is updated accordingly.

### Fixed

- `Concurrent`: on a step where the COD holds the responded operandum's reinforcer, a reinforcer that another component delivers on its tick at the same step is returned (with `meta["cod_suppressed"]`) instead of being replaced by an empty outcome, unless a punisher fires at that step. New conformance fixture `compound/concurrent_cod_tick.json`.

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
