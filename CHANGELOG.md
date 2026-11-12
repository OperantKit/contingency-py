# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions are taken from git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and a Support section in the README.
- `contingency.envs` (extra `gym`): `ScheduleEnv`, a Gymnasium environment that runs any `Schedule` as a free-operant session, and the registered IDs `contingency/FR-v0` and `contingency/FI-v0` (available after `import contingency.envs`).
- `contingency.to_spec` (in `contingency.bridge.spec`): translates a contingency-dsl AST to the JSON schedule spec of the conformance corpus, so that `build_schedule_from_spec(to_spec(ast))` behaves like `from_dsl(ast)`. DSL constructs the spec has no form for (`BO`, `Overlay(target=)`, `Lag`, non-IRT `Pctl`, `Adj(delay|amount)`, MTS correction, GoNoGo `LH`, program-level parameter declarations) raise the new `SpecUnrepresentableError`, a subclass of `ScheduleConfigError`.
- `contingency/ConcVIVI-v0`: concurrent VI VI with changeover delay on operanda `left` and `right` (`mean_left`, `mean_right`, `cod`), with the `conc_vi_vi` factory in `contingency.envs.presets`.
- `core/`: a maturin project that builds the `contingency_core` extension module (the contingency-rs engine through PyO3) as the `contingency-core` wheel. contingency-rs is taken from GitHub at a fixed tag.
- `.github/workflows/wheels.yml`: builds the `contingency-core` wheel with maturin-action for Linux x86_64 (manylinux2014), macOS x86_64 and arm64, and Windows x86_64, imports each wheel that the runner can load in a clean venv, and uploads the wheels as workflow artifacts.
- Schedule spec: `Concurrent` takes `cod_directional` (`[{"from", "to", "cod"}, ...]`) and `punish` (`{operandum: spec}`), so directional COD and per-operandum punishment have a spec form. `to_spec` writes them for `COD(a->b)=x` and `PUNISH(k)=S`; the `PUNISH` directional and changeover forms are rejected as by `from_dsl`. New conformance fixtures `compound/concurrent_cod_directional.json` and `compound/concurrent_punish.json`.
- `contingency.engine`: `RustSchedule` and `from_spec` run a schedule spec on `contingency_core` behind the `Schedule` protocol, with this package's `Outcome`, `Reinforcer`, `ResponseEvent` and errors (`ValueError` becomes `ScheduleConfigError`, `RuntimeError` becomes `ScheduleStateError`). `from_dsl` takes `engine="python"` (default) or `engine="rust"`.

### Changed

- `from_dsl` raises `ScheduleConfigError` for combinator parameters the runtime does not read, instead of dropping them: `BO` greater than zero on `Mult` / `Mix` (the runtime has no blackout), including a program-level `BO` declaration in a program with a `Mult` / `Mix`; `BO` on other combinators; `COD` / `FRCO` / `PUNISH` outside `Conc`; `Overlay(target=)`. Programs that were accepted before ran as if the parameter were absent. `BO=0` on `Mult` / `Mix` means no blackout and is accepted by `from_dsl` and written as no `BO` by `to_spec`.
- `Concurrent` COD now holds a reinforcer that a component sets up during the changeover delay and delivers it on the first response on that operandum after the COD, instead of discarding it; the component's next interval or ratio starts from that delivery (Herrnstein, 1961). Results of concurrent schedules with `cod > 0` differ from earlier releases. The conformance fixture `compound/concurrent_cod.json` is updated accordingly.

### Fixed

- `Concurrent`: on a step where the COD holds the responded operandum's reinforcer, a reinforcer that another component delivers on its tick at the same step is returned (with `meta["cod_suppressed"]`) instead of being replaced by an empty outcome, unless a punisher fires at that step. New conformance fixture `compound/concurrent_cod_tick.json`.
- `Chained`, `Tandem`, `Multiple`: time-based components (FI, VI, RI, FT, VT, RT, DRO) count from their first step after activation, as documented, instead of from session start or the component's last reinforcer. Each component runs on its own clock, which stops while the component is inactive; a re-entered component resumes with the time it had accumulated. For example, `Chained([FR(5), FI(30.0)])` with one response per second now reinforces at 35 and 71 instead of 30 and 60. `Mixed` behaves the same way. New conformance fixture `compound/chained_fr2_fi5.json`.

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
