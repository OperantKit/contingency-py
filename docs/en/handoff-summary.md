# `contingency-rs` handoff summary

This document is the one-page onboarding for a Rust implementer
porting `contingency-py` to `contingency-rs`. It summarises what
exists, what is deferred, and where to start reading.

## What exists in `contingency-py`

### Runtime schedules (production-ready)

| Family | Classes |
|---|---|
| Ratio | `FR`, `VR`, `RR`, `CRF` (factory -> `FR(1)`) |
| Interval | `FI`, `VI`, `RI`, `LimitedHold` |
| Time-based | `FT`, `VT`, `RT`, `EXT` |
| Differential | `DRO` (resetting/momentary), `DRL`, `DRH` |
| Compound | `Concurrent` (COD + COR), `Multiple`, `Chained`, `Tandem`, `Alternative` |
| Progressive | `ProgressiveRatio` with `arithmetic`, `geometric`, `richardson_roberts` step fns |

All implement the `Schedule` Protocol (`step(now, event) -> Outcome`,
`reset() -> None`). See [protocol.md](protocol.md) for the full
contract.

### Value objects

- `ResponseEvent(time, operandum="main")`
- `Observation(time, response_count=0)`
- `Reinforcer(time, magnitude=1.0, label="SR+")`
- `Outcome(reinforced=False, reinforcer=None, meta={})`

Invariants: `reinforced <=> reinforcer.is_some()`; enforced at
construction.

### Error taxonomy

```
ContingencyError
├── ScheduleConfigError
├── ScheduleStateError
└── HardwareError
    └── NotConnectedError
```

Maps to a four-variant Rust `enum Error` (see [protocol.md](protocol.md)).

### Helpers

- `contingency.helpers.fleshler_hoffman.generate_intervals`,
  `generate_ratios`, `generate_intervals_hantula1991` — deterministic
  pool generators with `seed` parameter.

### DSL bridge

`contingency.bridge.from_dsl(ast_node, time_unit_seconds=1.0) -> Schedule`
translates `contingency-dsl` AST nodes into runtime schedules. See
that module's docstring for the full translation table.

### Hardware abstraction layer

`contingency.hw` provides three backends (`VirtualApparatus`,
`SerialBackend`, `HilBridgeApparatus`) behind an `Apparatus`
Protocol. The Rust port will want to mirror this shape once the PyO3
/ KMP front-ends are decided; this package intentionally does not
document the HAL here because the Rust implementation will likely
be a fresh, lower-level `hal` crate sitting beside `contingency-rs`.

## What's deferred

From the bridge (`src/contingency/bridge/__init__.py`):

- `SecondOrder` — not implemented.
- `IdentifierRef` — requires the DSL analyzer pass to substitute
  let-bindings.
- `AdjustingSchedule`, `AversiveSchedule`, `InterlockingSchedule`,
  `ResponseCostWrapped`, `TimeoutWrapped`, `TrialBased` — all
  deferred.
- Combinators `CONJ`, `MIX`, `OVERLAY`, `INTERPOLATE`, `INTERP` —
  deferred.
- `Modifier` kind `Pctl` (percentile schedules) — deferred.
- `Concurrent` parameter `PUNISH` — deferred.
- `Concurrent` directional COD (`cod_asymmetric`) — deferred.

The Rust port should surface the same `Error::Config` for every
deferred construct until the feature is implemented.

## Known semantic divergences from the Swift original

See [swift-legacy-map.md](swift-legacy-map.md) for file-by-file
pointers. Key semantic differences that a Rust port must honour
(they are the **Python** behaviour, not the Swift behaviour):

1. **Anchoring**. Time-based schedules (`FT`, `VT`, `RT`, `DRO`)
   anchor their clock on the **first `step()` call**, not at `t=0`.
   `FI`, `VI`, `RI` anchor at construction (`arm_time = interval` or
   `arm_time = sequence[0]`).
2. **Single-fire per step**. If the caller steps late by `k > 1`
   intervals, only **one** reinforcer is emitted. Callers must step
   often enough to honour the schedule.
3. **Random-family sampling is internal**. `RR`, `RI`, `RT` sample
   draws internally (Bernoulli / exponential). Swift's `RI`/`RT`/`RR`
   files delegated to their Fixed counterparts with externally
   supplied values.
4. **Unit-agnostic time**. All times are unit-agnostic floats; the
   unit contract is external (`contingency-dsl` declares it,
   `experiment-io` enforces it). Swift was millisecond-typed.
5. **Outcome invariants are enforced at construction**. Do not allow
   a reinforced outcome without a reinforcer or vice versa.

## Gotchas for a Rust port

1. **`TIME_TOL = 1e-9`** must be applied exactly as specified in
   every comparison: monotonic time, event-time matching, elapsed
   time, sliding-window eviction, LimitedHold expiry.
2. **`Concurrent` advances every component on every step**, even
   components that did not receive the event. This is essential for
   time-based components (FT/VT/RT) to advance on ticks that arrive
   as responses on other operanda.
3. **`Concurrent` gating order**: the changeover response is covered
   by the COD it opens. Register the changeover *before* checking
   COD. COD does not gate tick-side reinforcement from other
   components.
4. **`Chained` / `Tandem` do not step inactive components**. A
   time-based terminal link anchors on its first step after
   activation, not at the transition instant.
5. **`Alternative` forwards `(now, event)` to both components on
   every step**. Both components maintain their own state; the
   wrapper only gates surfaced outcomes and resets both on a win.
6. **`DRO` momentary** evaluates the boundary **before** recording
   the incoming event (half-open window `[anchor, now)`). The
   resetting variant is the opposite: the event resets the timer and
   pre-empts any reinforcement on that step.
7. **`DRH`'s window is not emptied on reinforcement**. A sustained
   high-rate train keeps firing; breakpoint-like behaviour must be
   added externally.
8. **`ProgressiveRatio`'s `step_fn` is called lazily**. Invalid
   return values surface as `Config` errors only on the first
   response that consults the function — not at construction.
9. **`reset()` does not re-sample random parameters on non-seeded
   schedules**. `VR(mean, seed=None).reset()` re-seeds from
   `seed=None`, yielding a fresh RNG. For bit-reproducibility,
   always construct with a seed.
10. **`RR`, `RI`, `RT` snapshot the RNG state at construction**.
    When a caller passes a shared `Random`, the snapshot is taken
    immediately — not on every `reset()`. This means `reset()`
    returns the schedule's private draw sequence to its initial
    state even if the shared RNG has been advanced elsewhere.
11. **Sub-seed domain for VR/VT/VI**: `rng.randrange(0, 2**31 - 1)`
    for VR/VT and `rng.getrandbits(64)` for VI. Rust must use the
    same ranges only if aiming for Python bit-identity; otherwise
    pin sequences via conformance fixtures.
12. **`Outcome.meta` keys observed** across the family:
    `"current_component"` (str for `Multiple`/`Chained`, int for
    `Tandem`), `"chain_transition"` (bool), `"cod_suppressed"`
    (bool), `"operandum"` (str), `"alternative_winner"`
    (`"first"` | `"second"`).

## Suggested Rust module structure

```
contingency-rs/
├── src/
│   ├── lib.rs                   // re-exports
│   ├── entities.rs              // ResponseEvent, Reinforcer, Outcome, ...
│   ├── errors.rs                // Error enum
│   ├── schedule.rs              // Schedule trait
│   ├── schedules/
│   │   ├── mod.rs
│   │   ├── ratio.rs             // FR, VR, RR
│   │   ├── interval.rs          // FI, VI, RI, LimitedHold
│   │   ├── time_based.rs        // FT, VT, RT, EXT
│   │   ├── differential.rs      // DRO, DRL, DRH
│   │   ├── sequence.rs          // Multiple, Chained, Tandem
│   │   ├── concurrent.rs        // Concurrent
│   │   ├── alternative.rs       // Alternative
│   │   └── progressive.rs       // ProgressiveRatio + step fns
│   ├── helpers/
│   │   └── fleshler_hoffman.rs  // generate_intervals / generate_ratios
│   └── builder.rs               // ScheduleBuilder facade
└── tests/
    └── conformance.rs           // Loads ../../contingency-py/conformance/*.json
```

Suggested crate features:

- `serde` — for conformance fixture loading.
- `pyo3` — for Python bindings exposing `Schedule` as a Python type.
- `wasm` — for browser demos.
- `c-ffi` — for the HIL bridge binary.

## Where to start reading

1. [`protocol.md`](protocol.md) — the single source of truth for the
   public API shape.
2. [`algorithms/`](algorithms/) — one file per schedule, 80% of the
   implementation decisions.
3. [`../conformance/README.md`](../../conformance/README.md) — how to
   consume the fixtures from Rust tests.
4. [`swift-legacy-map.md`](swift-legacy-map.md) — only needed for the
   11 atomic schedules; the new compound schedules have no Swift
   source.

## Test surface of `contingency-py`

The Python package has 613 tests (100% green) and 99% line coverage.
Reading a test is often faster than reading the source — each
schedule has a dedicated file under [`../tests/`](../../tests/):

- `test_ratio_schedules.py`
- `test_interval_schedules.py`
- `test_time_schedules.py`
- `test_differential_schedules.py`
- `test_concurrent.py`
- `test_sequence_schedules.py`
- `test_alternative.py`
- `test_progressive.py`
- `test_fleshler_hoffman.py`
- `test_bridge.py` (DSL translation)
- `test_builder.py`
- `test_conformance.py` (regression check against the corpus)
