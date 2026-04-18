# Schedule Protocol — Language-Neutral Contract

This document is the authoritative surface that `contingency-rs` must
reproduce. Every schedule in the family conforms to it; compound
schedules compose over it.

## Python source of truth

- [src/contingency/interfaces.py](../src/contingency/interfaces.py) — the
  `Schedule` Protocol.
- [src/contingency/entities.py](../src/contingency/entities.py) — value
  objects exchanged across the Protocol.
- [src/contingency/errors.py](../src/contingency/errors.py) — error
  taxonomy.

## Schedule interface

### Python

```python
@runtime_checkable
class Schedule(Protocol):
    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome: ...
    def reset(self) -> None: ...
```

### Rust sketch

```rust
pub trait Schedule {
    fn step(&mut self, now: f64, event: Option<&ResponseEvent>) -> Outcome;
    fn reset(&mut self);
}
```

A mutable borrow is required because schedules are stateful. Implementers
that need interior mutability (e.g. wrappers storing `Box<dyn Schedule>`)
should use `&mut self` at the trait level and keep RNG / counters in
private fields. A dedicated `Box<dyn Schedule>` impl can be provided to
re-dispatch through the inner value.

## Semantic contract

### `step(now, event)`

1. **Monotonic time.** `now` must be `>= last_now` (where `last_now` is
   the `now` passed on the previous call; unset before the first call).
   If `now < last_now - 1e-9` the schedule raises
   `ScheduleStateError`.
2. **Event timestamp consistency.** If `event` is not `None`, then
   `|event.time - now| <= 1e-9`. Otherwise raises
   `ScheduleStateError`.
3. **Single outcome per call.** Exactly one `Outcome` is returned. Even
   if the elapsed time since last step is larger than multiple scheduled
   intervals, at most one reinforcer fires on this step (time-based
   schedules re-anchor to `now` after firing).
4. **Idempotent on `event=None`.** A pure tick may or may not produce a
   reinforcer depending on the schedule family:
   - Ratio schedules (FR, VR, RR): never reinforce on a tick.
   - Interval schedules (FI, VI, RI): never reinforce on a tick.
   - Time-based schedules (FT, VT, RT): may reinforce on a tick once
     the scheduled time has elapsed.
   - Differential schedules (DRO): the resetting variant reinforces on
     a tick after the interval elapses with no intervening response;
     the momentary variant reinforces at interval boundaries.
   - Compound (Concurrent, Multiple, Chained, Tandem, Alternative): may
     reinforce on a tick iff any active component does.
5. **Reinforcer timestamp.** Any emitted `Reinforcer` has
   `Reinforcer.time == now`.
6. **Anchor on first step.** Schedules that track "elapsed time since
   anchor" (FT, VT, RT, DRO) anchor their internal clock at the first
   `now` received. The first step therefore never reinforces.

### `reset()`

Returns the schedule to the state it had immediately after construction.

- RNG state is restored:
  - Schedules owning a seed re-seed from the original seed (VR, VI, VT).
  - Schedules borrowing an `rng: random.Random` snapshot the RNG state
    at construction and restore it (RR, RI, RT).
- Counters are zeroed; sequence cursors rewind; sequence pools are
  regenerated bit-identically if seeded.
- `last_now` and time anchors become `None` (unset).
- Compound schedules propagate `reset()` to every component, then clear
  their own bookkeeping.

`reset()` does **not** change the configured parameters (ratio, mean,
step function, component list, COD value, etc.).

## Value objects

All four are immutable. Rust port should use `#[derive(Clone, Debug, PartialEq)]`.

### `ResponseEvent`

```python
@dataclass(frozen=True)
class ResponseEvent:
    time: float
    operandum: str = "main"
```

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseEvent {
    pub time: f64,
    pub operandum: String,  // default "main"
}
```

- `operandum` identifies the operandum (lever, key, button) that was
  pressed. Used by `Concurrent` to route the event.
- Default `"main"` is used by non-compound schedules; its value is never
  inspected unless a compound schedule routes on it.

### `Observation`

```python
@dataclass(frozen=True)
class Observation:
    time: float
    response_count: int = 0
```

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub time: f64,
    pub response_count: u64,  // default 0
}
```

Observation is a snapshot currently unused by the runtime itself (kept
for future analyser hooks).

### `Reinforcer`

```python
@dataclass(frozen=True)
class Reinforcer:
    time: float
    magnitude: float = 1.0
    label: str = "SR+"
```

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Reinforcer {
    pub time: f64,
    pub magnitude: f64,   // default 1.0
    pub label: String,    // default "SR+"
}
```

- `magnitude` encodes the amount (pellets, ml, points).
- Negative magnitude + `label="SR-"` models aversive control.
- Every schedule in this library emits reinforcers with magnitude
  `1.0` and `label="SR+"` unless an experimental wrapper overrides
  them. Rust should match the defaults.

### `Outcome`

```python
@dataclass(frozen=True)
class Outcome:
    reinforced: bool = False
    reinforcer: Reinforcer | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.reinforced and self.reinforcer is None:
            raise ValueError(...)
        if not self.reinforced and self.reinforcer is not None:
            raise ValueError(...)
```

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub reinforced: bool,
    pub reinforcer: Option<Reinforcer>,
    pub meta: BTreeMap<String, MetaValue>,
}
```

**Invariants enforced at construction**:
- `reinforced == true` iff `reinforcer.is_some()`.

A Rust port should expose only safe constructors that enforce this —
for example:

```rust
impl Outcome {
    pub fn unreinforced() -> Self { ... }
    pub fn unreinforced_with_meta(meta: BTreeMap<String, MetaValue>) -> Self { ... }
    pub fn reinforced(r: Reinforcer) -> Self { ... }
    pub fn reinforced_with_meta(r: Reinforcer, meta: BTreeMap<String, MetaValue>) -> Self { ... }
}
```

`meta` is used by compound schedules to surface which component fired:

- `Concurrent`: sets `cod_suppressed: bool` and `operandum: str` when a
  reinforcer is gated away by COD.
- `Multiple` / `Chained`: sets `current_component: str` to the currently
  active component's stimulus label, and
  `chain_transition: bool` at non-terminal link transitions in
  `Chained`.
- `Tandem`: sets `current_component: int` (the active link index).
- `Alternative`: sets `alternative_winner: "first" | "second"`.

For cross-language fixtures, `meta` values are restricted to JSON
primitives (bool, int, string). Rust should mirror this.

## Error taxonomy

### Python hierarchy

```
ContingencyError (base)
├── ScheduleConfigError (also ValueError)
├── ScheduleStateError (also RuntimeError)
└── HardwareError
    └── NotConnectedError (also RuntimeError)
```

### Rust sketch

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("schedule configuration error: {0}")]
    Config(String),

    #[error("schedule state error: {0}")]
    State(String),

    #[error("hardware error: {0}")]
    Hardware(String),

    #[error("not connected: {0}")]
    NotConnected(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

| Python class | Rust variant | Raised when |
|---|---|---|
| `ScheduleConfigError` | `Error::Config` | Bad constructor params (negative ratio, unknown combinator, unsupported DSL node) |
| `ScheduleStateError` | `Error::State` | `step()` contract violated (non-monotonic time, event/now mismatch) |
| `HardwareError` | `Error::Hardware` | HAL I/O, transport, config, missing dependency |
| `NotConnectedError` | `Error::NotConnected` | HAL read/write before `connect()` or after `disconnect()` |

The `Schedule` trait's `step()` should return `Result<Outcome>` (or
equivalent); Rust code should prefer `Err` over panics for all state
violations.

## Time tolerance

A single constant governs floating-point comparisons:

```python
_TIME_TOL = 1e-9
```

It is applied:

- In monotonic-time checks: `now < last_now - 1e-9` fails.
- In event/now matching: `|event.time - now| > 1e-9` fails.
- In elapsed-time checks: `now - anchor + 1e-9 >= interval` fires.
- In sliding-window eviction (DRH): `w[0] < cutoff - 1e-9` evicts.
- In LimitedHold expiry: `now > arm_time + hold + 1e-9` expires.

Rust should expose a single `const TIME_TOL: f64 = 1e-9;` and apply it
identically. This is load-bearing for the conformance corpus.

## Threading model

The Python implementation is **not** thread-safe. Callers are expected
to serialise access per-schedule. Rust port can trivially be `Send` but
should not be `Sync` without explicit synchronisation — follow the
`&mut self` convention of the trait and let the borrow checker enforce
single-threaded access.
