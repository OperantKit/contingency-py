# contingency-py

:jp: [日本語版 README](README.ja.md)

Python reinforcement schedule engine for operant contingencies.

Bridges [`contingency-dsl`](../contingency-dsl/) AST nodes into executable `Schedule` runtime objects. Pure library with no session/context/renewal concerns — those live in [`experiment-core`](../../experiment/experiment-core/).

## Scope

- Atomic reinforcement schedules: FR, VR, RR, FI, VI, RI, FT, VT, RT, CRF, EXT
- Compound schedules: Concurrent (+ COD), Multiple, Chained, Tandem, Alternative
- Differential: DRO (resetting, momentary), DRL, DRH
- Progressive Ratio (arithmetic, geometric, Hantula step functions)
- Limited Hold wrapper
- Fleshler-Hoffman VI/VR distribution generator (Fleshler & Hoffman, 1962; Hantula, 1991)
- DSL bridge: `contingency.bridge.from_dsl(ast)` → `Schedule`
- HAL (`contingency.hw`): virtual / serial / HIL bridge backends

## Install

```bash
pip install contingency-py
```

For development:

```
mise install                                # Python 3.11
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

## 30-second example

Write the schedule in [contingency-dsl](https://github.com/OperantKit/contingency-dsl), compile it with `from_dsl`, and drive it with responses:

```python
from contingency_dsl import parse
from contingency_dsl.annotations import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry
from contingency import ResponseEvent, from_dsl

registry = ExtensionRegistry.build(TemporalExtension())
source = '@algorithm("fleshler-hoffman", n=12, seed=42)\nConc(VI30s, VI60s, COD=2s)'
schedule = from_dsl(parse(source, extensions=registry))

# Simulated subject: one response every 0.5 s, switching operanda every 10 s, for 10 min.
reinforcers = {"c0": 0, "c1": 0}
for i in range(1, 1201):
    now = i * 0.5
    operandum = "c0" if (i // 20) % 2 == 0 else "c1"
    outcome = schedule.step(now, ResponseEvent(time=now, operandum=operandum))
    if outcome.reinforced:
        reinforcers[operandum] += 1
print(reinforcers)  # {'c0': 9, 'c1': 3} — identical on every run (seeded)
```

`contingency-dsl-py` is installed as a dependency. The same `Schedule` object runs inside a full session via [session-runner](https://github.com/OperantKit/session-runner).

## Why not a 50-line hand-written schedule?

A short hand-rolled loop usually gets the following wrong, and each one changes the data you would report:

- **Variable-schedule intervals.** VI/VR values are generated with the Fleshler-Hoffman (1962) progression (the Hantula, 1991, program variant is also available), not ad-hoc `random.uniform` or exponential draws. `n` and `seed` are controllable, so a session can be replayed exactly.
- **Changeover delay in concurrent schedules.** `Conc` routes each response to its own component, holds an arranged reinforcer until the next response on that operandum, suppresses reinforcement for `COD` time units after a changeover, and supports a fixed-ratio changeover requirement (`FRCO`) and direction-specific COD.
- **Conformance fixtures.** 22 JSON fixtures under `conformance/` pin the exact outcome sequence of each schedule for a known event sequence. The Rust port [contingency-rs](https://github.com/OperantKit/contingency-rs) replays the same fixtures, so both runtimes are checked against one specification.
- **Round trip with Method sections.** Because the schedule is written in contingency-dsl, [contingency-dsl2procedure](https://github.com/OperantKit/contingency-dsl2procedure) turns it into a Method-section paragraph and [contingency-procedure2dsl](https://github.com/OperantKit/contingency-procedure2dsl) extracts a schedule from published Method text.
- **Input validation.** Non-monotonic time, malformed parameters, and unknown operanda raise explicit errors instead of silently producing wrong reinforcement times.

## Public API

```python
from contingency import Schedule, Observation, Reinforcer
from contingency.schedules import FR, VR, FI, VI, FT, VT, CRF, EXT, DRO
from contingency.schedules import Concurrent, Multiple, Chained, Tandem, Alternative
from contingency.builder import ScheduleBuilder
from contingency.bridge import from_dsl
```

## Support

- Report bugs, request features, and ask usage questions in the [issue tracker](https://github.com/OperantKit/contingency-py/issues). For a bug, include the schedule expression (or the code that builds the schedule), the response/event sequence that reproduces it, the expected and observed outcomes, and the versions of Python, contingency-py, and contingency-dsl-py.
- The project is maintained by a single developer on a best-effort basis. There is no guaranteed response time, but every issue is read.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute changes and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the code of conduct.

## References

- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 56(2), 391-392.
- Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.), *Operant behavior: Areas of research and application* (pp. 213-270). Appleton-Century-Crofts.
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.

## License

MIT. See [LICENSE](LICENSE).
