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

## Public API

```python
from contingency import Schedule, Observation, Reinforcer
from contingency.schedules import FR, VR, FI, VI, FT, VT, CRF, EXT, DRO
from contingency.schedules import Concurrent, Multiple, Chained, Tandem, Alternative
from contingency.builder import ScheduleBuilder
from contingency.bridge import from_dsl
```

## References

- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 56(2), 391-392.
- Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.), *Operant behavior: Areas of research and application* (pp. 213-270). Appleton-Century-Crofts.
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.

## License

MIT. See [LICENSE](LICENSE).
