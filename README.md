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
pip install "contingency-py[gym]"   # optional: Gymnasium environments
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

## Gymnasium environments

With the `gym` extra, `contingency.envs.ScheduleEnv` runs any `Schedule` as a free-operant session for reinforcement-learning agents. Each step advances the clock by `dt` (default 1); action `0` withholds responding and action `i` emits one response on the `i`-th operandum. The reward is the reinforcer's magnitude. The observation holds only what a subject could know: time and responses since the last reinforcer, whether the last step was reinforced, and the last operandum responded on. The episode is truncated at `max_time`.

Three IDs are registered on import: `contingency/FR-v0` (`n`), `contingency/FI-v0` (`interval`) and `contingency/ConcVIVI-v0` (`mean_left`, `mean_right`, `cod`; operanda `left` and `right`). Keyword arguments to `gym.make` override the defaults.

```python
import gymnasium as gym
import contingency.envs  # registers contingency/FR-v0, FI-v0 and ConcVIVI-v0

env = gym.make("contingency/ConcVIVI-v0", mean_left=30.0, mean_right=60.0, cod=2.0)
obs, info = env.reset(seed=0)
env.action_space.seed(0)
n_sr = 0
while True:  # 0 = no response, 1 = left, 2 = right; one step = 1 time unit
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    n_sr += reward > 0
    if terminated or truncated:  # truncated at max_time=600
        break
print(n_sr)  # 21
```

Changeover-delay suppression is reported in `info["outcome_meta"]["cod_suppressed"]`. In `tests/test_envs_concurrent.py`, a tabular Q-learner trained for 100 sessions on the default Conc VI 30 VI 60 (COD 2) earns 23.2 reinforcers per 600-unit session on 10 sessions not used for training, against 20.7 for a uniform random policy and 21.1 for random choice between the two operanda; the same table before training earns 21.5.

## Why not a 50-line hand-written schedule?

A short hand-rolled loop usually gets the following wrong, and each one changes the data you would report:

- **Variable-schedule intervals.** VI/VR values are generated with the Fleshler-Hoffman (1962) progression (the Hantula, 1991, program variant is also available), not ad-hoc `random.uniform` or exponential draws. `n` and `seed` are controllable, so a session can be replayed exactly.
- **Changeover delay in concurrent schedules.** `Conc` routes each response to its own component, holds an arranged reinforcer until the next response on that operandum, suppresses reinforcement for `COD` time units after a changeover, and supports a fixed-ratio changeover requirement (`FRCO`) and direction-specific COD.
- **Conformance fixtures.** 22 JSON fixtures under `conformance/` pin the exact outcome sequence of each schedule for a known event sequence. The Rust port [contingency-rs](https://github.com/OperantKit/contingency-rs) replays the same fixtures, so both runtimes are checked against one specification.
- **Round trip with Method sections.** Because the schedule is written in contingency-dsl, [contingency-dsl2procedure](https://github.com/OperantKit/contingency-dsl2procedure) turns it into a Method-section paragraph and [contingency-procedure2dsl](https://github.com/OperantKit/contingency-procedure2dsl) extracts a schedule from published Method text.
- **Input validation.** Non-monotonic time, malformed parameters, and unknown operanda raise explicit errors instead of silently producing wrong reinforcement times.

## Check your own implementation against the fixtures

`pip install contingency-py` also installs the conformance corpus and the `contingency-conformance` command. Any implementation, in any language, can be checked against the same 22 fixtures:

```bash
pip install contingency-py
contingency-conformance list                    # fixtures, kind, schedule type, steps
contingency-conformance run --only 'atomic/fr_*' -- python my_adapter.py
```

`contingency-conformance path` prints the fixture directory so a harness in another language can read the JSON files directly; `CONTINGENCY_CONFORMANCE_DIR` or `--dir` points the command at another copy.

**Observed JSON.** For each fixture, the implementation reports one entry per fixture step:

```json
{"fixture": "atomic/fr_basic.json",
 "steps": [{"reinforced": false},
           {"reinforced": true, "reinforcer": {"time": 5.0, "magnitude": 1.0, "label": "SR+"}}]}
```

`reinforced` must match exactly; when reinforced, `reinforcer.time` and `magnitude` must agree within 1e-9 and `label` must be equal; every `meta` key in the fixture must be present with the same value (extra keys are allowed); the number of steps must match. Write such files yourself and compare them with `contingency-conformance check observed/*.json`, or let `run` drive an adapter.

**Adapter for `run -- CMD`.** `run` starts CMD once per fixture, writes the fixture JSON to its stdin, and reads the observed JSON from its stdout (`fixture` may be omitted):

```python
# my_adapter.py — a hand-written FR schedule checked against the fixtures
import json
import sys


class MyFR:
    def __init__(self, n):
        self.n, self.count = n, 0

    def step(self, now, event):  # event: None or {"time", "operandum"}
        if event is None:
            return None
        self.count += 1
        if self.count < self.n:
            return None
        self.count = 0
        return {"time": event["time"], "magnitude": 1.0, "label": "SR+"}


def observe(fixture):
    schedule = MyFR(fixture["schedule"]["params"]["n"])
    steps = []
    for step in fixture["steps"]:
        sr = schedule.step(step["now"], step["event"])
        steps.append({"reinforced": True, "reinforcer": sr} if sr else {"reinforced": False})
    return {"implementation": "my-fr", "steps": steps}


if __name__ == "__main__":
    json.dump(observe(json.load(sys.stdin)), sys.stdout)
```

This adapter implements FR only and passes `atomic/fr_basic.json`; without `--only`, the other 21 fixtures are reported as `ERROR` because the adapter cannot build their schedules. To cover them, build each schedule `type` in `observe`. The 5 compound fixtures (Alternative, Chained, Concurrent, Multiple, Tandem) also require `meta` on some steps, such as `current_component`, `alternative_winner`, or `cod_suppressed`; the fixture files show which keys each step expects.

`contingency-conformance replay-python` is the same adapter for this package's own runtime; `contingency-conformance run -- contingency-conformance replay-python` passes all 22 fixtures.

**Seeded fixtures.** Six fixtures (seeded VR, VI, VT, RR, RI, RT) are pinned by Python's `random.Random(seed)`. `--stochastic` chooses how they are judged:

- `count` (default): the number of reinforcers must lie within 0.5-1.5 times the recorded number, and the first reinforcer must not precede the first step. Use this when your implementation has its own random number generator.
- `strict`: judged like the deterministic fixtures. Use this when you reproduce Python's generator.
- `skip`: not checked.

Output is one `PASS` / `FAIL` / `ERROR` / `SKIP` line per fixture with the first mismatch (step, field, expected, observed), then a summary; `--json` gives the same as JSON. The exit code is 0 when nothing failed, 1 when a fixture failed or the adapter crashed, and 2 for input errors (corpus not found, unknown fixture, unreadable file).

## Differences from the Swift original

This package ports the Swift library [OperantKit](https://github.com/YutoMizutani/OperantKit). `tests/test_legacy_swift_parity.py` builds the Swift original from a local checkout and runs it through the fixtures above with a small driver (`tools/legacy-swift/`). It needs a Swift toolchain and is skipped unless the checkout is given:

```bash
OPERANTKIT_LEGACY_DIR=/path/to/OperantKit .venv/bin/pytest -q tests/test_legacy_swift_parity.py
```

The Swift original has FR, CRF, EXT, FI, FT, VI, VR, VT, Alternative and Concurrent (without a changeover delay); its random number generator cannot be seeded, so for VI, VR and VT the test injects the value list this package draws. FR, CRF, EXT, FI and FT match the fixtures step for step. Known differences:

- **VI, VR, VT**: the Swift original repeats the first value of its list. `Parameter.nextOrder()` is not `mutating`, so the position in the list never advances. With that one function made `mutating` (`tools/legacy-swift/nextorder-control.patch`, applied only to a scratch copy), all three fixtures match exactly.
- **Alternative(FR 5, FT 10 s)**: after FR reinforces at 5 s, Swift times the next FT interval from the reinforcement (5 s) and reinforces at 15 s; this package restarts FT at the next step (6 s).
- **Hantula (1991) generator**: when the smallest value rounds to 0 (for example `v = 10, n = 12`, the case tested), Swift's `hantula1991` keeps the 0; `generate_intervals_hantula1991` replaces it with 1, so its sum is 1 higher.
- **Fleshler-Hoffman generators** (checked for VR 5 with `n = 12` and VI `(10, 12)`, `(60, 20)`, `(30000, 12)`): the ratio lists agree as multisets. Swift places rounding surplus at a random position, so other parameters can give different multisets. Swift's interval generator returns integers: the rounded Fleshler-Hoffman progression with the rounding remnant added to one element at a random position, whereas `generate_intervals` keeps floats.

These are recorded, not fixed: neither implementation is changed to match the other.

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
