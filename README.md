# contingency-py

:jp: [日本語版 README](README.ja.md)

contingency-py is a Python library that decides when a reinforcer is delivered. You give it a reinforcement schedule, such as `FR 5`, `VI 60-s`, or `Conc(VI 30-s, VI 60-s, COD=2-s)`, and the responses as they happen; for each step it tells you whether a reinforcer is delivered. Use it in a chamber program, a simulation, a teaching demo, or as a reference to test another implementation against. It is the Python schedule engine of [OperantKit](https://github.com/OperantKit); running a whole session (start and stop criteria, logging) is done by [session-runner](https://github.com/OperantKit/session-runner) on top of it.

- **Variable schedules use the Fleshler–Hoffman (1962) progression,** with the number of values and the seed under your control, so a session can be replayed exactly.
- **Concurrent schedules handle the changeover delay (COD)** — the time after a switch between operanda (levers, keys) during which no response is reinforced — and hold an arranged reinforcer until the next response on its operandum.
- **39 conformance fixtures pin the outcome of each schedule** for a known response sequence. The Rust engine [contingency-rs](https://github.com/OperantKit/contingency-rs) replays them, and you can check your own implementation against them.

[Schedule terms](docs/en/schedule-terms.md) explains these points (COD, the Fleshler–Hoffman progression, DRL and interresponse time, resetting DRO, when a VI interval starts) in a few sentences each. Notes for AI coding agents, including a comparison with schedule code written by AI assistants, are in [AGENTS.md](AGENTS.md).

## Installation

Python 3.11 or newer. contingency-py is not on PyPI yet. Install it and its parser, [contingency-dsl-py](https://github.com/OperantKit/contingency-dsl-py), from GitHub without dependency resolution:

```bash
G=git+https://github.com/OperantKit
pip install --no-deps $G/contingency-dsl-py $G/contingency-py
pip install gymnasium numpy     # optional: only for the Gymnasium environments
```

`pip check` reports a version mismatch between these development builds; it does not affect the examples below.

## Example

Write the schedule as a [contingency-dsl](https://github.com/OperantKit/contingency-dsl) expression, compile it with `from_dsl`, and feed it responses:

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
print(reinforcers)  # {'c0': 16, 'c1': 9} — the same on every run (seeded)
```

The components of a concurrent schedule are named `c0`, `c1`, … in the order they are written.

## What is included

| Area | Contents |
|---|---|
| Simple schedules | FR, VR, RR, FI, VI, RI, FT, VT, RT, CRF, EXT, limited hold |
| Compound schedules | Concurrent (with COD, fixed-ratio changeover, direction-specific COD), Multiple, Mixed, Chained, Tandem, Alternative, Conjunctive, Interpolate, Overlay, second-order |
| Differential and progressive | DRO (resetting, momentary), DRL, DRH, percentile, progressive ratio (arithmetic, geometric, Richardson–Roberts), adjusting, interlocking |
| Other procedures | Timeout, response cost, Sidman avoidance, discriminated avoidance, escape, matching-to-sample, go/no-go |
| Value generators | Fleshler–Hoffman (Fleshler & Hoffman, 1962) and the Hantula (1991) variant |
| Building schedules | `contingency.bridge.from_dsl` (from a contingency-dsl expression), `contingency.builder.ScheduleBuilder` (from Python calls) |
| Hardware abstraction layer (`contingency.hw`) | An `Apparatus` interface with a virtual backend, a serial backend (needs `pyserial`), and a TCP backend for a hardware-in-the-loop (HIL) test rig |
| Conformance | 39 JSON fixtures in `conformance/` and the `contingency-conformance` command |
| Gymnasium | `contingency.envs.ScheduleEnv` and three registered environments (optional) |

```python
from contingency import Schedule, Observation, Reinforcer
from contingency.schedules import FR, VR, FI, VI, FT, VT, CRF, EXT, DRO
from contingency.schedules import Concurrent, Multiple, Chained, Tandem, Alternative
from contingency.builder import ScheduleBuilder
from contingency.bridge import from_dsl
```

### Check your own implementation against the fixtures

The `contingency-conformance` command checks any schedule implementation, in any language, against the 39 fixtures. Your program reads a fixture as JSON and writes back which steps were reinforced:

```bash
contingency-conformance list                    # fixtures, kind, schedule type, steps
contingency-conformance run --only 'atomic/fr_*' -- python my_adapter.py
```

The JSON format, a complete adapter, and how seeded fixtures are judged are in [Checking your own implementation against the fixtures](docs/en/conformance-checker.md).

### Gymnasium environments

`contingency.envs.ScheduleEnv` runs any `Schedule` as a free-operant session for reinforcement-learning agents. Each step advances the clock by `dt` (default 1); action `0` withholds responding and action `i` emits one response on the `i`-th operandum. The `reward` value returned by `step` is the reinforcer's magnitude, minus `response_cost` (default 0) on every step with a response. The observation holds only what a subject could know: time and responses since the last reinforcer, whether the last step was reinforced, and the last operandum responded on. Four IDs are registered on import: `contingency/FR-v0` (`n`), `contingency/FI-v0` (`interval`), `contingency/ConcVIVI-v0` (`mean_left`, `mean_right`, `cod`; operanda `left` and `right`) and `contingency/ConcPhases-v0` (`phases`, `cod`).

`ConcPhases-v0` runs phases one after another in one session, for procedures such as resurgence. Each phase is `(left, right, steps)`: the VI mean in seconds on each operandum (`None` = extinction) and the number of steps it lasts. The default is VI 15 s on `left` for 18000 steps, then `left` in extinction with VI 15 s on `right` for 12000 steps, then both in extinction for 3000 steps. The phase is not in the observation; `info["phase"]` and `info["phase_step"]` report it, and the episode terminates when the last phase ends.

```python
import gymnasium as gym
import contingency.envs  # registers contingency/FR-v0, FI-v0, ConcVIVI-v0, ConcPhases-v0

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

`contingency.envs.tabular` provides an epsilon-greedy tabular Q-learner (`train_q`, `run_greedy`). In `tests/test_envs_concurrent.py`, a table trained for 100 sessions on the default Conc VI 30 VI 60 (COD 2) earns 23.2 reinforcers per 600-unit session on 10 sessions not used for training, against 20.7 for a uniform random policy, 21.1 for random choice between the two operanda (always responding), and 21.5 for the same table before training.

## Limitations

- Not on PyPI yet, and the API can still change (development status: alpha).
- This is a library, not a session program: it has no clock of its own. Your program calls `step` with the current time; timing accuracy depends on that program and on the hardware.
- Eight fixtures (seeded VR, VI, VT, RR, RI, RT, matching-to-sample, go/no-go) are pinned to Python's `random.Random(seed)`. An implementation with another random number generator can match them only approximately.
- The virtual and Gymnasium subjects are models. They show that the program delivers the schedule you wrote, not what an animal will do.

## Replayed against MED-PC session records

`tests/test_apparatus_replay.py` feeds the responses recorded by a MED-PC apparatus in published sessions to `DRL` and `ProgressiveRatio` and checks which responses are reinforced. Over the whole datasets — DRL 10 s, 928 sessions and 30,578 reinforcers (Sosa et al., 2026); progressive ratio 1, 4, 7, …, 273 sessions and 3,979 reinforcers (McCutcheon, 2022) — the engine reinforced the same response for 99.30% (DRL) and 99.75% (PR) of the recorded reinforcers, once the IRT clock is started at the onset of the DRL phase and responses in the 4 s after each reinforcer (5.1 s for PR) are left out, as the apparatus programs do. Every disagreement is a timing-resolution case: for DRL, 192 responses within ±70 ms of the 10 s boundary and 93 responses that follow them; for PR, 20 responses at the boundary of the 5.1 s timeout and the responses that follow them. The test keeps four of these sessions as fixtures.

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

## Development

```bash
git clone https://github.com/OperantKit/contingency-py && cd contingency-py
python -m venv .venv
.venv/bin/python -m pip install --no-deps git+https://github.com/OperantKit/contingency-dsl-py -e .
.venv/bin/python -m pip install pytest pytest-cov hypothesis gymnasium numpy
.venv/bin/pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the code of conduct.

## Support

Report bugs, request features, and ask usage questions in the [issue tracker](https://github.com/OperantKit/contingency-py/issues). For a bug, include the schedule expression (or the code that builds the schedule), the response sequence that reproduces it, the expected and observed outcomes, and the versions of Python, contingency-py, and contingency-dsl-py. The project is maintained by a single developer on a best-effort basis; there is no guaranteed response time, but every issue is read.

## References

- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A simple BASIC program to generate values for variable-interval schedules of reinforcement. *Journal of Applied Behavior Analysis*, 24(4), 799-801. https://doi.org/10.1901/jaba.1991.24-799
- McCutcheon, J. (2022). *Progressive ratio in protein-restricted vs. non-restricted rats* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5409201
- Sosa, R., Mata-Luévanos, J., Tapia-de-Jesús, A., Saavedra, P., & Buenrostro-Jáuregui, M. (2026). *Longitudinal association between different indices of behavioral inhibition* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19164959

## License

MIT. See [LICENSE](LICENSE).
