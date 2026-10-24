# Checking your own implementation against the fixtures

[日本語版](../ja/conformance-checker.md)

Installing contingency-py (see the [README](../../README.md#installation)) also installs the conformance corpus and the `contingency-conformance` command. Any implementation of a reinforcement schedule, in any language, can be checked against the same 39 fixtures:

```bash
contingency-conformance list                    # fixtures, kind, schedule type, steps
contingency-conformance run --only 'atomic/fr_*' -- python my_adapter.py
```

`contingency-conformance path` prints the fixture directory so a harness in another language can read the JSON files directly; `CONTINGENCY_CONFORMANCE_DIR` or `--dir` points the command at another copy.

## Observed JSON

For each fixture, the implementation reports one entry per fixture step:

```json
{"fixture": "atomic/fr_basic.json",
 "steps": [{"reinforced": false},
           {"reinforced": true, "reinforcer": {"time": 5.0, "magnitude": 1.0, "label": "SR+"}}]}
```

`reinforced` must match exactly; when reinforced, `reinforcer.time` and `magnitude` must agree within 1e-9 and `label` must be equal; every `meta` key in the fixture must be present with the same value (extra keys are allowed); the number of steps must match. Write such files yourself and compare them with `contingency-conformance check observed/*.json`, or let `run` drive an adapter.

## Adapter for `run -- CMD`

`run` starts CMD once per fixture, writes the fixture JSON to its stdin, and reads the observed JSON from its stdout (`fixture` may be omitted):

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

This adapter implements FR only and passes `atomic/fr_basic.json`; without `--only`, the other 38 fixtures are reported as `ERROR` because the adapter cannot build their schedules. To cover them, build each schedule `type` in `observe`. The 12 compound fixtures (Alternative, Chained, Concurrent, Conjunctive, Interpolate, Mixed, Multiple, Overlay, SecondOrder, Tandem), the 2 wrapper fixtures (Timeout, ResponseCost), the Percentile fixture, the 3 aversive-control fixtures (Sidman, DiscriminatedAvoidance, Escape), and the 2 trial-based fixtures (MatchingToSample, GoNoGo) also require `meta` on some steps, such as `current_component`, `alternative_winner`, `cod_suppressed`, `timeout_active`, `in_probe`, `net_magnitude`, `phase`, `aversive_event`, or `trial_outcome`; the fixture files show which keys each step expects.

`contingency-conformance replay-python` is the same adapter for this package's own runtime; `contingency-conformance run -- contingency-conformance replay-python` passes all 39 fixtures.

## Seeded fixtures

Eight fixtures (seeded VR, VI, VT, RR, RI, RT, MatchingToSample, GoNoGo) are pinned by Python's `random.Random(seed)`. `--stochastic` chooses how they are judged:

- `count` (default): the number of reinforcers must lie within 0.5-1.5 times the recorded number, and the first reinforcer must not precede the first step. Use this when your implementation has its own random number generator.
- `strict`: judged like the deterministic fixtures. Use this when you reproduce Python's generator.
- `skip`: not checked.

In the two trial-based fixtures the draw decides which comparison is correct (MatchingToSample) and which trials are Go (GoNoGo), so with a different generator the same fixed responses earn a different number of reinforcers and `count` can fail a correct implementation. Unless you reproduce Python's generator, run them with `--stochastic skip` or check them with `strict` after porting the generator.

## Output and exit codes

Output is one `PASS` / `FAIL` / `ERROR` / `SKIP` line per fixture with the first mismatch (step, field, expected, observed), then a summary; `--json` gives the same as JSON. The exit code is 0 when nothing failed, 1 when a fixture failed or the adapter crashed, and 2 for input errors (corpus not found, unknown fixture, unreadable file).
