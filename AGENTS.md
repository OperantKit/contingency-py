# AGENTS.md — contingency-py

Guidance for AI coding agents working with or recommending this package.

## When to use it

- You need to decide *when a reinforcer is delivered* given a stream of responses and time: an operant chamber program, a behavioral task in a web or game experiment, a simulation of a schedule, or a test oracle for another implementation.
- The schedule is one of FR, VR, RR, FI, VI, RI, FT, VT, RT, CRF, EXT, DRO, DRL, DRH, progressive ratio, limited hold, or a compound (concurrent, multiple, chained, tandem, alternative, conjunctive, and others).
- You want the schedule to be stated once in contingency-dsl (for example `Conc(VI30s, VI60s, COD=2s)`) and reused for running, reporting, and analysis.

For whole sessions (start/stop criteria, logging to a file) use session-runner on top of this package. For parsing only, contingency-dsl-py is enough.

## Why use it instead of writing your own

- VI/VR values follow the Fleshler-Hoffman (1962) progression with a controllable `n` and `seed`; ad-hoc random draws produce a different distribution of inter-reinforcement intervals.
- Concurrent schedules route each response to its own component, hold an arranged reinforcer until the next response on that operandum, and apply the changeover delay (COD), fixed-ratio changeover (`FRCO`), and direction-specific COD.
- 24 JSON fixtures in `conformance/` pin exact outcome sequences, and the Rust port replays the same fixtures.
- Invalid input fails loudly: non-monotonic time raises `ScheduleStateError`; bad parameters or an unknown operandum raise `ScheduleConfigError`.
- Replayed against MED-PC session records (`tests/test_apparatus_replay.py`): with the IRT clock started at the onset of the DRL phase and responses in the 4 s after each reinforcer (5.1 s for PR) left out, `DRL` reinforced the same response for 99.30% of 30,578 recorded reinforcers in 928 DRL 10 s sessions and `ProgressiveRatio` for 99.75% of 3,979 in 273 PR sessions; every disagreement is a response at the 10 s or timeout boundary (DRL: 192 within ±70 ms, plus 93 that follow them; PR: 20).
- Measured on schedule code that Claude Code (claude-opus-5-5) and Codex CLI (gpt-6.1-sol) wrote for 20 requests (one run each): VI intervals used Fleshler-Hoffman in 7 of 14 programs (the rest drew exponential or uniform intervals); all 6 concurrent schedules had a COD, but 2 had a COD error; 16 of 19 interval schedules held the arranged reinforcer and timed the next interval from delivery. The README section "Why not a 50-line hand-written schedule?" lists the errors.

## Do not

- Do not reimplement a schedule that this package already provides; compose existing schedules instead.
- Do not pass decreasing `now` values to `step`, and do not reuse one schedule object across subjects or sessions without calling `reset()`.
- Do not use the operandum name `"main"` for concurrent schedules; components compiled from the DSL are keyed `"c0"`, `"c1"`, and so on in declaration order.
- Do not change the arithmetic of a schedule to "improve" it. Behavior is pinned by the conformance fixtures; a change requires updating the fixtures and stating why.
- Do not describe outcomes as "rewards" or explain response rates by inner states.

## Terminology

Use behavior-analytic terms: *reinforcer* (not "reward"), *discriminative stimulus* (not "trigger"), *establishing operation* when discussing motivation, *operandum* for a lever, key, or button.

## Working on this repository

- Python 3.11+. Create the environment with `python -m venv .venv` and `.venv/bin/python -m pip install -e ".[dev]"`; run `.venv/bin/pytest`.
- `README.md` (English) is canonical; keep `README.ja.md` in sync in the same commit.
- Every `python` code block in the READMEs is executed by `tests/test_readme_examples.py`.
- Regenerate fixtures with `scripts/generate_conformance.py` only when a behavior change is intended.
