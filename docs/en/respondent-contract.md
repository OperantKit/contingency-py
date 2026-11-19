# Respondent execution contract

:jp: [日本語版](../ja/respondent-contract.md)

contingency-dsl splits respondent (Pavlovian, CS–US) procedures into two
tiers. Tier A is the small set of primitives in the DSL's respondent layer
(`spec/en/respondent/primitives.md` in contingency-dsl is the list; this page
does not copy it). Tier B is everything built from them across trials or
phases — blocking, renewal, latent inhibition and so on — and is defined by
the companion package contingency-respondent-dsl through the respondent
extension point (contingency-dsl `design-philosophy.md` §5.4).

Execution follows the same split:

| Tier | Syntax lives in | Expansion lives in |
|---|---|---|
| A | contingency-dsl (respondent layer) | contingency-rs, exposed by this package |
| B | contingency-respondent-dsl | contingency-respondent-py, registered per program |

This page fixes what "expanding a Tier A expression" returns, so that
contingency-rs, this package and any other implementation can be checked
against the same files in `conformance/respondent/`.

## The five extension properties on the execution side

contingency-dsl §5.4 lists five properties of the respondent extension point.
Each one has an execution-side reading:

| DSL property | Execution side |
|---|---|
| 1. The respondent grammar is immutable; extensions add productions. | The Tier A expansion is fixed by this contract. An extension never changes how a Tier A expression expands; it only adds expansions for its own primitives. |
| 2. Program-scoped closure: each program's registry decides which extension primitives are recognised. | Tier B expansions are added through a registry built per program (the same pattern as the DSL's `ExtensionRegistry`). There is no global registration; an expression whose name is not in the program's registry is an error, not a silent no-op. |
| 3. Tier A is fully statically verifiable; extensions supply their own rules. | Tier A expansion needs no state from a running session. A Tier B expansion may call the Tier A expansion for the expressions nested in it, and validates its own arguments. |
| 4. Extensions may consume runtime state, but the respondent layer does not add Turing completeness. | The Tier A expansion is a finite list computed from the expression and the program's options. Anything that reacts to responses during a session belongs to the extension or to the session runner. |
| 5. Equivalence of extension constituents is the extension's responsibility. | Two Tier B expressions are judged equivalent by the extension that defines them. The Tier A corpus only pins Tier A outputs. |

## Result of a Tier A expansion

Expanding one Tier A expression gives a list of trials. Each trial is one
JSON object with exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `stage` | string | `"main"` for a bare Tier A expression. Stage names come from the enclosing phase or Tier B procedure. |
| `cs` | list of strings, sorted | CS labels presented on the trial; empty for a US-alone or stimulus-free period. |
| `us` | string or null | US label, or null when no US is delivered. |
| `context` | string or null | Always null for Tier A: no Tier A primitive names a context. |
| `order` | list of strings or null | Presentation order of the CSs for `Serial`; null otherwise. |
| `time` | object | Times within the trial, in seconds from the trial start (below). |
| `iti_s` | number or null | Interval before this trial, in seconds; set only by `ITI`. |

`time` has the keys `cs_on`, `cs_off`, `us_on`, `us_off` and `gap`. A time is
null when the expression does not fix it (for example the CS duration of a
trace pair is left to the apparatus). `gap` is null or an object
`{"from", "to", "s", "at_least"}` that fixes the interval between two of the
other times when their absolute values are open: for example
`{"from": "cs_off", "to": "us_on", "s": 5.0, "at_least": false}` for a 5-s
trace. `at_least: true` means "no shorter than". `from: "any_cs"` is used
only by `ExplicitlyUnpaired`: it fixes the distance between this trial's US
onset and every CS in the list, before or after it, across trials. Durations
are converted to seconds; the unit must be one of `ms`, `s`, `min`, `h`.

### Per element

The trial count is the element's own `trials` argument when it has one, and
otherwise the program's option `trials`.

| Element | Trials | Each trial (or pair of trials) |
|---|---|---|
| `Pair.ForwardDelay` | n | CS (label or simultaneous `Compound`), US; `cs_on` 0, `cs_off` = `cs_duration`, `us_on` = `isi`. |
| `Pair.ForwardTrace` | n | CS, US; `cs_on` 0, `gap` CS offset → US onset = `trace_interval`. With `cs_duration`, `cs_off` and `us_on` are set; without it they are null. |
| `Pair.Simultaneous` | n | CS, US; `cs_on` = `us_on` = 0. |
| `Pair.Backward` | n | CS, US; `us_on` 0, `gap` US offset → CS onset = `isi`. |
| `Extinction`, `CSOnly` | n | CS, no US; `cs_on` 0. |
| `USOnly` | n | no CS, US; `us_on` 0. |
| `Compound` | n | all CSs in `cs_list` together, no US; `cs_on` 0. Only `mode=Simultaneous` is Tier A. |
| `Serial` | n | all CSs, no US; `cs_on` 0 (the first CS); `order` = `cs_list`; `gap` CS offset → next CS onset = `isi`. |
| `Contingency(p, q)` | 2n | n CS periods alternating with n no-CS periods, CS first. One draw `u` per period in that order; the US is delivered when `u < p` (CS period) or `u < q` (no-CS period). CS and US labels come from the program's options. |
| `TrulyRandom(cs, us, p)` | 2n | `Contingency(p, p)` with the named labels. Without `p` the expansion fails with `unresolved`. |
| `ExplicitlyUnpaired` | 2n | CS-alone trial, then US-alone period, n times. With `min_separation`, each US period carries `gap` `{"from": "any_cs", "to": "us_on", "at_least": true}`. |
| `Differential` | 2n | CS+ with the US, then CS− without, n times. The short form takes the US from the program's options and fails with `unresolved` when there is none. |
| `ITI(distribution, mean)` | n | stimulus-free entries carrying `iti_s`: `fixed` = mean; `uniform` = 2·mean·u; `exponential` = −mean·ln(1 − u). |

### Random draws

Each expansion starts a new generator from the integer `seed`, and every
draw `u` is the next value of CPython's `random.Random(seed).random()`. In
terms another implementation can reproduce:

- MT19937 is seeded by `init_by_array` with the 32-bit words of `|seed|`,
  least significant word first (so `-7` and `7` give the same stream; `0`
  is one word `0`).
- `u = ((a >> 5) * 2^26 + (b >> 6)) / 2^53`, where `a` and `b` are the next
  two 32-bit outputs.
- Only integer seeds are part of the contract.

Other implementations use this generator, not one of their own. A draw is
taken for every period even when its probability is 0 or 1, so the stream
position depends only on the trial index.

### Values left to the program

The expression does not carry everything needed to run it. The program
supplies these through the expansion options (in a session they come from
`@cs`, `@us` and `@iti` annotations or the phase):

| Option | Used by |
|---|---|
| `trials` | every element without its own `trials` |
| `seed` | `Contingency`, `TrulyRandom`, `ITI(uniform/exponential)` |
| `labels.cs`, `labels.us` | `Contingency`; `labels.us` for the short form of `Differential` |

A missing value that the element needs is an error
(`{"error": "unresolved", "missing": "<name>"}`, where `<name>` is the
option name above or the element's own argument, such as `p`), never a
default chosen by the implementation.

## The corpus

`conformance/respondent/<name>.json` holds one case per file:

```json
{
  "name": "pair_forward_trace",
  "element": "Pair.ForwardTrace",
  "description": "R2. ... (Ellison, 1964).",
  "spec": {"type": "PairForwardTrace", "cs": "tone", "us": "food",
           "trace_interval": {"value": 5.0, "unit": "s"}},
  "options": {"trials": 2},
  "expected": {"trials": [ ... ]}
}
```

`spec` is the Tier A AST node in contingency-dsl's respondent schema;
`expected` is either `{"trials": [...]}` or the `unresolved` error above.
Every Tier A element has at least one case. The files are written by
`scripts/generate_respondent_conformance.py`, which is the reference reading
of this page; `python scripts/generate_respondent_conformance.py --check`
fails when a file differs from what it would write. The schedule runner
(`contingency-conformance`) skips this directory: these files are not
schedule fixtures.

## Using it from Python

`contingency.respondent.expand(spec, options)` runs a Tier A expression on
contingency-rs and returns the trial list described above;
`RespondentUnresolvedError` carries `missing`. `RespondentRegistry.build(...)`
is the per-program registry of property 2: an expansion declares `name`,
`types` and `expand(spec, options, expand_nested)`, cannot claim a Tier A type
or a type another expansion claims, and must return trials with exactly the
keys above. `tests/test_respondent_expand.py` replays this corpus through it.
