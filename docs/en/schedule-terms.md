# Schedule terms: changeover delay, Fleshler-Hoffman, DRL, resetting DRO, VI

:jp: [日本語版](../ja/schedule-terms.md)

Short answers to questions that come up when a schedule of reinforcement is programmed by hand. Each section ends with the contingency-dsl expression that contingency-py runs for it. The example at the end of the page runs the COD, DRL, DRO and VI behavior described here and fails if any of it changes.

## Changeover delay (COD) in concurrent variable-interval schedules

In a concurrent schedule, two or more schedules run at the same time on separate operanda, and the subject can switch between them. A changeover delay (COD) withholds reinforcement for a fixed time after each switch, so that a response made right after switching is not reinforced by the schedule just switched to. A reinforcer that becomes available on the new operandum during the COD is not cancelled: it is held and delivered by the first response after the COD has elapsed. The COD keeps switching itself from being followed closely by reinforcement, and it is part of the procedure in the studies behind the matching law (Herrnstein, 1961; Catania, 1963).

In operantkit: `Conc(VI30s, VI60s, COD=2s)` (add `FRCO=n` for a fixed-ratio changeover requirement).

## Fleshler-Hoffman progression for variable-interval schedules

A variable-interval (VI) schedule needs a list of intervals whose mean equals the nominal value, for example 60 s for VI 60 s. Fleshler and Hoffman (1962) gave a progression that produces such a list so that the probability of reinforcement stays roughly constant over the time since the last reinforcer. Drawing each interval from a uniform or exponential distribution gives a different distribution of intervals and does not reproduce the published procedure. The same progression is used for variable-ratio (VR) requirements, rounded to whole responses.

In operantkit: `VI60s` already draws its intervals from the Fleshler-Hoffman progression (12 values by default); `@algorithm("fleshler-hoffman", n=12, seed=42)` before it fixes the number of values and the seed so a session can be replayed. `contingency.helpers.fleshler_hoffman.generate_intervals(60, n=12, seed=42)` returns the same values as a list.

## DRL: each response resets the interresponse-time timer

Differential reinforcement of low rate (DRL) reinforces a response only when the time since the previous response, the interresponse time (IRT), is at least the schedule value; the first response of a session, which has no previous response, is reinforced. Every response restarts the timer, whether or not it was reinforced, so a response made too early postpones the next possible reinforcer. Waiting without responding does not by itself produce a reinforcer; a response is required (Kramer & Rilling, 1970).

In operantkit: `DRL20s`.

## Resetting DRO: every response resets the interval

Differential reinforcement of other behavior (DRO) delivers a reinforcer when the target response has not occurred for the whole interval. In the resetting form, each target response restarts the interval, and after a reinforcer the next interval starts at once. The reinforcer is delivered by the passage of time, not by a response, so the timer has to be checked even when nothing is emitted. DRO procedures also differ in whether the interval is fixed or variable and in whether the whole interval or only its last moment is checked (Lindberg et al., 1999).

In operantkit: `DRO10s` (resetting, whole interval, fixed).

## Variable-interval schedule: the reinforcer is held, and the next interval starts after reinforcement

In a VI schedule, the first response after the current interval has elapsed is reinforced. Responses made before then have no programmed consequence and do not restart the interval. Once the interval has elapsed, the reinforcer stays available until the next response, however long that takes. The next interval is timed from the delivery of that reinforcer (Ferster & Skinner, 1957; Catania & Reynolds, 1968).

In operantkit: `VI60s`.

## Runnable check

```python
from contingency_dsl import parse
from contingency_dsl.annotations import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry
from contingency import ResponseEvent, from_dsl

registry = ExtensionRegistry.build(TemporalExtension())


def compile_(source):
    return from_dsl(parse(source, extensions=registry))


def respond(schedule, now, operandum="main"):
    return schedule.step(now, ResponseEvent(time=now, operandum=operandum))


# Changeover delay: the switch at 4 s opens a 2-s COD on c1. The FI 3 s on c1
# has already elapsed, so its reinforcer is held and delivered at 6.5 s.
conc = compile_("Conc(FI3s, FI3s, COD=2s)")
assert not respond(conc, 1.0, "c0").reinforced
assert respond(conc, 4.0, "c1").meta["cod_suppressed"]
assert respond(conc, 5.0, "c1").meta["cod_suppressed"]
assert respond(conc, 6.5, "c1").reinforced

# DRL 20 s: responses at 10 s and 50 s come too early and restart the timer.
drl = compile_("DRL20s")
times = [0.0, 10.0, 35.0, 50.0, 75.0]
assert [t for t in times if respond(drl, t).reinforced] == [0.0, 35.0, 75.0]

# Resetting DRO 10 s: the response at 4 s moves the first reinforcer to 14 s.
dro = compile_("DRO10s")
delivered = []
for t in range(31):
    event = ResponseEvent(time=float(t), operandum="main") if t == 4 else None
    if dro.step(float(t), event).reinforced:
        delivered.append(t)
assert delivered == [14, 24]

# VI 60 s: the reinforcer waits for the first response, whether it comes at
# 500 s or at 700 s, and the next interval is timed from that delivery, so the
# second reinforcer moves by the same 200 s.
def second_reinforcer(first_response):
    vi = compile_('@algorithm("fleshler-hoffman", n=12, seed=42)\nVI60s')
    assert respond(vi, float(first_response)).reinforced
    t = first_response + 1
    while not respond(vi, float(t)).reinforced:
        t += 1
    return t


assert second_reinforcer(700) - second_reinforcer(500) == 200
```

## References

- Catania, A. C. (1963). Concurrent performances: Reinforcement interaction and response independence. *Journal of the Experimental Analysis of Behavior*, 6(2), 253-263. https://doi.org/10.1901/jeab.1963.6-253
- Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis of the responding maintained by interval schedules of reinforcement. *Journal of the Experimental Analysis of Behavior*, 11(3, Pt. 2), 327-383. https://doi.org/10.1901/jeab.1968.11-s327
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Herrnstein, R. J. (1961). Relative and absolute strength of response as a function of frequency of reinforcement. *Journal of the Experimental Analysis of Behavior*, 4(3), 267-272. https://doi.org/10.1901/jeab.1961.4-267
- Kramer, T. J., & Rilling, M. (1970). Differential reinforcement of low rates: A selective critique. *Psychological Bulletin*, 74(4), 225-254. https://doi.org/10.1037/h0029813
- Lindberg, J. S., Iwata, B. A., Kahng, S., & DeLeon, I. G. (1999). DRO contingencies: An analysis of variable-momentary schedules. *Journal of Applied Behavior Analysis*, 32(2), 123-136. https://doi.org/10.1901/jaba.1999.32-123
