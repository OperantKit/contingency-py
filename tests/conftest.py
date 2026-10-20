"""Shared pytest configuration.

Under mutation testing (mutmut sets ``MUTANT_UNDER_TEST``), the same test
functions run several times in one process and every mutated function goes
through a trampoline. Hypothesis then reports ``differing_executors`` for
``@given`` methods on test classes and may hit its deadline. Neither is a
property of the code under test, so a relaxed profile is loaded only in that
environment. Normal runs keep the Hypothesis defaults.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

if "MUTANT_UNDER_TEST" in os.environ:
    settings.register_profile(
        "mutmut",
        deadline=None,
        suppress_health_check=[HealthCheck.differing_executors, HealthCheck.too_slow],
    )
    settings.load_profile("mutmut")
