"""Respondent (Pavlovian, CS–US) procedures as trial lists.

contingency-dsl splits respondent procedures into two tiers, and execution
follows the same split (``docs/en/respondent-contract.md``):

* **Tier A** — the primitives of the DSL's respondent layer
  (``Pair.ForwardDelay``, ``Extinction``, ``Contingency`` and the rest). Their
  expansion into trials is fixed by the contract and computed by
  contingency-rs through the ``contingency_core`` extension module.
  :func:`expand` runs one Tier A expression.
* **Tier B** — procedures built from Tier A across trials or phases
  (blocking, renewal and so on), defined by extensions such as
  contingency-respondent-dsl. Their expansions are added per program
  through a :class:`RespondentRegistry`, the same pattern as the DSL's
  ``ExtensionRegistry``: there is no global registration, and a type that
  is neither Tier A nor in the program's registry is an error.

A trial is a dict with the keys ``stage``, ``cs``, ``us``, ``context``,
``order``, ``time`` and ``iti_s``; the contract describes each one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from contingency.engine import _core
from contingency.errors import (
    RespondentSpecError,
    RespondentUnresolvedError,
    UnknownRespondentElementError,
)

__all__ = [
    "TIER_A_TYPES",
    "TRIAL_KEYS",
    "ExpandNested",
    "RespondentExpansion",
    "RespondentRegistry",
    "RespondentSpecError",
    "RespondentUnresolvedError",
    "UnknownRespondentElementError",
    "expand",
]

#: The AST ``type`` of every Tier A element of contingency-dsl's respondent
#: layer, as contingency-rs reads them. ``conformance/respondent/`` has at
#: least one case for each.
TIER_A_TYPES: frozenset[str] = frozenset(
    {
        "PairForwardDelay",
        "PairForwardTrace",
        "PairSimultaneous",
        "PairBackward",
        "Extinction",
        "CSOnly",
        "USOnly",
        "Contingency",
        "TrulyRandom",
        "ExplicitlyUnpaired",
        "Compound",
        "Serial",
        "ITI",
        "Differential",
    }
)

#: The keys of every trial, Tier A or Tier B (the contract's trial shape).
TRIAL_KEYS: frozenset[str] = frozenset({"stage", "cs", "us", "context", "order", "time", "iti_s"})

Trial = dict[str, Any]
ExpandNested = Callable[[Mapping[str, Any], Mapping[str, Any]], list[Trial]]
"""The callable a Tier B expansion receives to expand nested expressions."""


class RespondentExpansion(Protocol):
    """A Tier B expansion registered in a :class:`RespondentRegistry`.

    ``name`` identifies the expansion; ``types`` are the AST ``type`` values
    it expands (none of them Tier A). ``expand`` returns the trial list of
    one expression. It validates its own arguments and calls
    ``expand_nested`` for the expressions nested in it, which goes back
    through the same registry (Tier A to contingency-rs, Tier B to their
    owners).
    """

    name: str
    types: frozenset[str]

    def expand(
        self,
        spec: Mapping[str, Any],
        options: Mapping[str, Any],
        expand_nested: ExpandNested,
    ) -> list[Trial]: ...


def _as_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RespondentSpecError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RespondentSpecError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _type_of(spec: Mapping[str, Any]) -> str:
    type_name = spec.get("type")
    if not isinstance(type_name, str):
        raise RespondentSpecError("a respondent expression needs a string 'type'")
    return type_name


def _plain(value: Any) -> Any:
    """Copy nested mappings and sequences into plain dicts and lists."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _expand_tier_a(spec: Mapping[str, Any], options: Mapping[str, Any]) -> list[Trial]:
    core = _core()
    try:
        trials: list[Trial] = core.expand_respondent(_plain(spec), _plain(options))
    except core.UnresolvedError as exc:
        raise RespondentUnresolvedError(str(getattr(exc, "missing", ""))) from exc
    except ValueError as exc:
        raise RespondentSpecError(str(exc)) from exc
    return trials


def _check_expansion(ext: Any) -> frozenset[str]:
    name = getattr(ext, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"respondent expansion {ext!r} has no string 'name'")
    if not callable(getattr(ext, "expand", None)):
        raise ValueError(f"respondent expansion '{name}' has no callable 'expand'")
    types = getattr(ext, "types", None)
    if isinstance(types, str) or not isinstance(types, (set, frozenset)):
        raise ValueError(f"respondent expansion '{name}' must declare 'types' as a set of strings")
    if not types:
        raise ValueError(f"respondent expansion '{name}' declares no types")
    if not all(isinstance(t, str) and t for t in types):
        raise ValueError(f"respondent expansion '{name}' declares a type that is not a string")
    return frozenset(types)


def _check_trials(owner: str, trials: Any) -> list[Trial]:
    if not isinstance(trials, list):
        raise RespondentSpecError(
            f"respondent expansion '{owner}' returned {type(trials).__name__}, not a list of trials"
        )
    for i, trial in enumerate(trials):
        if not isinstance(trial, dict) or set(trial) != TRIAL_KEYS:
            raise RespondentSpecError(
                f"respondent expansion '{owner}' returned trial {i} without exactly the keys "
                f"{sorted(TRIAL_KEYS)}"
            )
    return trials


def expand(
    spec: Mapping[str, Any] | str, options: Mapping[str, Any] | str | None = None
) -> list[Trial]:
    """Expand one Tier A respondent expression into its trial list.

    Parameters
    ----------
    spec:
        A contingency-dsl respondent AST node (a mapping or a JSON string),
        such as ``{"type": "PairForwardDelay", "cs": "tone", "us": "shock",
        "isi": {"value": 10, "unit": "s"}}``.
    options:
        The values left to the program: ``trials``, ``seed`` and
        ``labels`` (``{"cs": ..., "us": ...}``). Omit it when the
        expression carries everything it needs.

    Raises
    ------
    RespondentUnresolvedError
        When the element needs a value that neither the expression nor
        ``options`` supplies; ``missing`` names it. No default is chosen.
    UnknownRespondentElementError
        When ``spec`` is not a Tier A element. Tier B procedures go through
        a :class:`RespondentRegistry`.
    RespondentSpecError
        When ``spec`` or ``options`` is malformed.
    ImportError
        When ``contingency_core`` is not installed.
    """
    return RespondentRegistry.build().expand(spec, options)


@dataclass(frozen=True)
class RespondentRegistry:
    """The respondent expansions one program recognises.

    Build one per program with :meth:`build`. Tier A types always expand on
    contingency-rs and cannot be claimed by an expansion; each Tier B type
    belongs to exactly one registered expansion.
    """

    _owners: Mapping[str, RespondentExpansion] = field(default_factory=lambda: MappingProxyType({}))
    names: frozenset[str] = frozenset()

    @staticmethod
    def build(*expansions: RespondentExpansion) -> RespondentRegistry:
        """Create a registry from Tier B expansions.

        Raises ``ValueError`` when an expansion lacks a string ``name``, a
        callable ``expand`` or a non-empty set of string ``types``, when two
        expansions share a name, or when an expansion claims a Tier A type
        or a type another expansion already claims.
        """
        owners: dict[str, RespondentExpansion] = {}
        names: set[str] = set()
        for ext in expansions:
            types = _check_expansion(ext)
            if ext.name in names:
                raise ValueError(f"duplicate respondent expansion name: '{ext.name}'")
            names.add(ext.name)
            for type_name in sorted(types):
                if type_name in TIER_A_TYPES:
                    raise ValueError(
                        f"respondent expansion '{ext.name}' claims '{type_name}', "
                        "a Tier A type; Tier A expansion is fixed by the contract"
                    )
                if type_name in owners:
                    raise ValueError(
                        f"respondent type '{type_name}' claimed by both "
                        f"'{owners[type_name].name}' and '{ext.name}'"
                    )
                owners[type_name] = ext
        return RespondentRegistry(_owners=MappingProxyType(owners), names=frozenset(names))

    @property
    def types(self) -> frozenset[str]:
        """Every type this registry expands: Tier A plus the registered ones."""
        return TIER_A_TYPES | frozenset(self._owners)

    def expand(
        self,
        spec: Mapping[str, Any] | str,
        options: Mapping[str, Any] | str | None = None,
    ) -> list[Trial]:
        """Expand one respondent expression (Tier A or registered Tier B).

        Raises the same errors as :func:`expand`;
        :class:`UnknownRespondentElementError` when the type is neither Tier
        A nor registered here, and :class:`RespondentSpecError` when a Tier
        B expansion returns something other than a list of trials with the
        contract's keys, or nests itself without end.
        """
        spec_map = _as_mapping(spec, "spec")
        options_map = {} if options is None else _as_mapping(options, "options")
        try:
            return self._expand(spec_map, options_map)
        except RecursionError as exc:
            raise RespondentSpecError(
                "respondent expansion nested without end (a Tier B expansion expands itself)"
            ) from exc

    def _expand(self, spec: Mapping[str, Any], options: Mapping[str, Any]) -> list[Trial]:
        type_name = _type_of(spec)
        if type_name in TIER_A_TYPES:
            return _expand_tier_a(spec, options)
        owner = self._owners.get(type_name)
        if owner is None:
            raise UnknownRespondentElementError(type_name)
        return _check_trials(owner.name, owner.expand(spec, options, self._expand_nested))

    def _expand_nested(self, spec: Mapping[str, Any], options: Mapping[str, Any]) -> list[Trial]:
        return self._expand(_as_mapping(spec, "nested spec"), _as_mapping(options, "options"))
