"""Repair of transcription errors, verified against the mass surface.

Correcting evaluated data is dangerous and this module is off by default.
What makes it defensible is that a repair is never a guess about what a number
*ought* to be. It is a hypothesis about how a specific transcription failed,
drawn from a fixed vocabulary, accepted only when it measurably improves the
data's internal consistency, and recorded in full so the change can be undone
or disputed.

The vocabulary is small because real transcription errors are:

``negate``
    A sign lost or added. ``ensdf.112`` has eleven Q records with S(n) and
    S(p) negated; :sup:`112`\\ Sn reads ``-10788 +/- 5`` where the true value is
    +10.788 MeV.
``scale``
    A power of ten, usually an exponent dropped from a field too narrow to
    hold it. :sup:`173`\\ Ir's S(n) field reads ``1.0960`` where the value is
    10960 keV.

The arbiter is the mass surface. Separation energies and beta Q-values are
differences of the same mass excesses, so

    S(n)[Z, A] - S(n)[Z+1, A] == Q(b-)[Z, A-1] - Q(b-)[Z, A]

holds exactly whatever the masses are. A repair is accepted only if it makes
loops close that did not close before, which is evidence independent of any
opinion about what the value should be. Where no loop constrains a value --
Q(alpha) does not enter this identity -- nothing is repaired, however obviously
wrong the number looks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from .survey import NuclideSummary, loops_touching, mass_surface_loops


def _with(state: NuclideSummary, quantity: str, value: float,
          mark: bool = False) -> NuclideSummary:
    """A copy of state with one Q-value changed."""
    fields: dict = {quantity: value}
    if mark:
        fields["repaired"] = tuple(sorted({*state.repaired, quantity}))
    return replace(state, **fields)

__all__ = ["TRANSFORMS", "Repair", "repair"]

#: Transformations corresponding to known transcription failures. Each is
#: ``(name, function)``; nothing outside this list is ever applied.
TRANSFORMS: tuple[tuple[str, Callable[[float], float]], ...] = (
    ("negate", lambda v: -v),
    ("scale x10", lambda v: v * 10.0),
    ("scale x100", lambda v: v * 100.0),
    ("scale x1000", lambda v: v * 1000.0),
    ("scale x10000", lambda v: v * 10000.0),
    ("scale /10", lambda v: v / 10.0),
    ("scale /100", lambda v: v / 100.0),
)

#: Quantities a mass-surface identity can vouch for. ``q_alpha`` is absent
#: because no identity here constrains it.
_REPAIRABLE = ("s_n", "s_p", "q_beta_minus")


@dataclass(frozen=True)
class Repair:
    """One accepted correction, with the evidence that justified it."""

    nuclide: object
    quantity: str
    original: float
    corrected: float
    transform: str
    residual_before: float
    residual_after: float
    loops_fixed: int

    def __str__(self) -> str:
        return (
            f"{self.nuclide} {self.quantity}: {self.original:g} -> "
            f"{self.corrected:g} keV ({self.transform}); "
            f"loop residual {self.residual_before:.0f} -> "
            f"{self.residual_after:.0f} keV, {self.loops_fixed} loop(s) closed"
        )


def _score(table, loops) -> tuple[int, float]:
    """How many loops close, and the total residual over the rest."""
    closed = sum(1 for loop in loops if loop.closed)
    penalty = sum(loop.residual for loop in loops if not loop.closed)
    return closed, penalty


def _rebuild(table, key) -> list:
    """Only the loops a change at ``key`` can possibly affect."""
    return mass_surface_loops(table, loops_touching(key))


def repair(
    states: Sequence[NuclideSummary],
    max_passes: int = 50,
) -> tuple[list[NuclideSummary], list[Repair]]:
    """Correct transcription errors that the mass surface can confirm.

    Returns the repaired survey and the list of changes. Repaired values are
    marked in :attr:`NuclideSummary.repaired`, so a figure or a downstream
    calculation can tell which numbers are no longer the file's.

    Candidates are considered only for values already participating in a loop
    that fails to close, and only one repair is accepted per pass -- the one
    that closes the most loops. Errors cluster (eleven of ``ensdf.112``'s Q
    records are wrong at once) so several passes are needed before the
    surviving failures can be attributed.

    Nothing is repaired on the strength of looking wrong. If no loop
    constrains a value, it is left alone and remains flagged by
    :func:`~nook.survey.inconsistencies`.
    """
    table = {(s.z, s.nuclide.a): s for s in states}
    if not mass_surface_loops(table):
        return list(states), []

    accepted: list[Repair] = []
    for _ in range(max_passes):
        loops = mass_surface_loops(table)
        suspect = {
            key for loop in loops if not loop.closed for key, _ in loop.terms
        }
        if not suspect:
            break

        best = None
        for key in suspect:
            state = table[key]
            local = _rebuild(table, key)
            closed_base, penalty_base = _score(table, local)
            for quantity in _REPAIRABLE:
                original = getattr(state, quantity)
                if original is None or quantity in state.estimated:
                    continue
                for name, transform in TRANSFORMS:
                    candidate = transform(original)
                    if candidate == original:
                        continue
                    trial = dict(table)
                    trial[key] = _with(state, quantity, candidate)
                    after = _rebuild(trial, key)
                    closed_after, penalty_after = _score(trial, after)
                    gain = closed_after - closed_base
                    if gain <= 0:
                        continue
                    # never trade one closing loop for another: a repair that
                    # breaks agreement elsewhere is not a repair
                    was = {
                        (loop.name, loop.anchor.nuclide)
                        for loop in local if loop.closed
                    }
                    now = {
                        (loop.name, loop.anchor.nuclide)
                        for loop in after if loop.closed
                    }
                    if was - now:
                        continue
                    newly = [
                        loop for loop in after
                        if loop.closed
                        and (loop.name, loop.anchor.nuclide) not in was
                    ]
                    matching = [
                        loop for loop in local
                        if (loop.name, loop.anchor.nuclide)
                        in {(n.name, n.anchor.nuclide) for n in newly}
                    ]
                    score = (gain, penalty_base - penalty_after)
                    if best is None or score > best[0]:
                        best = (
                            score, key, quantity, original, candidate, name,
                            max(loop.residual for loop in matching),
                            max(loop.residual for loop in newly), gain,
                        )
        if best is None:
            break

        _, key, quantity, original, candidate, name, before, after, gain = best
        table[key] = _with(table[key], quantity, candidate, mark=True)
        accepted.append(
            Repair(
                nuclide=table[key].nuclide, quantity=quantity, original=original,
                corrected=candidate, transform=name, residual_before=before,
                residual_after=after, loops_fixed=gain,
            )
        )

    order = {(s.z, s.nuclide.a): i for i, s in enumerate(states)}
    repaired = sorted(table.values(), key=lambda s: order[(s.z, s.nuclide.a)])
    return repaired, accepted
