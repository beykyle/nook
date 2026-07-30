"""Cross-source comparison: levels and masses where the backends overlap.

Two questions this module answers:

* ``levels()`` -- RIPL's discrete levels *derive from* ENSDF, so matching a
  RIPL scheme against an ENSDF one (or Livechart's adopted levels) checks a
  parsing chain end to end, and the disagreements that remain are real:
  levels RIPL dropped, spins RIPL invented, and the two completeness
  cutoffs -- RIPL's evaluated ``Nmax`` against the heuristic
  :meth:`~nook.model.LevelScheme.complete_up_to`.
* ``masses()`` -- the same nuclide's mass excess across evaluations: AME
  via Livechart, Audi-as-shipped via RIPL, and the FRDM95/HFB-14 theory
  values.  ``sigma_between`` says whether two experimental values actually
  disagree; theory-minus-experiment is a model residual, not a discrepancy.

Level matching is energy-sorted greedy nearest-neighbour, one-to-one,
within ``max(tolerance_kev, 3 * combined sigma)``.  Ground states always
match each other.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field as dataclass_field

from .model import Level, LevelScheme
from .nuclide import Nuclide
from .quantities import Uncertain

__all__ = ["LevelComparison", "LevelMatch", "MassComparison", "levels", "masses", "mass_table"]


@dataclass(frozen=True)
class LevelMatch:
    """One matched pair of levels."""

    a: Level
    b: Level
    delta_kev: float
    combined_sigma_kev: float | None
    #: True/False when both sides have candidates; None when either is unknown.
    jpi_agree: bool | None

    @property
    def significant(self) -> bool:
        """The energies disagree beyond 3 combined sigma."""
        if self.combined_sigma_kev is None or self.combined_sigma_kev == 0.0:
            return abs(self.delta_kev) > 1.0
        return abs(self.delta_kev) > 3.0 * self.combined_sigma_kev


@dataclass(frozen=True)
class LevelComparison:
    nuclide: Nuclide
    source_a: str
    source_b: str
    matched: tuple[LevelMatch, ...]
    only_a: tuple[Level, ...]
    only_b: tuple[Level, ...]
    #: Levels in the heuristically complete part of scheme a.
    cutoff_a: int | None = None
    #: RIPL's own Nmax for scheme b, when b is RIPL.
    cutoff_b: int | None = None

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    @property
    def rms_delta_kev(self) -> float | None:
        if not self.matched:
            return None
        return math.sqrt(
            sum(m.delta_kev ** 2 for m in self.matched) / len(self.matched)
        )

    @property
    def max_delta_kev(self) -> float | None:
        return max((abs(m.delta_kev) for m in self.matched), default=None)

    @property
    def jpi_agreement_fraction(self) -> float | None:
        judged = [m.jpi_agree for m in self.matched if m.jpi_agree is not None]
        if not judged:
            return None
        return sum(judged) / len(judged)

    def to_records(self) -> list[dict]:
        out = []
        for m in self.matched:
            out.append({
                "kind": "matched",
                "energy_a_kev": m.a.energy_kev,
                "energy_b_kev": m.b.energy_kev,
                "delta_kev": m.delta_kev,
                "combined_sigma_kev": m.combined_sigma_kev,
                "significant": m.significant,
                "jpi_a": str(m.a.spin_parity),
                "jpi_b": str(m.b.spin_parity),
                "jpi_agree": m.jpi_agree,
            })
        for side, levels_ in (("only_a", self.only_a), ("only_b", self.only_b)):
            for lv in levels_:
                out.append({
                    "kind": side,
                    "energy_kev": lv.energy_kev,
                    "jpi": str(lv.spin_parity),
                })
        return out

    def __str__(self) -> str:
        parts = [
            f"{self.nuclide}: {self.n_matched} matched "
            f"({self.source_a} vs {self.source_b})",
            f"only {self.source_a}: {len(self.only_a)}",
            f"only {self.source_b}: {len(self.only_b)}",
        ]
        if self.rms_delta_kev is not None:
            parts.append(f"rms dE {self.rms_delta_kev:.3g} keV")
        if self.jpi_agreement_fraction is not None:
            parts.append(f"Jpi agree {100 * self.jpi_agreement_fraction:.0f}%")
        if self.cutoff_a is not None or self.cutoff_b is not None:
            parts.append(f"complete: {self.cutoff_a} vs Nmax {self.cutoff_b}")
        return "  ".join(parts)


def _jpi_agree(a: Level, b: Level) -> bool | None:
    sa, sb = a.spin_parity, b.spin_parity
    if not sa.candidates or not sb.candidates:
        return None
    for ja in sa.candidates:
        for jb in sb.candidates:
            if ja.two_j != jb.two_j:
                continue
            if ja.parity is None or jb.parity is None or ja.parity == jb.parity:
                return True
    return False


def _combined_sigma(a: Level, b: Level) -> float | None:
    sa, sb = a.energy.symmetric, b.energy.symmetric
    if sa is None and sb is None:
        return None
    return math.hypot(sa or 0.0, sb or 0.0)


def _match_levels(
    levels_a: tuple[Level, ...],
    levels_b: tuple[Level, ...],
    tolerance_kev: float,
) -> tuple[list[tuple[Level, Level]], list[Level], list[Level]]:
    """Greedy nearest-neighbour, one-to-one, closest pairs first."""
    # Both schemes are energy-sorted, so instead of the full A x B cross
    # product, bisect an over-wide window into b for each level in a: the
    # per-pair window max(tol, 3*hypot(sa, sb)) never exceeds
    # max(tol, 3*(sa + max_sb)), so no accepted pair is missed.
    sorted_b = sorted(
        (lb.energy_kev, j)
        for j, lb in enumerate(levels_b)
        if lb.energy_kev is not None
    )
    energies_b = [energy for energy, _ in sorted_b]
    max_sigma_b = max(
        (lb.energy.symmetric or 0.0 for lb in levels_b if lb.energy_kev is not None),
        default=0.0,
    )
    candidates: list[tuple[float, int, int]] = []
    for i, la in enumerate(levels_a):
        if la.energy_kev is None:
            continue
        sigma_a = la.energy.symmetric or 0.0
        wide = max(tolerance_kev, 3.0 * (sigma_a + max_sigma_b))
        lo = bisect.bisect_left(energies_b, la.energy_kev - wide)
        hi = bisect.bisect_right(energies_b, la.energy_kev + wide)
        for energy_b, j in sorted_b[lo:hi]:
            lb = levels_b[j]
            delta = abs(la.energy_kev - energy_b)
            sigma = _combined_sigma(la, lb)
            window = max(tolerance_kev, 3.0 * sigma) if sigma else tolerance_kev
            # ground states pair regardless of window (both are E=0 anchors)
            if delta <= window or (la.energy_kev == 0.0 and lb.energy_kev == 0.0):
                candidates.append((delta, i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[Level, Level]] = []
    for _, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((levels_a[i], levels_b[j]))
    pairs.sort(key=lambda pair: pair[0].energy_kev or 0.0)
    only_a = [lv for i, lv in enumerate(levels_a) if i not in used_a]
    only_b = [lv for j, lv in enumerate(levels_b) if j not in used_b]
    return pairs, only_a, only_b


def _fetch(nuclide, source: str, path, ripl_path, cache) -> LevelScheme:
    from . import SOURCE_ALIASES, level_scheme

    canon = SOURCE_ALIASES.get(source, source)
    return level_scheme(
        nuclide,
        source=canon,
        path=ripl_path if canon == "ripl3" else path,
        cache=cache,
    )


def levels(
    nuclide,
    sources: tuple[str, str] = ("file", "ripl3"),
    below: float | None = None,
    tolerance_kev: float = 2.0,
    path=None,
    ripl_path=None,
    cache=None,
) -> LevelComparison:
    """Match one nuclide's level scheme across two sources.

    ``sources`` names any two of ``"file"``, ``"ripl3"``, ``"livechart"``.
    ``below`` truncates both schemes first; floating (offset) levels are
    dropped because their absolute position is undefined.
    """
    nuc = Nuclide.parse(nuclide)
    source_a, source_b = sources
    scheme_a = _fetch(nuc, source_a, path, ripl_path, cache)
    scheme_b = _fetch(nuc, source_b, path, ripl_path, cache)
    if below is not None:
        scheme_a = scheme_a.below(below)
        scheme_b = scheme_b.below(below)
    else:
        scheme_a = scheme_a.filter(lambda lv: not lv.is_floating)
        scheme_b = scheme_b.filter(lambda lv: not lv.is_floating)

    pairs, only_a, only_b = _match_levels(
        scheme_a.levels, scheme_b.levels, tolerance_kev
    )
    matched = tuple(
        LevelMatch(
            a=la,
            b=lb,
            delta_kev=(la.energy_kev or 0.0) - (lb.energy_kev or 0.0),
            combined_sigma_kev=_combined_sigma(la, lb),
            jpi_agree=_jpi_agree(la, lb),
        )
        for la, lb in pairs
    )

    return LevelComparison(
        nuclide=nuc,
        source_a=scheme_a.source,
        source_b=scheme_b.source,
        matched=matched,
        only_a=tuple(only_a),
        only_b=tuple(only_b),
        cutoff_a=len(scheme_a.complete_up_to().levels),
        # Strictly scheme b's own Nmax: falling back to scheme a's here would
        # print a's cutoff as if it described b whenever the RIPL scheme is
        # passed first.
        cutoff_b=scheme_b.metadata.get("nmax"),
    )


# --------------------------------------------------------------------------
# masses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MassComparison:
    """One nuclide's mass excess across evaluations, keV.

    ``entries`` keys: ``"ame-livechart"``, ``"ripl-exp"`` (experimental) and
    ``"frdm95"``, ``"hfb14"`` (theory).
    """

    nuclide: Nuclide
    entries: dict[str, Uncertain] = dataclass_field(default_factory=dict)

    @property
    def experimental(self) -> dict[str, Uncertain]:
        return {k: v for k, v in self.entries.items() if k in ("ame-livechart", "ripl-exp")}

    @property
    def spread_kev(self) -> float | None:
        """Max minus min over every evaluation that has a value."""
        values = [u.value for u in self.entries.values() if u.value is not None]
        if len(values) < 2:
            return None
        return max(values) - min(values)

    def sigma_between(self, key_a: str, key_b: str) -> float | None:
        """|difference| / combined sigma between two entries."""
        ua, ub = self.entries.get(key_a), self.entries.get(key_b)
        if ua is None or ub is None or ua.value is None or ub.value is None:
            return None
        sigma = math.hypot(ua.symmetric or 0.0, ub.symmetric or 0.0)
        if sigma == 0.0:
            return None
        return abs(ua.value - ub.value) / sigma

    def residual_kev(self, theory: str = "frdm95") -> float | None:
        """Theory minus experiment (RIPL experimental value)."""
        exp = self.entries.get("ripl-exp") or self.entries.get("ame-livechart")
        th = self.entries.get(theory)
        if exp is None or th is None or exp.value is None or th.value is None:
            return None
        return th.value - exp.value

    def __str__(self) -> str:
        rows = [f"{self.nuclide} mass excess [keV]"]
        for key in ("ame-livechart", "ripl-exp", "frdm95", "hfb14"):
            if key in self.entries and self.entries[key].value is not None:
                rows.append(f"  {key:<14} {self.entries[key]}")
        if self.spread_kev is not None:
            rows.append(f"  spread         {self.spread_kev:.1f}")
        return "\n".join(rows)


def masses(nuclide, cache=None, ripl_path=None, with_livechart: bool = True) -> MassComparison:
    """One nuclide's mass excess across every evaluation we can reach.

    ``with_livechart=False`` skips the network/API entry and compares the
    local RIPL tables only.
    """
    from .sources.ripl3 import Ripl3Source

    nuc = Nuclide.parse(nuclide)
    entries: dict[str, Uncertain] = {}
    entry = Ripl3Source(path=ripl_path).masses(nuc)
    if entry.mass_excess_exp.value is not None:
        entries["ripl-exp"] = entry.mass_excess_exp
    if entry.mass_excess_frdm95 is not None:
        entries["frdm95"] = entry.mass_excess_frdm95
    if entry.mass_excess_hfb14 is not None:
        entries["hfb14"] = entry.mass_excess_hfb14
    if with_livechart:
        from . import ground_state

        entries["ame-livechart"] = ground_state(nuc, cache=cache).mass_excess
    return MassComparison(nuclide=nuc, entries=entries)


def mass_table(
    nuclides, cache=None, ripl_path=None, with_livechart: bool = False
) -> dict[tuple[int, int], MassComparison]:
    """`masses` over many nuclides, keyed ``(z, a)`` like a survey table."""
    out: dict[tuple[int, int], MassComparison] = {}
    for item in nuclides:
        nuc = Nuclide.parse(item)
        try:
            out[(nuc.z, nuc.a)] = masses(
                nuc, cache=cache, ripl_path=ripl_path, with_livechart=with_livechart
            )
        except LookupError:
            continue
    return out
