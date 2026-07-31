"""Ground-state properties across many nuclides, from local flat files.

The Livechart API answers this in one call, but only for adopted data and only
with a network. Walking the mass-chain files gives the same picture offline
and lets you see exactly which dataset each number came from.

One pass over a full ENSDF distribution takes a couple of minutes, so results
are cached to a JSON file by default; delete it to refresh.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .nuclide import Nuclide
from .sources.ensdf_file import _parse_chain, default_ensdf_path

__all__ = [
    "DECAY_ORDER",
    "Loop",
    "NuclideSummary",
    "closure_failures",
    "mass_surface_loops",
    "decay_modes_from",
    "inconsistencies",
    "survey",
]

#: Decay modes in the order a chart should resolve ties, commonest first.
DECAY_ORDER = ("IT", "B-", "EC", "B+", "A", "P", "N", "SF", "2B-", "12C", "14C")


def decay_modes_from(properties) -> tuple[str, ...]:
    """Read decay branchings out of a level's continuation properties.

    ENSDF spells these several ways and two of them are easy to miss:
    ``%EC+%B+`` is one key naming *two* modes, and delayed emission appends the
    particle (``%B-N``, ``%B-2N``, ``%ECP``). Matching only the bare names
    silently drops every electron-capture nuclide.
    """
    from .quantities import parse_value_with_uncertainty

    branching: dict[str, float] = {}
    for key, text in properties.items():
        if not key.startswith("%"):
            continue
        quantity = parse_value_with_uncertainty(str(text).split("$")[0])
        percent = quantity.value if quantity.value is not None else 0.0
        for part in key[1:].upper().split("+%"):
            part = part.strip()
            if not part:
                continue
            if part not in DECAY_ORDER:
                # delayed emission: the primary mode is the leading beta or EC
                part = next(
                    (
                        p
                        for p in ("B-", "B+", "EC")
                        if part.startswith(p) and len(part) > len(p)
                    ),
                    None,
                )
                if part is None:
                    continue
            branching[part] = max(branching.get(part, 0.0), percent)
    # strongest branch first, so dominant_decay reflects the data rather than
    # a fixed convention -- heavy nuclei often have both alpha and EC open
    return tuple(sorted(branching, key=lambda mode: -branching[mode]))


@dataclass(frozen=True)
class NuclideSummary:
    """One nuclide's adopted ground state, summarised from its mass chain.

    Distinct from :class:`~nook.groundstate.GroundState`, which is
    the Livechart aggregate of the mass, radius and moment compilations.
    This one carries only what the flat files themselves record.
    """

    nuclide: Nuclide
    energy_kev: float | None = None
    jpi: str = ""
    two_j: int | None = None
    parity: int | None = None
    half_life_s: float | None = None
    stable: bool = False
    decay_modes: tuple[str, ...] = ()
    s_n: float | None = None
    s_n_unc: float | None = None
    s_p: float | None = None
    q_beta_minus: float | None = None
    q_beta_minus_unc: float | None = None
    q_alpha: float | None = None
    #: Q-record quantities the evaluator marked SY (systematics) or CA
    #: (calculated) rather than measured.
    estimated: tuple[str, ...] = ()
    #: Quantities changed by :func:`nook.repair.repair`; empty unless
    #: repair was explicitly requested. A non-empty tuple means the value is
    #: no longer what the file says.
    repaired: tuple[str, ...] = ()

    @property
    def z(self) -> int:
        return self.nuclide.z

    @property
    def n(self) -> int:
        return self.nuclide.n

    @property
    def dominant_decay(self) -> str | None:
        """The strongest branch, or ``None`` if none is recorded."""
        return self.decay_modes[0] if self.decay_modes else None

    def as_dict(self) -> dict:
        return {
            "z": self.nuclide.z,
            "a": self.nuclide.a,
            "jpi": self.jpi,
            "two_j": self.two_j,
            "parity": self.parity,
            "half_life_s": self.half_life_s,
            "stable": self.stable,
            "decay_modes": list(self.decay_modes),
            "s_n": self.s_n,
            "s_p": self.s_p,
            "q_beta_minus": self.q_beta_minus,
            "q_alpha": self.q_alpha,
            "s_n_unc": self.s_n_unc,
            "q_beta_minus_unc": self.q_beta_minus_unc,
            "estimated": list(self.estimated),
            "repaired": list(self.repaired),
        }

    @classmethod
    def from_dict(cls, row: dict) -> "NuclideSummary":
        return cls(
            nuclide=Nuclide(row["z"], row["a"]),
            jpi=row["jpi"],
            two_j=row["two_j"],
            parity=row["parity"],
            half_life_s=row["half_life_s"],
            stable=row["stable"],
            decay_modes=tuple(row["decay_modes"]),
            s_n=row["s_n"],
            s_p=row["s_p"],
            q_beta_minus=row["q_beta_minus"],
            q_alpha=row["q_alpha"],
            s_n_unc=row.get("s_n_unc"),
            q_beta_minus_unc=row.get("q_beta_minus_unc"),
            estimated=tuple(row.get("estimated", ())),
            repaired=tuple(row.get("repaired", ())),
        )


def _adopted_ground_states(path: Path) -> Iterator[NuclideSummary]:
    for dataset in _parse_chain(path):
        if not dataset.is_adopted or not dataset.levels:
            continue
        placed = [
            lv
            for lv in dataset.levels
            if lv.energy_kev is not None and not lv.is_floating
        ]
        if not placed:
            continue
        ground = min(placed, key=lambda lv: lv.energy_kev or 0.0)
        q_record: dict = dataset.properties.get("Q", {}) or {}  # type: ignore[assignment]

        def q(name):
            value = q_record.get(name)
            return getattr(value, "value", None) if value is not None else None

        def dq(name):
            value = q_record.get(name)
            return getattr(value, "symmetric", None) if value is not None else None

        seconds = ground.half_life.seconds.value
        yield NuclideSummary(
            nuclide=dataset.nuclide,
            energy_kev=ground.energy_kev,
            jpi=ground.spin_parity.raw,
            two_j=ground.spin_parity.unique.two_j
            if ground.spin_parity.unique
            else None,
            parity=ground.spin_parity.unique.parity
            if ground.spin_parity.unique
            else None,
            half_life_s=None if seconds is None or math.isinf(seconds) else seconds,
            stable=ground.half_life.stable,
            decay_modes=tuple(name.upper() for name, _ in ground.decay_modes)
            or decay_modes_from(ground.properties),
            s_n=q("s_n"),
            s_n_unc=dq("s_n"),
            s_p=q("s_p"),
            q_beta_minus=q("q_beta_minus"),
            q_beta_minus_unc=dq("q_beta_minus"),
            q_alpha=q("q_alpha"),
            estimated=tuple(
                name
                for name in ("s_n", "s_p", "q_beta_minus", "q_alpha")
                if str(getattr(q_record.get(name), "operator", "")) in ("SY", "CA")
            ),
        )


def survey(
    path: str | os.PathLike[str] | None = None,
    cache: str | os.PathLike[str] | None = None,
    mass_range: tuple[int, int] | None = None,
) -> list[NuclideSummary]:
    """Ground states for every nuclide in a local ENSDF distribution.

    ``cache`` names a JSON file; when it exists it is read instead of the
    chains. Pass ``cache=False`` to always re-read.
    """
    root = Path(path) if path is not None else default_ensdf_path()
    cache_path = (
        None
        if cache is False
        else Path(cache)
        if cache is not None
        else root / "ground-state-survey.json"
    )
    if cache_path is not None and cache_path.is_file():
        rows = json.loads(cache_path.read_text())
        states = [NuclideSummary.from_dict(row) for row in rows]
    else:
        states = []
        for chain in sorted(root.glob("ensdf.[0-9][0-9][0-9]")):
            states.extend(_adopted_ground_states(chain))
        if cache_path is not None:
            cache_path.write_text(json.dumps([s.as_dict() for s in states]))

    if mass_range is not None:
        low, high = mass_range
        states = [s for s in states if low <= s.nuclide.a <= high]
    return states


#: Separation energies larger than this are outside any physical range and
#: indicate a bad record rather than an exotic nuclide.
_ENERGY_CEILING_KEV = 60_000.0

#: Anything living longer than this cannot be particle-unbound.
_PROMPT_LIMIT_S = 1e-9


def inconsistencies(
    states: "Sequence[NuclideSummary]",
) -> list[tuple["NuclideSummary", str]]:
    """Flag records that contradict themselves.

    Evaluated files carry occasional transcription errors, and a survey that
    absorbs them silently will put them straight into a figure. These checks
    are deliberately conservative -- each one is a statement that cannot be
    true of any nuclide, not a guess about which value is wrong:

    * a stable nuclide cannot have a negative separation energy;
    * a stable nuclide cannot have a positive beta-decay Q-value;
    * a neutron-unbound nuclide cannot have a measurable half-life, since
      neutron emission from an unbound state is prompt (~1e-20 s);
    * no nuclide has a separation energy of tens of MeV.

    Nothing is corrected. The caller decides whether to drop the record, plot
    it anyway, or go and read the file.
    """

    def note(state, name):
        return " (from systematics)" if name in state.estimated else ""

    flagged = []
    for state in states:
        if state.stable and (state.s_n or 0) < 0:
            flagged.append(
                (state, f"stable but S(n) = {state.s_n:g} keV" + note(state, "s_n"))
            )
        if state.stable and (state.s_p or 0) < 0:
            flagged.append(
                (state, f"stable but S(p) = {state.s_p:g} keV" + note(state, "s_p"))
            )
        if state.stable and (state.q_beta_minus or 0) > 0:
            flagged.append((state, f"stable but Q(beta-) = {state.q_beta_minus:g} keV"))
        if (state.s_n or 0) < 0 and (state.half_life_s or 0) > _PROMPT_LIMIT_S:
            flagged.append(
                (
                    state,
                    f"S(n) = {state.s_n:g} keV but T(1/2) = {state.half_life_s:g} s; "
                    "an unbound neutron is emitted promptly" + note(state, "s_n"),
                )
            )
        for name in ("s_n", "s_p", "q_alpha", "q_beta_minus"):
            value = getattr(state, name)
            if value is not None and abs(value) > _ENERGY_CEILING_KEV:
                flagged.append((state, f"{name} = {value:g} keV is out of range"))

    for state, residual, sigma in closure_failures(states):
        ratio = f"{residual / sigma:.0f}" if sigma else "inf"
        flagged.append(
            (
                state,
                f"mass-surface loop misses closure by {residual:.0f} keV "
                f"({ratio} sigma); one of S(n) or Q(beta-) nearby is wrong",
            )
        )
    return flagged


#: A closure residual must exceed both of these to be reported: several times
#: the combined uncertainty, and an absolute floor so that quantities quoted to
#: a fraction of a keV do not trip on rounding.
_CLOSURE_SIGMA = 5.0
_CLOSURE_FLOOR_KEV = 10.0

#: Atomic mass excesses of the free nucleon and of hydrogen, in keV.
_NEUTRON_EXCESS_KEV = 8071.318
_HYDROGEN_EXCESS_KEV = 7288.971


@dataclass(frozen=True)
class Loop:
    """One instance of a mass-surface identity, with its residual."""

    name: str
    anchor: "NuclideSummary"
    residual: float
    sigma: float
    #: ``(nuclide_key, quantity)`` pairs the identity constrains.
    terms: tuple[tuple[tuple[int, int], str], ...]

    @property
    def closed(self) -> bool:
        return self.residual <= max(_CLOSURE_SIGMA * self.sigma, _CLOSURE_FLOOR_KEV)


def _quad(*values) -> float:
    return math.sqrt(sum((v or 0.0) ** 2 for v in values))


def loops_touching(key: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Anchors of every loop that can involve the nuclide at ``key``.

    A value participates in at most five loops, so a repair candidate can be
    scored against those alone instead of rebuilding the whole surface.
    """
    z, a = key
    return ((z, a), (z - 1, a), (z, a + 1), (z + 1, a + 1))


def mass_surface_loops(
    table: dict, anchors: "Sequence[tuple[int, int]] | None" = None
) -> list[Loop]:
    """Every testable instance of the mass-surface identities.

    Two identities are used, both exact differences of the same mass excesses
    and therefore true whatever the masses are:

    ``neutron surface``
        ``S(n)[Z,A] - S(n)[Z+1,A] == Q(b-)[Z,A-1] - Q(b-)[Z,A]``

    ``proton surface``
        ``S(p)[Z,A] - S(n)[Z,A] == Q(b-)[Z-1,A-1] + D(1H) - D(n)``

    The second is what makes ``S(p)`` checkable; the first alone leaves it
    unconstrained. Loops touching a value flagged ``SY`` or ``CA`` are skipped,
    since extrapolations are not expected to close.
    """
    loops: list[Loop] = []
    keys = table.keys() if anchors is None else anchors
    for key in keys:
        state = table.get(key)
        if state is None:
            continue
        z, a = key
        above, lighter = table.get((z + 1, a)), table.get((z, a - 1))
        if (
            above is not None
            and lighter is not None
            and not any("s_n" in r.estimated for r in (state, above))
            and not any("q_beta_minus" in r.estimated for r in (lighter, state))
            and None
            not in (state.s_n, above.s_n, lighter.q_beta_minus, state.q_beta_minus)
        ):
            loops.append(
                Loop(
                    name="neutron surface",
                    anchor=state,
                    residual=abs(
                        (state.s_n - above.s_n)
                        - (lighter.q_beta_minus - state.q_beta_minus)
                    ),
                    sigma=_quad(
                        state.s_n_unc,
                        above.s_n_unc,
                        lighter.q_beta_minus_unc,
                        state.q_beta_minus_unc,
                    ),
                    terms=(
                        (key, "s_n"),
                        ((z + 1, a), "s_n"),
                        ((z, a - 1), "q_beta_minus"),
                        (key, "q_beta_minus"),
                    ),
                )
            )

        diagonal = table.get((z - 1, a - 1))
        if (
            diagonal is not None
            and not any(q in state.estimated for q in ("s_p", "s_n"))
            and "q_beta_minus" not in diagonal.estimated
            and None not in (state.s_p, state.s_n, diagonal.q_beta_minus)
        ):
            loops.append(
                Loop(
                    name="proton surface",
                    anchor=state,
                    residual=abs(
                        (state.s_p - state.s_n)
                        - diagonal.q_beta_minus
                        - (_HYDROGEN_EXCESS_KEV - _NEUTRON_EXCESS_KEV)
                    ),
                    sigma=_quad(state.s_n_unc, diagonal.q_beta_minus_unc),
                    terms=(
                        (key, "s_p"),
                        (key, "s_n"),
                        ((z - 1, a - 1), "q_beta_minus"),
                    ),
                )
            )
    return loops


def closure_failures(
    states: "Sequence[NuclideSummary]",
) -> list[tuple[NuclideSummary, float, float]]:
    """Find Q-values that cannot all be right, using the mass surface.

    Separation energies and beta Q-values are differences of the same mass
    excesses, so they satisfy a closed loop::

        S(n)[Z, A] - S(n)[Z+1, A] == Q(b-)[Z, A-1] - Q(b-)[Z, A]

    Both sides reduce to the same four masses, so the identity holds exactly
    whatever the masses are. A loop that fails to close means at least one of
    the four numbers is wrong -- and this catches errors that no single-record
    check can, because each value looks perfectly reasonable on its own.

    Loops containing a value flagged ``SY`` or ``CA`` are skipped: those are
    extrapolations and are not expected to close. Across a full ENSDF
    distribution the measured loops are sharply bimodal -- they close to a
    median of 0.02 sigma, or they fail by more than 20 sigma, with nothing in
    between -- so the threshold is not delicate.

    Returns ``(anchor, residual_keV, sigma_keV)`` per failing loop.
    """
    table = {(s.z, s.nuclide.a): s for s in states}
    return sorted(
        (
            (loop.anchor, loop.residual, loop.sigma)
            for loop in mass_surface_loops(table)
            if not loop.closed
        ),
        key=lambda row: -row[1],
    )
