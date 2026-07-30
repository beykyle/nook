"""Decay datasets: parent, normalisation and feeding records.

Kept out of :class:`~nook.model.LevelScheme` on purpose -- a feeding
intensity belongs to a *decay*, and the same level carries different feedings
depending on which parent populated it.

:class:`Normalization` is the part that matters. ENSDF intensities are relative
to whatever the evaluator called 100 within one dataset, and only become
comparable across datasets once scaled by ``NR * BR`` or ``NB * BR``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, ClassVar, Iterator

from .model import LevelScheme
from .nuclide import Nuclide
from .quantities import HalfLife, SpinParity, Uncertain, add, multiply

__all__ = ["DecayScheme", "Feeding", "Normalization", "Parent"]


@dataclass(frozen=True)
class Parent:
    """The decaying state, from a ``P`` record."""

    nuclide: Nuclide | None = None
    energy: Uncertain = field(default_factory=lambda: Uncertain(None))
    spin_parity: SpinParity = field(default_factory=SpinParity)
    half_life: HalfLife = field(default_factory=HalfLife)
    q_value: Uncertain = field(default_factory=lambda: Uncertain(None))
    ionisation: str | None = None
    raw: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __str__(self) -> str:
        e = self.energy.value
        head = f"{self.nuclide}" if self.nuclide else "parent"
        at = "" if not e else f" at {e:g} keV"
        return f"{head}{at}  {self.spin_parity}  T1/2 = {self.half_life}"


@dataclass(frozen=True)
class Normalization:
    """An ``N`` record: multipliers onto an absolute per-decay scale."""

    photon: Uncertain | None = None       # NR
    transition: Uncertain | None = None   # NT
    branching: Uncertain | None = None    # BR
    feeding: Uncertain | None = None      # NB
    delayed: Uncertain | None = None      # NP
    raw: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def factor_for(self, kind: str = "photon") -> Uncertain | None:
        """The multiplier taking ``kind`` onto a per-parent-decay scale.

        ``"photon"`` gives ``NR * BR``, ``"feeding"`` gives ``NB * BR``.
        ``"delayed"`` gives ``NP`` alone -- it is already defined against total
        parent decays, so applying ``BR`` again would double-count the branch.
        That last one is inferred from a single dataset; see
        ``docs/limitations.md``.
        """
        if kind == "delayed":
            return self.delayed
        base = {"photon": self.photon, "feeding": self.feeding}.get(kind)
        if base is None or base.value is None:
            return None
        if self.branching is not None and self.branching.value is not None:
            return multiply(base, self.branching)
        return base

    def per_100_decays(self, intensity: Uncertain, kind: str = "photon") -> Uncertain:
        """Convert a relative intensity to one per 100 parent decays.

        A missing multiplier gives back a value of ``None`` rather than
        silently assuming unity.
        """
        factor = self.factor_for(kind)
        if factor is None or factor.value is None:
            return Uncertain(None, raw="no normalisation available")
        return multiply(intensity, factor)

    def __str__(self) -> str:
        bits = [
            f"{name}={value}"
            for name, value in (
                ("NR", self.photon), ("NT", self.transition),
                ("BR", self.branching), ("NB", self.feeding),
                ("NP", self.delayed),
            )
            if value is not None and value.value is not None
        ]
        return "  ".join(bits) or "(empty)"


@dataclass(frozen=True)
class Feeding:
    """A ``B``, ``E``, ``A`` or ``D`` record feeding one daughter level.

    Which fields carry information depends on ``kind``:

    ==========  ==========================================================
    ``"B"``     beta-minus: ``intensity`` (IB), ``log_ft``, ``forbiddenness``
    ``"E"``     EC/beta-plus: ``intensity`` (IB, positron branch),
                ``ec_intensity`` (IE), ``total_intensity``, ``log_ft``
    ``"A"``     alpha: ``energy``, ``intensity`` (IA), ``hindrance_factor``
    ``"D"``     delayed particle: ``particle``, ``energy``, ``intensity``
    ==========  ==========================================================
    """

    kind: str
    level_index: int | None = None
    energy: Uncertain = field(default_factory=lambda: Uncertain(None))
    intensity: Uncertain = field(default_factory=lambda: Uncertain(None))
    ec_intensity: Uncertain | None = None
    total_intensity: Uncertain | None = None
    log_ft: Uncertain | None = None
    hindrance_factor: Uncertain | None = None
    forbiddenness: str | None = None
    particle: str | None = None
    #: D records only: the emitting level in the intermediate nuclide (EI).
    intermediate_energy: Uncertain | None = None
    #: D records only: width of the particle-emitting transition.
    width: Uncertain | None = None
    #: D records only: angular momentum carried off by the particle.
    transfer_l: tuple[int, ...] = ()
    questionable: bool = False
    properties: Mapping[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )
    raw: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    #: ENSDF particle symbols on ``D`` records.
    _PARTICLES: ClassVar[dict[str, str]] = {"N": "neutron", "P": "proton", "A": "alpha"}

    @property
    def label(self) -> str:
        particle = self._PARTICLES.get((self.particle or "").upper(), self.particle)
        return {
            "B": "\u03b2-", "E": "\u03b5/\u03b2+", "A": "\u03b1",
            "D": f"delayed {particle or 'particle'}",
        }.get(self.kind, self.kind)

    def __str__(self) -> str:
        bits = [f"{self.label} \u2192 level {self.level_index}", f"I = {self.intensity}"]
        if self.log_ft is not None and self.log_ft.value is not None:
            bits.append(f"log ft = {self.log_ft}")
        if self.hindrance_factor is not None and self.hindrance_factor.value is not None:
            bits.append(f"HF = {self.hindrance_factor}")
        return "  ".join(bits)


@dataclass(frozen=True)
class DecayScheme:
    """One decay dataset: parents, normalisation, feedings and the daughter."""

    nuclide: Nuclide
    dsid: str
    levels: LevelScheme
    parents: tuple[Parent, ...] = ()
    normalization: Normalization | None = None
    feedings: tuple[Feeding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def parent(self) -> Parent | None:
        return self.parents[0] if self.parents else None

    def __len__(self) -> int:
        return len(self.feedings)

    def __iter__(self) -> Iterator[Feeding]:
        return iter(self.feedings)

    def feeding_of(self, level: int) -> tuple[Feeding, ...]:
        """Every feeding record pointing at the level with this index."""
        return tuple(f for f in self.feedings if f.level_index == level)

    @staticmethod
    def _kind_of(feeding: "Feeding") -> str:
        """Delayed particles normalise through NP, everything else through NB."""
        return "delayed" if feeding.kind == "D" else "feeding"

    def absolute_feedings(self) -> tuple[tuple[Feeding, Uncertain], ...]:
        """Each feeding paired with its intensity per 100 parent decays."""
        if self.normalization is None:
            return tuple((f, Uncertain(None, raw="no N record")) for f in self.feedings)
        return tuple(
            (f, self.normalization.per_100_decays(f.intensity, self._kind_of(f)))
            for f in self.feedings
        )

    def total_feeding(self, kind: str | None = None) -> Uncertain:
        """Summed feeding per 100 parent decays, correlations handled.

        Every feeding shares the same ``NB * BR``, so quadrature over the
        scaled values would understate the total. Summing first and scaling
        once is exact and needs no correlation tracking.
        """
        selected = [f for f in self.feedings if kind is None or f.kind == kind]
        if not selected:
            return Uncertain(None, raw="no feedings")
        kinds = {self._kind_of(f) for f in selected}
        if len(kinds) > 1:
            raise ValueError(
                "delayed-particle and beta feedings normalise differently; "
                "pass kind= to total one at a time"
            )
        relative = add(*(f.intensity for f in selected))
        if self.normalization is None:
            return Uncertain(None, raw="no N record")
        factor = self.normalization.factor_for(kinds.pop())
        if factor is None or factor.value is None:
            return Uncertain(None, raw="no normalisation available")
        return multiply(relative, factor)

    def total_photon_intensity(self) -> Uncertain:
        """Summed gamma intensity per 100 parent decays, correlations handled.

        Same argument as :meth:`total_feeding`: ``NR * BR`` is common to every
        gamma in the dataset, so it is applied once to the summed relative
        intensities rather than to each term.
        """
        gammas = [g for g in self.levels.gammas if g.intensity is not None]
        if not gammas or self.normalization is None:
            return Uncertain(None, raw="no N record")
        factor = self.normalization.factor_for("photon")
        if factor is None or factor.value is None:
            return Uncertain(None, raw="no normalisation available")
        intensities = [g.intensity for g in gammas if g.intensity is not None]
        return multiply(add(*intensities), factor)

    def absolute_photon_intensity(self, gamma) -> Uncertain:
        """A gamma's ``RI`` expressed per 100 parent decays."""
        if self.normalization is None or gamma.intensity is None:
            return Uncertain(None, raw="no N record")
        return self.normalization.per_100_decays(gamma.intensity, "photon")

    def __str__(self) -> str:
        parent = str(self.parent) if self.parent else "unknown parent"
        return (
            f"{self.dsid}\n"
            f"  parent: {parent}\n"
            f"  norm:   {self.normalization or '(none)'}\n"
            f"  {len(self.feedings)} feedings, {len(self.levels)} daughter levels"
        )
