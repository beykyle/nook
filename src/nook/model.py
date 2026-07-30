"""The data model returned by every source backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from collections.abc import Mapping
from typing import Any, Callable, Iterator, Sequence, overload

from .nuclide import Nuclide
from .quantities import HalfLife, JPi, SpinParity, Uncertain

__all__ = ["Gamma", "Level", "LevelScheme"]


@dataclass(frozen=True)
class Level:
    """A single nuclear level.

    ``energy`` is keV above the ground state. When ``energy_offset`` is set the
    position is unknown and ``energy`` is only the numeric part of
    ``offset + energy`` (manual V.18).
    """

    index: int
    energy: Uncertain
    spin_parity: SpinParity = field(default_factory=SpinParity)
    half_life: HalfLife = field(default_factory=HalfLife)
    energy_offset: str | None = None
    band: str | None = None
    metastable: bool = False
    questionable: bool = False
    decay_modes: tuple[tuple[str, Uncertain], ...] = ()
    dataset: str | None = None
    #: ENSDF XREF symbols -- which datasets reported this level.
    xref: tuple[str, ...] = ()
    #: The same, resolved to dataset identifiers via the X records.
    observed_in: tuple[str, ...] = ()
    #: Transferred angular momenta from the L field of a reaction dataset.
    transfer_l: tuple[int, ...] = ()
    transfer_l_raw: str | None = None
    #: Spectroscopic factor from the S field.
    spectroscopic_factor: Uncertain | None = None
    #: Magnetic dipole moment, nuclear magnetons (ENSDF ``MOMM1``).
    magnetic_dipole: Uncertain | None = None
    #: Electric quadrupole moment, barn (ENSDF ``MOME2``).
    electric_quadrupole: Uncertain | None = None
    #: Everything from the data-continuation records, as raw text.
    properties: Mapping[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )
    raw: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def energy_kev(self) -> float | None:
        return self.energy.value

    @property
    def is_floating(self) -> bool:
        """True if the excitation energy carries an unknown offset."""
        return self.energy_offset is not None

    @property
    def jpi(self) -> JPi | None:
        """Shorthand for an unambiguous spin-parity, else ``None``."""
        return self.spin_parity.unique

    def __str__(self) -> str:
        e = "?" if self.energy_kev is None else f"{self.energy_kev:.2f}"
        if self.energy_offset:
            e = f"{self.energy_offset}+{e}"
        return f"{e:>12} keV  {self.spin_parity!s:<12}  {self.half_life!s}"


@dataclass(frozen=True)
class Gamma:
    """An electromagnetic transition between two levels."""

    energy: Uncertain
    start_index: int | None = None
    end_index: int | None = None
    intensity: Uncertain | None = None
    multipolarity: str | None = None
    mixing_ratio: Uncertain | None = None
    conversion_coefficient: Uncertain | None = None
    #: Final-level energy stated by an ``FL=`` continuation, if present.
    final_level_energy: float | None = None
    #: Everything from the gamma data-continuation records, as raw text.
    properties: Mapping[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )
    raw: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def energy_kev(self) -> float | None:
        return self.energy.value

    def __str__(self) -> str:
        e = "?" if self.energy_kev is None else f"{self.energy_kev:9.2f}"
        mult = f"  {self.multipolarity}" if self.multipolarity else ""
        return f"{e} keV  {self.start_index} -> {self.end_index}{mult}"


@dataclass(frozen=True)
class LevelScheme:  # noqa: D101
    """Levels plus gamma transitions for one nuclide from one source."""

    nuclide: Nuclide
    levels: tuple[Level, ...] = ()
    gammas: tuple[Gamma, ...] = ()
    source: str = ""
    dataset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.levels)

    def __iter__(self) -> Iterator[Level]:
        return iter(self.levels)

    @overload
    def __getitem__(self, item: int) -> Level: ...

    @overload
    def __getitem__(self, item: slice) -> "LevelScheme": ...

    def __getitem__(self, item: int | slice) -> "Level | LevelScheme":
        if isinstance(item, slice):
            return self._respin(self.levels[item])
        return self.levels[item]

    # -- queries ------------------------------------------------------------

    def _respin(self, levels: Sequence[Level]) -> "LevelScheme":
        """Rebuild a scheme around a subset of levels.

        A gamma is kept when the level it *depopulates* survives; also
        requiring the final level would silently drop transitions whose
        placement is unknown. Indices are stable, so a kept gamma may point at
        a pruned level -- :meth:`level_by_index` returns ``None`` there.
        """
        keep = {lv.index for lv in levels}
        gammas = tuple(g for g in self.gammas if g.start_index in keep)
        return LevelScheme(
            nuclide=self.nuclide,
            levels=tuple(levels),
            gammas=gammas,
            source=self.source,
            dataset=self.dataset,
            metadata=self.metadata,
        )

    def filter(self, predicate: Callable[[Level], bool]) -> "LevelScheme":
        """A new scheme keeping only levels satisfying ``predicate``."""
        return self._respin([lv for lv in self.levels if predicate(lv)])

    def below(self, energy_kev: float) -> "LevelScheme":
        """Levels at or below ``energy_kev`` (floating levels are dropped)."""
        return self.filter(
            lambda lv: lv.energy_kev is not None
            and not lv.is_floating
            and lv.energy_kev <= energy_kev
        )

    def with_known_jpi(self, allow_tentative: bool = True) -> "LevelScheme":
        """Levels carrying an unambiguous spin-parity assignment."""

        def keep(lv: Level) -> bool:
            sp = lv.spin_parity
            if sp.unique is None or sp.unique.parity is None:
                return False
            return allow_tentative or not (sp.tentative or sp.assumed)

        return self.filter(keep)

    def complete_up_to(
        self, allow_tentative: bool = True
    ) -> "LevelScheme":
        """Truncate at the first level lacking a firm ``E``, ``J`` and ``pi``.

        This reproduces the usual RIPL-style "discrete level cutoff" used
        when handing a level scheme to a reaction code: everything below the
        first incompletely characterised level.
        """
        out: list[Level] = []
        for lv in sorted(self.levels, key=_energy_key):
            sp = lv.spin_parity
            ok = (
                lv.energy_kev is not None
                and not lv.is_floating
                and sp.unique is not None
                and sp.unique.parity is not None
                and (allow_tentative or not (sp.tentative or sp.assumed))
            )
            if not ok:
                break
            out.append(lv)
        return self._respin(out)

    def isomers(self, min_half_life_s: float = 1e-9) -> "LevelScheme":
        """Levels living longer than ``min_half_life_s``."""
        return self.filter(
            lambda lv: lv.half_life.seconds.value is not None
            and lv.half_life.seconds.value >= min_half_life_s
        )

    def seen_in(self, pattern: str) -> "LevelScheme":
        """Levels reported by a dataset whose identifier contains ``pattern``.

        Provenance filtering: ``scheme.seen_in("(P,D)")`` keeps only levels a
        transfer measurement actually observed, rather than every level the
        evaluator adopted from anywhere.
        """
        needle = pattern.upper()
        return self.filter(
            lambda lv: any(needle in dsid.upper() for dsid in lv.observed_in)
        )

    def band(self, label: str) -> "LevelScheme":
        """Levels carrying the ENSDF band flag ``label`` (column 77)."""
        return self.filter(lambda lv: lv.band == label)

    def bands(self) -> "dict[str, LevelScheme]":
        """Every labelled band, ordered by bandhead energy.

        Not every flag has a ``BAND()`` definition -- 180Ta uses 27 and
        documents 24 -- so :meth:`band_definition` may return ``None`` for a
        group returned here. Labels are case-sensitive.
        """
        labels = {lv.band for lv in self.levels if lv.band}
        grouped = {label: self.band(label) for label in labels}
        return dict(
            sorted(
                grouped.items(),
                key=lambda kv: min(
                    (_energy_key(lv) for lv in kv[1].levels), default=math.inf
                ),
            )
        )

    def band_definition(self, label: str) -> str | None:
        """The decoded ``BAND()`` text, or ``None`` if it was never defined."""
        return self.metadata.get("band_definitions", {}).get(label)

    @cached_property
    def _by_index(self) -> dict[int, Level]:
        return {lv.index: lv for lv in self.levels}

    def level_by_index(self, index: int | None) -> Level | None:
        """The level with this ENSDF index, or ``None`` if it isn't present."""
        if index is None:
            return None
        return self._by_index.get(index)

    @cached_property
    def _gammas_by_start(self) -> dict[int, tuple[Gamma, ...]]:
        by_start: dict[int, list[Gamma]] = {}
        for g in self.gammas:
            if g.start_index is None:
                continue
            by_start.setdefault(g.start_index, []).append(g)
        return {idx: tuple(gs) for idx, gs in by_start.items()}

    def decays_from(self, level: Level | int) -> tuple[Gamma, ...]:
        idx = level.index if isinstance(level, Level) else level
        return self._gammas_by_start.get(idx, ())

    # -- export -------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Flat dict-per-level, convenient for ``pandas.DataFrame``."""
        rows = []
        for lv in self.levels:
            sp = lv.spin_parity
            rows.append(
                {
                    "index": lv.index,
                    "energy_kev": lv.energy_kev,
                    "energy_unc_kev": lv.energy.symmetric,
                    "energy_offset": lv.energy_offset,
                    "jpi": str(sp),
                    "two_j": sp.unique.two_j if sp.unique else None,
                    "parity": sp.unique.parity if sp.unique else None,
                    "tentative": sp.tentative,
                    "half_life_s": lv.half_life.seconds.value,
                    "stable": lv.half_life.stable,
                    "from_width": lv.half_life.from_width,
                }
            )
        return rows

    def to_dataframe(self):  # pragma: no cover - optional dependency
        """Requires ``pandas``; kept optional so the core stays stdlib-only."""
        import pandas as pd

        return pd.DataFrame(self.to_records())

    def __str__(self) -> str:
        head = (
            f"{self.nuclide} level scheme "
            f"({len(self.levels)} levels, {len(self.gammas)} gammas"
            f"{', ' + self.source if self.source else ''})"
        )
        body = "\n".join(f"  [{lv.index:>3}] {lv}" for lv in self.levels[:20])
        more = "" if len(self.levels) <= 20 else f"\n  ... {len(self.levels) - 20} more"
        return f"{head}\n{body}{more}"


def _energy_key(level: Level) -> float:
    if level.is_floating or level.energy_kev is None:
        return math.inf
    return level.energy_kev
