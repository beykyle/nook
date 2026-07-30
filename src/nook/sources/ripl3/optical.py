"""The RIPL-3 optical segment: the optical model parameter archive.

``om-parameter-u.dat`` holds 566 potentials separated by ``+++...`` lines.
Each entry is free-format Fortran: a reference number, author/reference/
summary text, applicability ranges, then six potential components (real and
imaginary volume, surface, spin-orbit), each with per-energy-range
geometry (13 radius + 13 diffuseness coefficients) and strength (25
coefficients), then a Coulomb block and optional coupled-channel data.

This module stores the numbers, faithfully grouped, and does **not**
evaluate potentials: turning the coefficient tables into V(r, E) means
reimplementing om-retrieve's formula zoo (dispersive integrals included),
which is out of scope.  ``jrange < 0`` (volume-integral convention) is
preserved as ``PotentialComponent.as_volume_integral``.

Beware the number format: depths appear as ``.00000+0`` and ``-3.00000-1``
-- Fortran exponents with the ``E`` elided -- which ``float()`` rejects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...nuclide import Nuclide
from ._util import MEV_TO_KEV, data_lines, parse_file, require_file

__all__ = [
    "Deformation",
    "OpticalPotential",
    "PotentialComponent",
    "PotentialRange",
    "load_deformations",
    "load_optical_potentials",
]

_COMPONENTS = (
    "real_volume", "imag_volume", "real_surface",
    "imag_surface", "real_spin_orbit", "imag_spin_orbit",
)

_PROJECTILES = {(0, 1): "n", (1, 1): "p", (1, 2): "d", (1, 3): "t",
                (2, 3): "3He", (2, 4): "a"}

_FORTRAN_EXP = re.compile(r"^([+-]?\d*\.?\d+)([+-]\d+)$")


def _fortran_float(token: str) -> float:
    """``float`` plus the exponent-without-E spelling: ``-3.00000-1``."""
    try:
        return float(token)
    except ValueError:
        m = _FORTRAN_EXP.match(token)
        if m:
            return float(f"{m.group(1)}E{m.group(2)}")
        raise


@dataclass(frozen=True)
class PotentialRange:
    """One energy range of one component: geometry and strength coefficients."""

    e_max_mev: float
    radius_coefficients: tuple[float, ...]       # rco, 13
    diffuseness_coefficients: tuple[float, ...]  # aco, 13
    strength_coefficients: tuple[float, ...]     # pot, 25


@dataclass(frozen=True)
class PotentialComponent:
    kind: str                     # one of _COMPONENTS
    ranges: tuple[PotentialRange, ...]
    #: ``jrange`` was negative: coefficients give volume integrals instead
    #: of potential strengths.
    as_volume_integral: bool = False


@dataclass(frozen=True)
class OpticalPotential:
    """One archive entry, applicability plus coefficient tables."""

    ripl_id: int
    projectile: str               # "n", "p", ... or "Z=z,A=a" if exotic
    author: str
    reference: str
    summary: str
    e_range_mev: tuple[float, float]
    z_range: tuple[int, int]
    a_range: tuple[int, int]
    model: int                    # 0 spherical, 1 rigid rotor, 2 vibrational, 3 soft rotor
    relativistic: int
    dispersive: int
    components: tuple[PotentialComponent, ...]
    #: (ecoul, rcoul0, rcoul, rcoul1, rcoul2, beta, acoul, rcoul3) rows.
    coulomb: tuple[tuple[float, ...], ...] = ()

    def applies_to(
        self, nuclide: Nuclide | None = None, energy_mev: float | None = None
    ) -> bool:
        """Whether the entry covers the given nuclide and/or energy.

        Either filter may be given alone; an omitted one is not checked.
        """
        if nuclide is not None:
            z_lo, z_hi = self.z_range
            a_lo, a_hi = self.a_range
            if not (z_lo <= nuclide.z <= z_hi and a_lo <= nuclide.a <= a_hi):
                return False
        if energy_mev is not None:
            e_lo, e_hi = self.e_range_mev
            return e_lo <= energy_mev <= e_hi
        return True


@dataclass(frozen=True)
class Deformation:
    """A row of ``om-deformations.dat``: an excited-level beta_L."""

    nuclide: Nuclide
    energy_kev: float
    spin: float
    parity: int
    multipole: int
    beta: float
    reference: str


def _parse_entry(text: str) -> OpticalPotential:
    lines = text.strip("\n").splitlines()
    ripl_id = int(lines[0].split()[0])
    author = lines[1].strip()
    reference = lines[2].strip()
    summary = " ".join(line.strip() for line in lines[3:7])
    tokens = " ".join(lines[7:]).split()
    pos = 0

    def take(count: int) -> list[float]:
        nonlocal pos
        values = [_fortran_float(t) for t in tokens[pos:pos + count]]
        if len(values) != count:
            raise ValueError(f"entry {ripl_id}: expected {count} more values")
        pos += count
        return values

    emin, emax = take(2)
    izmin, izmax = (int(v) for v in take(2))
    iamin, iamax = (int(v) for v in take(2))
    imodel, izproj, iaproj, irel, idr = (int(v) for v in take(5))

    components = []
    for kind in _COMPONENTS:
        jrange = int(take(1)[0])
        ranges = []
        for _ in range(abs(jrange)):
            e_max = take(1)[0]
            rco = tuple(take(13))
            aco = tuple(take(13))
            pot = tuple(take(25))
            ranges.append(PotentialRange(e_max, rco, aco, pot))
        components.append(
            PotentialComponent(
                kind=kind, ranges=tuple(ranges), as_volume_integral=jrange < 0
            )
        )

    jcoul = int(take(1)[0])
    coulomb = tuple(tuple(take(8)) for _ in range(jcoul))
    # anything after this is coupled-channel model data; not parsed.

    return OpticalPotential(
        ripl_id=ripl_id,
        projectile=_PROJECTILES.get((izproj, iaproj), f"Z={izproj},A={iaproj}"),
        author=author,
        reference=reference,
        summary=summary,
        e_range_mev=(emin, emax),
        z_range=(izmin, izmax),
        a_range=(iamin, iamax),
        model=imodel,
        relativistic=irel,
        dispersive=idr,
        components=tuple(components),
        coulomb=coulomb,
    )


def _parse_archive(text: str) -> dict[int, OpticalPotential]:
    table: dict[int, OpticalPotential] = {}
    for chunk in re.split(r"^\+{10,}\s*$", text, flags=re.MULTILINE):
        if not chunk.strip():
            continue
        entry = _parse_entry(chunk)
        table[entry.ripl_id] = entry
    return table


def load_optical_potentials(
    path: Path,
    nuclide: Nuclide | None = None,
    projectile: str | None = None,
    ripl_id: int | None = None,
    energy_mev: float | None = None,
) -> tuple[OpticalPotential, ...]:
    """Archive entries, optionally filtered by applicability."""
    file = path / "optical" / "om-parameter-u.dat"
    require_file(file, "RIPL-3 optical archive")
    table = parse_file(_parse_archive, file)
    if ripl_id is not None:
        entry = table.get(ripl_id)
        if entry is None:
            raise LookupError(f"no RIPL-3 optical potential with id {ripl_id}")
        return (entry,)
    return tuple(
        entry
        for entry in table.values()
        if (projectile is None or entry.projectile == projectile)
        and entry.applies_to(nuclide, energy_mev)
    )


def _parse_deformations(text: str) -> dict[tuple[int, int], tuple[Deformation, ...]]:
    table: dict[tuple[int, int], list[Deformation]] = {}
    for line in data_lines(text):
        tokens = line.split()
        z, a = int(tokens[0]), int(tokens[1])
        table.setdefault((z, a), []).append(
            Deformation(
                nuclide=Nuclide(z, a),
                energy_kev=float(tokens[3]) * MEV_TO_KEV,
                spin=float(tokens[4]),
                parity=int(tokens[5]),
                multipole=int(tokens[6]),
                beta=float(tokens[7]),
                reference=tokens[8] if len(tokens) > 8 else "",
            )
        )
    return {key: tuple(rows) for key, rows in table.items()}


def load_deformations(path: Path, nuclide: Nuclide) -> tuple[Deformation, ...]:
    """Excited-level deformation parameters for coupled-channel calculations."""
    file = path / "optical" / "om-deformations.dat"
    require_file(file, "RIPL-3 deformation file")
    out = parse_file(_parse_deformations, file).get((nuclide.z, nuclide.a))
    if not out:
        raise LookupError(f"no RIPL-3 deformation entries for {nuclide}")
    return out
