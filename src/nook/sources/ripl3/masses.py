"""The RIPL-3 masses segment: experimental and theoretical mass tables.

Three tables, two theory families:

* ``mass-frdm95.dat`` -- FRDM (1995) masses and deformations, plus the Audi
  experimental mass excesses the file ships with, ``(2i4,1x,a2,1x,i1,4f10.3,4f8.3)``.
* ``mass-hfb14.dat`` -- HFB-14 masses, deformations and Fermi-function
  density-shape parameters, same leading columns.
* ``abundance.dat`` -- natural abundances (Nuclear Wallet Cards 2005).

The experimental columns cannot be split on whitespace: a nucleus with no
measured mass (``fl`` = 0) has an *empty* ``Mexp`` but a present ``Mth``,
which whitespace splitting silently shifts left.  Everything slices by
column.  Energies convert MeV -> keV at parse time, matching the rest of the
package; deformations and density parameters are dimensionless or fm-based
and pass through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...groundstate import GroundState
from ...nuclide import Nuclide
from ...quantities import HalfLife, Uncertain
from ._util import (
    MEV_TO_KEV,
    data_lines,
    float_field,
    int_field,
    parse_file,
    require_file,
    z_blocks,
)

__all__ = [
    "MassEntry",
    "MatterDensity",
    "load_mass_entry",
    "load_mass_table",
    "load_matter_density",
]


@dataclass(frozen=True)
class MassEntry:
    """One nuclide across the RIPL-3 mass tables.  Energies in keV."""

    nuclide: Nuclide
    #: Audi experimental/recommended mass excess as shipped with RIPL-3.
    mass_excess_exp: Uncertain
    #: ``fl`` = 1: the "experimental" value is recommended, not measured.
    recommended_only: bool = False
    mass_excess_frdm95: Uncertain | None = None
    mass_excess_hfb14: Uncertain | None = None
    #: FRDM microscopic correction, keV.
    emic_frdm95: float | None = None
    beta2_frdm95: float | None = None
    beta3_frdm95: float | None = None
    beta4_frdm95: float | None = None
    beta6_frdm95: float | None = None
    beta2_hfb14: float | None = None
    beta4_hfb14: float | None = None
    #: HFB-14 Fermi-function density parameters: (rho0 fm^-3, r fm, a fm).
    neutron_density_params: tuple[float, float, float] | None = None
    proton_density_params: tuple[float, float, float] | None = None
    abundance: Uncertain | None = None

    def provenance(self, quantity: str) -> str | None:
        from . import RIPL3_EVALUATIONS

        key = {
            "mass_excess_exp": "mass_excess_exp",
            "mass_excess_frdm95": "mass_excess_frdm95",
            "emic_frdm95": "mass_excess_frdm95",
            "mass_excess_hfb14": "mass_excess_hfb14",
            "beta2_frdm95": "beta2_frdm95",
            "beta3_frdm95": "beta2_frdm95",
            "beta4_frdm95": "beta2_frdm95",
            "beta6_frdm95": "beta2_frdm95",
            "beta2_hfb14": "beta2_hfb14",
            "beta4_hfb14": "beta2_hfb14",
            "abundance": "abundance",
        }.get(quantity)
        return RIPL3_EVALUATIONS.get(key) if key else None

    def units(self, quantity: str) -> str | None:
        if quantity.startswith("mass_excess") or quantity.startswith("emic"):
            return "keV"
        if quantity == "abundance":
            return "% (mole fraction)"
        if quantity.endswith("density_params"):
            return "(fm^-3, fm, fm)"
        return None  # the beta deformations are dimensionless

    def to_ground_state(self) -> GroundState:
        """Project onto the shared :class:`~nook.groundstate.GroundState`.

        Only the fields RIPL actually carries are filled; theory values ride
        in ``metadata`` so the schema stays identical across sources.
        """
        return GroundState(
            nuclide=self.nuclide,
            half_life=HalfLife(),
            mass_excess=self.mass_excess_exp,
            mass_from_systematics=self.recommended_only,
            abundance=self.abundance if self.abundance is not None else Uncertain(None),
            source="ripl3-local",
            metadata={
                "mass_excess_frdm95_kev": (
                    None
                    if self.mass_excess_frdm95 is None
                    else self.mass_excess_frdm95.value
                ),
                "mass_excess_hfb14_kev": (
                    None
                    if self.mass_excess_hfb14 is None
                    else self.mass_excess_hfb14.value
                ),
                "beta2_frdm95": self.beta2_frdm95,
                "beta2_hfb14": self.beta2_hfb14,
            },
        )


@dataclass(frozen=True)
class MatterDensity:
    """A spherically averaged HFB-14 matter-density profile."""

    nuclide: Nuclide
    beta2: float | None
    #: (radius fm, neutron density fm^-3, proton density fm^-3) rows.
    rows: tuple[tuple[float, float, float], ...]


def _mev(value: float | None, err: float | None, raw: str) -> Uncertain | None:
    if value is None:
        return None
    sigma = None if err is None else err * MEV_TO_KEV
    return Uncertain(value * MEV_TO_KEV, sigma, sigma, raw=raw)


# Both mass tables share the leading Z A s fl Mexp Err Mth columns -- which
# is why the experimental mass is parsed from either file (710 nuclides exist
# only in the HFB table).  They differ only in the trailing columns.
_FRDM_EXTRAS = (
    ("emic", 43, 53),
    ("beta2", 53, 61),
    ("beta3", 61, 69),
    ("beta4", 69, 77),
    ("beta6", 77, 85),
)
_HFB_EXTRAS = (
    ("beta2", 43, 51),
    ("beta4", 51, 59),
    ("rhon", 59, 68),
    ("rn", 68, 77),
    ("an", 77, 86),
    ("rhop", 86, 95),
    ("rp", 95, 104),
    ("ap", 104, 113),
)


def _parse_mass_table(text: str, extras) -> dict:
    table: dict[tuple[int, int], dict] = {}
    for line in data_lines(text):
        z, a = int_field(line, 0, 4), int_field(line, 4, 8)
        if z is None or a is None:
            continue
        row = {
            "flag": int_field(line, 12, 13) or 0,
            "mexp": float_field(line, 13, 23),
            "err": float_field(line, 23, 33),
            "mth": float_field(line, 33, 43),
        }
        for name, start, stop in extras:
            row[name] = float_field(line, start, stop)
        table[(z, a)] = row
    return table


def _parse_frdm(text: str) -> dict:
    return _parse_mass_table(text, _FRDM_EXTRAS)


def _parse_hfb(text: str) -> dict:
    return _parse_mass_table(text, _HFB_EXTRAS)


def _parse_abundance(text: str) -> dict:
    table: dict[tuple[int, int], tuple[float, float]] = {}
    for line in data_lines(text):
        parts = line.split()
        table[(int(parts[0]), int(parts[1]))] = (float(parts[3]), float(parts[4]))
    return table


def _build_entry(nuclide: Nuclide, frdm, hfb, abundance) -> MassEntry:
    """Assemble one :class:`MassEntry` from the per-table row dicts."""
    exp = frdm if frdm is not None else hfb
    hfb_np = None
    hfb_pp = None
    if hfb and hfb["rhon"] is not None:
        hfb_np = (hfb["rhon"], hfb["rn"], hfb["an"])
        hfb_pp = (hfb["rhop"], hfb["rp"], hfb["ap"])
    return MassEntry(
        nuclide=nuclide,
        mass_excess_exp=_mev(exp["mexp"], exp.get("err"), "Audi via RIPL-3")
        or Uncertain(None),
        recommended_only=exp.get("flag") == 1,
        mass_excess_frdm95=None if frdm is None else _mev(frdm["mth"], None, "FRDM95"),
        mass_excess_hfb14=None if hfb is None else _mev(hfb["mth"], None, "HFB-14"),
        emic_frdm95=(
            None if frdm is None or frdm["emic"] is None else frdm["emic"] * MEV_TO_KEV
        ),
        beta2_frdm95=None if frdm is None else frdm["beta2"],
        beta3_frdm95=None if frdm is None else frdm["beta3"],
        beta4_frdm95=None if frdm is None else frdm["beta4"],
        beta6_frdm95=None if frdm is None else frdm["beta6"],
        beta2_hfb14=None if hfb is None else hfb["beta2"],
        beta4_hfb14=None if hfb is None else hfb["beta4"],
        neutron_density_params=hfb_np,
        proton_density_params=hfb_pp,
        abundance=None
        if abundance is None
        else Uncertain(
            abundance[0], abundance[1], abundance[1], raw="RIPL-3 abundance"
        ),
    )


def _mass_tables(path: Path) -> tuple[dict, dict, dict]:
    """The three parsed mass tables (FRDM required, HFB/abundance optional)."""
    frdm_file = path / "masses" / "mass-frdm95.dat"
    require_file(frdm_file, "RIPL-3 mass table")
    frdm = parse_file(_parse_frdm, frdm_file)
    hfb_file = path / "masses" / "mass-hfb14.dat"
    hfb = parse_file(_parse_hfb, hfb_file) if hfb_file.is_file() else {}
    abundance_file = path / "masses" / "abundance.dat"
    abundance = (
        parse_file(_parse_abundance, abundance_file) if abundance_file.is_file() else {}
    )
    return frdm, hfb, abundance


def load_mass_entry(path: Path, nuclide: Nuclide) -> MassEntry:
    """Everything the mass tables say about one nuclide."""
    frdm, hfb, abundance = _mass_tables(path)
    key = (nuclide.z, nuclide.a)
    if frdm.get(key) is None and hfb.get(key) is None:
        raise LookupError(f"no RIPL-3 mass entry for {nuclide}")
    return _build_entry(nuclide, frdm.get(key), hfb.get(key), abundance.get(key))


def load_mass_table(path: Path) -> dict[tuple[int, int], MassEntry]:
    """Every nuclide across the mass tables, keyed ``(z, a)`` (bulk data)."""
    frdm, hfb, abundance = _mass_tables(path)
    return {
        key: _build_entry(
            Nuclide(*key), frdm.get(key), hfb.get(key), abundance.get(key)
        )
        for key in sorted(frdm.keys() | hfb.keys())
    }


def _parse_matter_density_file(text: str) -> dict:
    table: dict[tuple[int, int], tuple[float | None, tuple]] = {}
    for z, a, header, body in z_blocks(text, lambda line: line.startswith(" Z=")):
        beta2: float | None = None
        tokens = header.split()
        for pos, token in enumerate(tokens):
            # both `beta2=0.36` and the spaced `beta2= 0.36` occur
            if token.startswith("beta2="):
                beta2 = float(token.removeprefix("beta2=") or tokens[pos + 1])
        rows = []
        for line in body:
            parts = line.split()
            if len(parts) == 3 and not line.lstrip().startswith("r["):
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        table[(z, a)] = (beta2, tuple(rows))
    return table


def load_matter_density(path: Path, nuclide: Nuclide) -> MatterDensity:
    """The tabulated HFB-14 density profile for one nuclide (bulk data)."""
    file = (
        path
        / "masses"
        / "matter-densities-hfb14"
        / "density-hfb14"
        / f"z{nuclide.z:03d}.dat"
    )
    require_file(file, "HFB-14 matter-density table")
    entry = parse_file(_parse_matter_density_file, file).get((nuclide.z, nuclide.a))
    if entry is None or not entry[1]:
        raise LookupError(f"no HFB-14 matter density for {nuclide}")
    beta2, rows = entry
    return MatterDensity(nuclide=nuclide, beta2=beta2, rows=rows)
