"""The RIPL-3 densities segment: level-density parameters and HFB tables.

Analytic parameter files, one row per nuclide with a D0 fit:

* ``level-densities-egsm.dat`` -- Enhanced Generalised Superfluid Model.
* ``level-densities-bfmeff.dat`` -- Back-shifted Fermi gas (effective).
* ``level-densities-ctmeff.dat`` -- Constant temperature (effective).
* ``level-densities-hfm.dat`` -- Hartree-Fock microscopic normalisation.

The unit trap: EGSM lists ``D0`` in **keV**, the other three in **eV**.
Both normalise to keV here (``spacing_kev``), matching the resonances
segment; the file's own numbers stay in ``fields``.

``level-densities-hfb/z*.tab`` are the microscopic HFB-plus-combinatorial
tables: per nuclide, one block per parity, rows of excitation energy
against total and spin-resolved densities.  Odd-A files label spin columns
``J=00, J=01...`` too -- the column index means ``J = i + 1/2`` there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...nuclide import Nuclide
from ...quantities import Uncertain
from ._util import data_lines, parse_file, require_file, z_blocks

__all__ = [
    "DensityTable",
    "LevelDensityParams",
    "load_hfb_density",
    "load_level_density_params",
]

#: model name -> (filename, D0 scale to keV, column names after Z A El).
_MODELS = {
    "egsm": (
        "level-densities-egsm.dat", 1.0,
        ("i0", "bn_mev", "d0", "d0_err", "esh_mev", "da_plus", "a", "da_minus"),
    ),
    "bfm": (
        "level-densities-bfmeff.dat", 1e-3,
        ("i0", "bn_mev", "d0", "d0_err", "nlow", "ulow", "ntop", "utop",
         "dw", "gamma", "ainf", "aerr", "pairing"),
    ),
    "ctm": (
        "level-densities-ctmeff.dat", 1e-3,
        ("i0", "bn_mev", "d0", "d0_err", "nlow", "ulow", "ntop", "utop",
         "dw", "gamma", "ainf", "aerr", "pairing", "ematch", "e0", "temperature"),
    ),
    "hfm": (
        "level-densities-hfm.dat", 1e-3,
        ("i0", "bn_mev", "d0", "d0_err", "nlow", "ulow", "ntop", "utop",
         "ctable", "deltac", "ptable"),
    ),
}


@dataclass(frozen=True)
class LevelDensityParams:
    """One nuclide's fit parameters for one analytic level-density model."""

    nuclide: Nuclide
    model: str
    target_spin: float | None
    bn_mev: float | None
    #: D0 normalised to keV regardless of the file's native unit.
    spacing_kev: Uncertain
    #: Every column as named in the file (D0 left in its native unit).
    fields: dict[str, float]


@dataclass(frozen=True)
class DensityTable:
    """A spin- and parity-resolved HFB level-density table, rho in 1/MeV."""

    nuclide: Nuclide
    #: parity (+1/-1) -> tuple of rows; each row is
    #: (U MeV, T MeV, cumulative, rho_observed, rho_total, spin-resolved...).
    grids: dict[int, tuple[tuple[float, ...], ...]]

    def _closest(self, parity: int, e_mev: float) -> tuple[float, ...]:
        grid = self.grids[parity]
        return min(grid, key=lambda row: abs(row[0] - e_mev))

    def rho(
        self, e_mev: float, two_j: int | None = None, parity: int | None = None
    ) -> float:
        """Nearest-grid-point *level* density.  Total over J unless ``two_j``
        given; summed over parity unless ``parity`` given.

        The total is RHOOBS (the sum of the spin columns), so
        ``rho(e)`` == ``sum over J of rho(e, two_j)``.  RHOTOT -- the
        2J+1-weighted *state* density -- is a different quantity and stays
        available as ``row[4]`` of :attr:`grids`.
        """
        parities = (parity,) if parity is not None else tuple(self.grids)
        total = 0.0
        for p in parities:
            row = self._closest(p, e_mev)
            if two_j is None:
                total += row[3]
            else:
                # column i holds J = i (even A) or J = i + 1/2 (odd A)
                if (two_j - (self.nuclide.a % 2)) % 2:
                    raise ValueError(
                        f"two_j={two_j} is impossible for A={self.nuclide.a} "
                        f"({'even' if self.nuclide.a % 2 == 0 else 'odd'} A "
                        f"needs {'integer' if self.nuclide.a % 2 == 0 else 'half-integer'} J)"
                    )
                index = (two_j - (self.nuclide.a % 2)) // 2
                if index < 0 or 5 + index >= len(row):
                    raise ValueError(f"two_j={two_j} outside the table's spin columns")
                total += row[5 + index]
        return total


def _parse_density_params(text: str, model: str) -> dict:
    name, scale, columns = _MODELS[model]
    table: dict[tuple[int, int], LevelDensityParams] = {}
    for line in data_lines(text):
        tokens = line.split()
        if len(tokens) < 5 or not tokens[0].isdigit():
            continue
        z, a = int(tokens[0]), int(tokens[1])
        values = {}
        for column, token in zip(columns, tokens[3:]):
            try:
                values[column] = float(token)
            except ValueError:
                pass
        d0 = values.get("d0")
        err = values.get("d0_err")
        table[(z, a)] = LevelDensityParams(
            nuclide=Nuclide(z, a),
            model=model,
            target_spin=values.get("i0"),
            bn_mev=values.get("bn_mev"),
            spacing_kev=Uncertain(
                None if d0 is None else d0 * scale,
                None if err is None else err * scale,
                None if err is None else err * scale,
                raw=f"D0 from {name}",
            ),
            fields=values,
        )
    return table


def load_level_density_params(
    path: Path, nuclide: Nuclide, model: str = "egsm"
) -> LevelDensityParams:
    """The analytic-model parameters for one nuclide."""
    if model not in _MODELS:
        raise ValueError(f"unknown level-density model {model!r}; "
                         f"choose from {sorted(_MODELS)}")
    file = path / "densities" / _MODELS[model][0]
    require_file(file, "RIPL-3 level-density file")
    entry = parse_file(_parse_density_params, file, model).get((nuclide.z, nuclide.a))
    if entry is None:
        raise LookupError(f"no {model} level-density parameters for {nuclide}")
    return entry


def _parse_hfb_tab(text: str) -> dict[tuple[int, int], dict[int, tuple]]:
    tables: dict[tuple[int, int], dict[int, tuple]] = {}
    is_header = lambda line: line.lstrip().startswith("*") and "Level Density" in line
    for z, a, header, body in z_blocks(text, is_header):
        parity = 1 if "Positive-Parity" in header else -1
        rows = []
        for line in body:
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", "U[")):
                continue
            try:
                rows.append(tuple(float(t) for t in stripped.split()))
            except ValueError:
                continue
        tables.setdefault((z, a), {})[parity] = tuple(rows)
    return tables


def load_hfb_density(path: Path, nuclide: Nuclide) -> DensityTable:
    """The microscopic HFB level-density table for one nuclide."""
    file = path / "densities" / "level-densities-hfb" / f"z{nuclide.z:03d}.tab"
    require_file(file, "HFB level-density table")
    grids = parse_file(_parse_hfb_tab, file).get((nuclide.z, nuclide.a))
    if not grids:
        raise LookupError(f"no HFB level-density table for {nuclide}")
    return DensityTable(nuclide=nuclide, grids=grids)
