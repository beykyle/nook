"""RIPL-3 backend: the IAEA Reference Input Parameter Library, from disk.

RIPL-3 is a static library (Capote et al., Nucl. Data Sheets 110 (2009)
3107; the discrete-levels segment was last cut in 2021), so unlike the other
backends it is mirrored wholesale into the repository -- ``data/ripl3/`` --
by ``tools/fetch_ripl3.py``, and parsed from there.

The discrete levels derive from ENSDF, which is what makes cross-source
comparison meaningful; everything else (masses, resonances, optical
potentials, level densities, gamma strength, fission barriers) has no
counterpart in the other two backends.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...groundstate import GroundState
from ...model import LevelScheme
from ...nuclide import Nuclide
from ._util import clear_ripl_cache, parse_file, require_file
from .levels import (
    LevelsParam,
    RiplNuclideLevels,
    load_levels,
    parse_levels_file,
    parse_levels_param,
)
from .masses import (
    MassEntry,
    MatterDensity,
    load_mass_entry,
    load_mass_table,
    load_matter_density,
)
from .resonances import ResonanceEntry, load_resonances, parse_resonances
from .gamma import GDREntry, GSFTable, load_gdr, load_gsf
from .fission import BarrierEntry, FissionBarriers, load_fission_barriers
from .optical import (
    Deformation,
    OpticalPotential,
    load_deformations,
    load_optical_potentials,
)
from .densities import (
    DensityTable,
    LevelDensityParams,
    load_hfb_density,
    load_level_density_params,
)

__all__ = [
    "RIPL3_EVALUATIONS",
    "RIPL3_UNITS",
    "BarrierEntry",
    "Deformation",
    "DensityTable",
    "FissionBarriers",
    "GDREntry",
    "GSFTable",
    "LevelDensityParams",
    "LevelsParam",
    "MassEntry",
    "MatterDensity",
    "OpticalPotential",
    "ResonanceEntry",
    "Ripl3Source",
    "RiplNuclideLevels",
    "clear_ripl_cache",
    "default_ripl_path",
    "parse_levels_file",
    "parse_levels_param",
]

#: Which compilation each RIPL-3 segment or quantity comes from.  The same
#: idea as :data:`nook.groundstate.EVALUATIONS`: attribution families, not
#: per-value citations.
RIPL3_EVALUATIONS = {
    "levels": "RIPL-3 discrete levels (2021 cut; derived from ENSDF, "
              "NUBASE2020 ground states, AME2020 separation energies)",
    "levels_param": "RIPL-3 constant-temperature fit of discrete level schemes",
    "mass_excess_exp": "Audi mass evaluation as shipped with RIPL-3",
    "mass_excess_frdm95": "FRDM (Moller, Nix, Myers, Swiatecki 1995)",
    "mass_excess_hfb14": "HFB-14 (Goriely, Samyn, Pearson 2007)",
    "beta2_frdm95": "FRDM ground-state deformations",
    "beta2_hfb14": "HFB-14 ground-state deformations",
    "abundance": "isotopic abundances as shipped with RIPL-3",
    "resonances": "RIPL-3 average neutron resonance parameters",
    "gdr": "RIPL-3 giant dipole resonance parameters",
    "gsf": "RIPL-3 gamma-ray strength functions",
    "level_density": "RIPL-3 level-density parameters",
    "hfb_density": "HFB plus combinatorial level densities (Goriely et al.)",
    "optical": "RIPL-3 optical model parameter archive",
    "fission_barriers": "RIPL-3 empirical and HFB fission barriers",
}

#: Units after parsing.  File-native units (MeV, mostly) are converted at
#: parse time; ``raw`` fields keep the file's numbers.
RIPL3_UNITS = {
    "level_energy": "keV",
    "sn": "keV",
    "sp": "keV",
    "half_life": "s",
    "gamma_energy": "keV",
    "gamma_intensity": "probability (Pg, not per-100)",
    "mass_excess": "keV",
    "abundance": "% (mole fraction)",
    "d0": "keV",
    "d1": "keV",
    "s0": "1e-4",
    "s1": "1e-4",
    "gamma_width": "meV",
    "gdr_energy": "MeV",
    "gdr_width": "MeV",
    "gdr_cross_section": "mb",
    "barrier_height": "MeV",
    "barrier_hw": "MeV",
    "temperature": "MeV",
}


def default_ripl_path() -> Path:
    """``$RIPL_PATH``, else the repository's committed mirror.

    The mirror lives at ``data/ripl3`` in the repository root, which resolves
    from an editable install; a wheel install has no mirror and must set
    ``$RIPL_PATH``.
    """
    env = os.environ.get("RIPL_PATH")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[4] / "data" / "ripl3"


class Ripl3Source:
    """Read RIPL-3 data from a local mirror."""

    name = "ripl3-local"

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else default_ripl_path()

    # -- discrete levels ----------------------------------------------------

    def levels(self, nuclide) -> RiplNuclideLevels:
        """The raw RIPL block: counts, cutoffs, levels, gammas."""
        return load_levels(self.path, Nuclide.parse(nuclide))

    def fetch(self, nuclide, up_to_nmax: bool = False) -> LevelScheme:
        """The discrete level scheme as a :class:`~nook.model.LevelScheme`.

        ``up_to_nmax=True`` truncates at RIPL's own completeness cutoff
        ``Nmax`` -- the evaluator's judgement, as opposed to the heuristic
        :meth:`~nook.model.LevelScheme.complete_up_to`.
        """
        block = self.levels(nuclide)
        scheme = block.to_scheme()
        if up_to_nmax:
            # levels are index-ordered, so Nmax truncation is a plain slice;
            # _respin applies the same gamma-pruning rule as every other
            # truncation (below, filter, complete_up_to).
            scheme = scheme[: block.n_complete]
        return scheme

    def levels_param(self) -> dict[tuple[int, int], LevelsParam]:
        """The constant-temperature fit table for every nuclide."""
        file = self.path / "levels" / "levels-param-2021.data"
        require_file(file, "RIPL-3 levels-param file")
        return parse_file(parse_levels_param, file)

    def sn_table(self, zs=None) -> dict[tuple[int, int], float]:
        """Neutron separation energies (keV) from the levels headers (bulk).

        Walks every per-element levels file -- half a minute cold, cached
        after -- so ``zs`` can restrict the sweep to an iterable of Z values.
        """
        levels_dir = self.path / "levels"
        require_file(levels_dir / "z001.dat", "RIPL-3 levels segment")
        table: dict[tuple[int, int], float] = {}
        for file in sorted(levels_dir.glob("z*.dat")):
            if zs is not None and int(file.stem[1:]) not in set(zs):
                continue
            for block in parse_file(parse_levels_file, file):
                if block.sn_kev is not None:
                    table[(block.nuclide.z, block.nuclide.a)] = block.sn_kev
        return table

    # -- masses -------------------------------------------------------------

    def masses(self, nuclide) -> MassEntry:
        """Experimental and theoretical masses, deformations, abundance."""
        return load_mass_entry(self.path, Nuclide.parse(nuclide))

    def mass_table(self) -> dict[tuple[int, int], MassEntry]:
        """Every nuclide across the mass tables, keyed ``(z, a)`` (bulk data)."""
        return load_mass_table(self.path)

    def ground_state(self, nuclide) -> GroundState:
        """The mass tables projected onto the shared ground-state model.

        Far sparser than the Livechart ground state: RIPL carries mass
        excess and abundance, and its theory-only values (FRDM95, HFB-14,
        deformations) ride in ``metadata``.
        """
        return self.masses(nuclide).to_ground_state()

    def matter_density(self, nuclide) -> MatterDensity:
        """The tabulated HFB-14 spherical matter-density profile."""
        return load_matter_density(self.path, Nuclide.parse(nuclide))

    # -- resonances, gamma, fission -----------------------------------------

    def resonances(self, nuclide, wave: int = 0) -> ResonanceEntry:
        """Average s-wave (0) or p-wave (1) resonance parameters for a target."""
        return load_resonances(self.path, Nuclide.parse(nuclide), wave=wave)

    def resonance_table(self, wave: int = 0) -> dict[tuple[int, int], ResonanceEntry]:
        """Every target's average resonance parameters for one wave (bulk data)."""
        file = self.path / "resonances" / f"resonances{wave}.dat"
        require_file(file, "RIPL-3 resonance file")
        return parse_file(parse_resonances, file, wave)

    def gdr(self, nuclide) -> tuple[GDREntry, ...]:
        """Every GDR parameterisation: experimental fits first, then theory."""
        return load_gdr(self.path, Nuclide.parse(nuclide))

    def gsf(self, nuclide) -> GSFTable:
        """The microscopic (HFB+QRPA) E1 strength function table."""
        return load_gsf(self.path, Nuclide.parse(nuclide))

    def fission_barriers(self, nuclide, model: str = "empirical") -> FissionBarriers:
        """Fission barriers, ``model`` = ``"empirical"`` or ``"hfb"``."""
        return load_fission_barriers(self.path, Nuclide.parse(nuclide), model=model)

    # -- optical, level densities -------------------------------------------

    def optical_potentials(
        self,
        nuclide=None,
        projectile: str | None = None,
        ripl_id: int | None = None,
        energy_mev: float | None = None,
    ) -> tuple[OpticalPotential, ...]:
        """Archive entries filtered by applicability (records, not V(r))."""
        return load_optical_potentials(
            self.path,
            nuclide=None if nuclide is None else Nuclide.parse(nuclide),
            projectile=projectile,
            ripl_id=ripl_id,
            energy_mev=energy_mev,
        )

    def deformations(self, nuclide) -> tuple[Deformation, ...]:
        """Excited-level deformation parameters (coupled-channel input)."""
        return load_deformations(self.path, Nuclide.parse(nuclide))

    def level_density_params(self, nuclide, model: str = "egsm") -> LevelDensityParams:
        """Analytic level-density parameters; models: egsm, bfm, ctm, hfm."""
        return load_level_density_params(self.path, Nuclide.parse(nuclide), model=model)

    def hfb_density(self, nuclide) -> DensityTable:
        """The microscopic HFB level-density table rho(E, J, parity)."""
        return load_hfb_density(self.path, Nuclide.parse(nuclide))
