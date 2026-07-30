"""nook: nuclear level schemes from ENSDF.

Two backends behind one data model -- the IAEA Livechart API for adopted
levels, the NNDC flat files for every evaluated dataset. See ``docs/usage.md``.
"""

from .cache import Cache, default_cache_dir
from .decay import DecayScheme, Feeding, Normalization, Parent
from .groundstate import EVALUATIONS, UNITS, GroundState
from .model import Gamma, Level, LevelScheme
from .nuclide import Nuclide
from .quantities import (
    HalfLife,
    JPi,
    SpinParity,
    Uncertain,
    parse_half_life,
    parse_spin_parity,
    parse_value_with_uncertainty,
)
from .sources.ensdf_file import default_ensdf_path
from .repair import Repair, repair
from .survey import NuclideSummary, survey
from .sources import (
    ENSDFFileSource,
    LivechartError,
    LivechartSource,
    Ripl3Source,
    default_ripl_path,
    parse_references,
    place_gammas,
)

from . import compare

__version__ = "0.1.0"

#: Accepted spelling -> canonical source name.  Every dispatcher that takes a
#: ``source`` string (here, ``compare``) resolves through this one map.
SOURCE_ALIASES = {
    "iaea": "livechart",
    "api": "livechart",
    "ensdf": "file",
    "flat": "file",
    "ripl": "ripl3",
}

__all__ = [
    "EVALUATIONS",
    "UNITS",
    "Cache",
    "DecayScheme",
    "ENSDFFileSource",
    "Feeding",
    "Gamma",
    "HalfLife",
    "JPi",
    "Level",
    "LevelScheme",
    "LivechartError",
    "LivechartSource",
    "Normalization",
    "Nuclide",
    "Parent",
    "SpinParity",
    "Uncertain",
    "__version__",
    "decay_schemes",
    "default_cache_dir",
    "ground_state",
    "level_scheme",
    "parse_half_life",
    "parse_references",
    "survey",
    "compare",
    "default_ensdf_path",
    "default_ripl_path",
    "Ripl3Source",
    "GroundState",
    "parse_spin_parity",
    "parse_value_with_uncertainty",
    "place_gammas",
]


def level_scheme(
    nuclide,
    source: str = "livechart",
    dataset: str | None = None,
    with_gammas: bool = True,
    cache=None,
    path=None,
) -> LevelScheme:
    """Fetch a level scheme for ``nuclide`` (``"24Mg"``, ``"Mg-24"``, ``(12, 24)``).

    ``source`` is ``"livechart"`` for the IAEA API, which serves adopted levels
    only; ``"file"`` for local flat files, where ``dataset`` picks one
    evaluation by DSID substring; or ``"ripl3"`` for the committed RIPL-3
    mirror (levels cut 2021, derived from ENSDF).
    """
    nuc = Nuclide.parse(nuclide)
    source = SOURCE_ALIASES.get(source, source)
    if source == "livechart":
        if dataset is not None:
            raise ValueError(
                "the Livechart API only serves adopted levels; "
                "use source='file' to select a specific dataset"
            )
        return LivechartSource(cache=cache).fetch(nuc, with_gammas=with_gammas)
    if source == "file":
        return ENSDFFileSource(path=path).fetch(nuc, dataset=dataset)
    if source == "ripl3":
        if dataset is not None:
            raise ValueError(
                "RIPL-3 carries one levels dataset per nuclide; "
                "use source='file' to select a specific ENSDF evaluation"
            )
        return Ripl3Source(path=path).fetch(nuc)
    raise ValueError(f"unknown source {source!r}")


def decay_schemes(nuclide, path=None) -> "list[DecayScheme]":
    """Every decay dataset populating ``nuclide``, from local ENSDF files.

    Decay data is only available from the flat files; the Livechart API
    serves adopted levels alone.
    """
    return ENSDFFileSource(path=path).decay_schemes(Nuclide.parse(nuclide))


def ground_state(nuclide, source: str = "livechart", cache=None, path=None) -> "GroundState":
    """Mass, abundance, charge radius and moments for ``nuclide``.

    ``source="livechart"`` aggregates AME, NUBASE, charge radii and moment
    compilations (see :data:`nook.EVALUATIONS` for attribution and
    :data:`nook.UNITS` for units -- ``atomic_mass`` is in micro-u, not u).
    ``source="ripl3"`` serves the sparser RIPL-3 mass tables: mass excess
    and abundance, with FRDM95/HFB-14 theory values in ``metadata``.
    The ENSDF flat files carry none of this.
    """
    source = SOURCE_ALIASES.get(source, source)
    if source == "livechart":
        return LivechartSource(cache=cache).ground_state(Nuclide.parse(nuclide))
    if source == "ripl3":
        return Ripl3Source(path=path).ground_state(Nuclide.parse(nuclide))
    raise ValueError(f"unknown ground-state source {source!r}")
