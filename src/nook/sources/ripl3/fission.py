"""The RIPL-3 fission segment: barriers, empirical and HFB.

* ``empirical-barriers.dat`` -- RIPL-3 recommended experimental barrier
  heights and curvatures.  Rows are ragged: light actinides list a single
  barrier with no curvature, actinides list inner + outer + pairing gap.
  The symmetry codes are letters (``S``, ``GA``, ``MA``...), which is what
  makes the ragged rows parseable: a letter token opens a new barrier group.
* ``empirical-hfb-barriers.dat`` -- HFB barriers normalised to experiment,
  fixed 12 numeric columns (inner, outer, outer2 x height/curvature/
  asymmetry/pairing shift).
* ``hfb-levden/`` -- level densities at the saddle points (Max1/Max2/Max3),
  exposed as raw tables per nuclide, not parsed into records here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...nuclide import Nuclide
from ...quantities import Uncertain
from ._util import data_lines, parse_file, require_file
from ._util import is_number as _is_number

__all__ = ["BarrierEntry", "FissionBarriers", "load_fission_barriers"]


@dataclass(frozen=True)
class BarrierEntry:
    """One saddle point: height, curvature, and the evaluator's symmetry code.

    Symmetry codes: ``S`` axially and mass symmetric, ``GA`` axially
    asymmetric, ``MA`` mass asymmetric (empirical file; the HFB file gives a
    numeric asymmetry instead).
    """

    height_mev: Uncertain
    hw_mev: Uncertain = Uncertain(None)
    symmetry: str | None = None


@dataclass(frozen=True)
class FissionBarriers:
    """Every barrier RIPL-3 carries for one nuclide, one model family."""

    nuclide: Nuclide
    model: str  # "empirical" or "hfb"
    barriers: tuple[BarrierEntry, ...]
    #: Pairing gap at the saddle (empirical Deltaf), MeV.
    pairing_gap_mev: float | None = None

    @property
    def inner(self) -> BarrierEntry | None:
        return self.barriers[0] if self.barriers else None

    @property
    def outer(self) -> BarrierEntry | None:
        return self.barriers[1] if len(self.barriers) > 1 else None


def _parse_empirical(text: str) -> dict[tuple[int, int], FissionBarriers]:
    table: dict[tuple[int, int], FissionBarriers] = {}
    for line in data_lines(text):
        tokens = line.split()
        z, a = int(tokens[0]), int(tokens[1])
        barriers: list[BarrierEntry] = []
        symmetry: str | None = None
        numbers: list[float] = []
        pairing: float | None = None

        def close_group() -> None:
            nonlocal numbers, symmetry, pairing
            if symmetry is None and not numbers:
                return
            if len(numbers) >= 3:
                # a trailing third number after hw is the Deltaf column
                pairing = numbers[2]
            height = Uncertain(numbers[0]) if numbers else Uncertain(None)
            hw = Uncertain(numbers[1]) if len(numbers) > 1 else Uncertain(None)
            barriers.append(
                BarrierEntry(height_mev=height, hw_mev=hw, symmetry=symmetry)
            )
            numbers, symmetry = [], None

        for token in tokens[3:]:
            if _is_number(token):
                numbers.append(float(token))
            else:
                close_group()
                symmetry = token
        close_group()
        table[(z, a)] = FissionBarriers(
            nuclide=Nuclide(z, a),
            model="empirical",
            barriers=tuple(barriers),
            pairing_gap_mev=pairing,
        )
    return table


def _parse_hfb(text: str) -> dict[tuple[int, int], FissionBarriers]:
    table: dict[tuple[int, int], FissionBarriers] = {}
    for line in data_lines(text):
        tokens = line.split()
        z, a = int(tokens[0]), int(tokens[2])  # Z N A s ...
        values = [float(t) for t in tokens[4:]]
        # groups of (height, hw, asymmetry, pairing shift); most nuclides
        # carry inner + outer, some a second outer barrier
        barriers = tuple(
            BarrierEntry(
                height_mev=Uncertain(values[i]),
                hw_mev=Uncertain(values[i + 1]),
                symmetry=None,
            )
            for i in range(0, len(values) - len(values) % 4, 4)
        )
        table[(z, a)] = FissionBarriers(
            nuclide=Nuclide(z, a),
            model="hfb",
            barriers=barriers,
        )
    return table


def load_fission_barriers(
    path: Path, nuclide: Nuclide, model: str = "empirical"
) -> FissionBarriers:
    """Barriers for one nuclide from the chosen model family."""
    name = {
        "empirical": "empirical-barriers.dat",
        "hfb": "empirical-hfb-barriers.dat",
    }.get(model)
    if name is None:
        raise ValueError(f"unknown fission barrier model {model!r}")
    file = path / "fission" / name
    require_file(file, "RIPL-3 fission barrier file")
    parser = _parse_empirical if model == "empirical" else _parse_hfb
    entry = parse_file(parser, file).get((nuclide.z, nuclide.a))
    if entry is None:
        raise LookupError(f"no RIPL-3 {model} fission barriers for {nuclide}")
    return entry
