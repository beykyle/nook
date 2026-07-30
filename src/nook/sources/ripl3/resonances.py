"""The RIPL-3 resonances segment: average neutron resonance parameters.

``resonances0.dat`` (s-wave, 300 target nuclides) and ``resonances1.dat``
(p-wave, 119).  Stated format
``(i3,1x,a2,1x,i3,2x,f3.1,2x,f6.3,2x,2(e8.2,2x),1x,2(f4.2,2x),2(f4.1,1x),2x,a4)``.

Two things worth naming: spacings ``D0``/``D1`` are in **keV** (the strength
functions are the usual 1e-4, widths meV), and absent values are a literal
``-``, not a blank -- ``float_field`` maps both to ``None``.  An asterisk on
the reference marks nuclides whose D0 was *estimated from D1*, which is a
provenance downgrade worth keeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...nuclide import Nuclide, symbol_to_z
from ...quantities import Uncertain
from ._util import (
    MEV_TO_KEV,
    data_lines,
    field,
    float_field,
    int_field,
    parse_file,
    require_file,
)

__all__ = ["ResonanceEntry", "load_resonances", "parse_resonances"]


@dataclass(frozen=True)
class ResonanceEntry:
    """Average resonance parameters for one target nuclide and one wave."""

    nuclide: Nuclide  # the *target*; Bn refers to the compound system
    wave: int  # 0 (s) or 1 (p)
    target_spin: float | None
    bn_kev: float | None  # neutron binding of the compound nucleus
    spacing_kev: Uncertain = Uncertain(None)  # D0 or D1
    strength_1e4: Uncertain = Uncertain(None)  # S0 or S1
    gamma_width_mev_milli: Uncertain = Uncertain(None)  # <Gamma_gamma>, meV
    #: True when D0 was estimated from D1 rather than measured (the ``*``).
    estimated: bool = False
    reference: str = ""


def parse_resonances(text: str, wave: int) -> dict[tuple[int, int], ResonanceEntry]:
    table: dict[tuple[int, int], ResonanceEntry] = {}
    for line in data_lines(text):
        z = int_field(line, 0, 3)
        a = int_field(line, 7, 10)
        if z is None or a is None:
            continue
        # Cross-check the Z column against the element symbol: resonances1.dat
        # files Ar-40 as ` 20 Ar  40`, which would collide with (and lose to)
        # the genuine Ca-40 row.  The symbol agrees with A and Bn there, so it
        # wins on a mismatch.
        try:
            z = symbol_to_z(field(line, 4, 6))
        except ValueError:
            pass  # unreadable symbol: keep the Z column
        d = float_field(line, 25, 33)
        dd = float_field(line, 35, 43)
        s = float_field(line, 45, 50)
        ds = float_field(line, 50, 56)
        gg = float_field(line, 56, 62)
        dgg = float_field(line, 62, 67)
        ref = field(line, 67, 80)
        bn_mev = float_field(line, 16, 23)
        table[(z, a)] = ResonanceEntry(
            nuclide=Nuclide(z, a),
            wave=wave,
            target_spin=float_field(line, 11, 15),
            bn_kev=None if bn_mev is None else bn_mev * MEV_TO_KEV,
            spacing_kev=Uncertain(d, dd, dd, raw=line[25:43].strip()),
            strength_1e4=Uncertain(s, ds, ds, raw=line[45:56].strip()),
            gamma_width_mev_milli=Uncertain(gg, dgg, dgg, raw=line[56:67].strip()),
            estimated=ref.startswith("*"),
            reference=ref.lstrip("*").strip(),
        )
    return table


def load_resonances(path: Path, nuclide: Nuclide, wave: int = 0) -> ResonanceEntry:
    """The s-wave (``wave=0``) or p-wave (``wave=1``) entry for a target."""
    file = path / "resonances" / f"resonances{wave}.dat"
    require_file(file, "RIPL-3 resonance file")
    entry = parse_file(parse_resonances, file, wave).get((nuclide.z, nuclide.a))
    if entry is None:
        raise LookupError(f"no RIPL-3 {'sp'[wave]}-wave resonance data for {nuclide}")
    return entry
