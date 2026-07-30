"""The RIPL-3 gamma segment: GDR parameters and gamma-ray strength functions.

Three parsed tables:

* ``gdr-parameters&errors-exp-{SLO,MLO}.dat`` -- experimental Lorentzian fits,
  one or two resonances per row, with the *errors on a continuation line*
  aligned under the values.  Token counts distinguish the shapes (11 tokens =
  one resonance, 14 = two; error lines have 3 or 6).
* ``gdr-parameters-theor.dat`` -- theoretical two-Lorentzian parameters for
  6000+ nuclei (no uncertainties).
* ``gamma-strength-micro/z*.dat`` -- microscopic (HFB+QRPA) E1 strength
  tables, f(E1) in mb/MeV on an energy grid per nuclide.

A deformed nucleus legitimately has two GDR components; ``GDREntry`` stores
one component per record and a shared ``(z, a)`` key groups them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...nuclide import Nuclide
from ...quantities import Uncertain
from ._util import data_lines, parse_file, require_file, z_blocks
from ._util import is_number as _is_number

__all__ = [
    "GDREntry",
    "GSFTable",
    "load_gdr",
    "load_gsf",
    "parse_gdr_exp",
    "parse_gdr_theor",
]


@dataclass(frozen=True)
class GDREntry:
    """One giant-dipole-resonance fit for one nuclide."""

    nuclide: Nuclide
    kind: str  # "exp-SLO", "exp-MLO", or "theor"
    #: (energy MeV, width MeV, peak cross-section mb or None) per Lorentzian.
    peaks: tuple[tuple[Uncertain, Uncertain, Uncertain | None], ...]
    #: Deformation-driven splitting parameter eta (theory table only).
    eta: float | None = None
    fit_range_mev: tuple[float, float] | None = None
    reference: str = ""


@dataclass(frozen=True)
class GSFTable:
    """A microscopic E1 strength function on an energy grid."""

    nuclide: Nuclide
    #: (E MeV, f(E1) mb/MeV) rows.
    rows: tuple[tuple[float, float], ...]


def parse_gdr_exp(text: str, kind: str) -> dict[tuple[int, int], tuple[GDREntry, ...]]:
    """Parse an experimental GDR file, pairing each row with its error row."""
    table: dict[tuple[int, int], list[GDREntry]] = {}
    pending: tuple[tuple[int, int], GDREntry] | None = None
    for line in data_lines(text):
        tokens = line.split()
        if not _is_number(tokens[0]):
            continue
        if (
            pending is not None
            and len(tokens) in (3, 6)
            and all(_is_number(t) for t in tokens)
        ):
            # the continuation line: errors under E1 CS1 W1 [E2 CS2 W2]
            key, entry = pending
            errs = [float(t) for t in tokens]
            peaks = []
            for i, (e, w, cs) in enumerate(entry.peaks):
                chunk = errs[3 * i : 3 * i + 3]
                de, dcs, dw = chunk + [None] * (3 - len(chunk))
                peaks.append(
                    (
                        Uncertain(e.value, de, de, raw=e.raw),
                        Uncertain(w.value, dw, dw, raw=w.raw),
                        None
                        if cs is None
                        else Uncertain(cs.value, dcs, dcs, raw=cs.raw),
                    )
                )
            table[key][-1] = GDREntry(
                nuclide=entry.nuclide,
                kind=entry.kind,
                peaks=tuple(peaks),
                fit_range_mev=entry.fit_range_mev,
                reference=entry.reference,
            )
            pending = None
            continue

        z, a = int(tokens[0]), int(tokens[1])
        # tokens: Z A El Reac E1 CS1 W1 [E2 CS2 W2] Emin - Emax Ref
        numbers = tokens[4:-4]
        emin, emax = float(tokens[-4]), float(tokens[-2])
        reference = tokens[-1]
        peaks = []
        for i in range(0, len(numbers), 3):
            e, cs, w = (float(n) for n in numbers[i : i + 3])
            peaks.append(
                (
                    Uncertain(e, raw=f"{e} MeV"),
                    Uncertain(w, raw=f"{w} MeV"),
                    Uncertain(cs, raw=f"{cs} mb"),
                )
            )
        entry = GDREntry(
            nuclide=Nuclide(z, a),
            kind=kind,
            peaks=tuple(peaks),
            fit_range_mev=(emin, emax),
            reference=reference,
        )
        table.setdefault((z, a), []).append(entry)
        pending = ((z, a), entry)
    return {key: tuple(entries) for key, entries in table.items()}


def parse_gdr_theor(text: str) -> dict[tuple[int, int], GDREntry]:
    """``gdr-parameters-theor.dat``: Z A El eta E1 W1 E2 W2."""
    table: dict[tuple[int, int], GDREntry] = {}
    for line in data_lines(text):
        tokens = line.split()
        z, a = int(tokens[0]), int(tokens[1])
        eta, e1, w1, e2, w2 = (float(t) for t in tokens[3:8])
        peaks = [(Uncertain(e1, raw=f"{e1} MeV"), Uncertain(w1, raw=f"{w1} MeV"), None)]
        if (e2, w2) != (e1, w1):
            peaks.append(
                (Uncertain(e2, raw=f"{e2} MeV"), Uncertain(w2, raw=f"{w2} MeV"), None)
            )
        table[(z, a)] = GDREntry(
            nuclide=Nuclide(z, a),
            kind="theor",
            peaks=tuple(peaks),
            eta=eta,
        )
    return table


def load_gdr(path: Path, nuclide: Nuclide) -> tuple[GDREntry, ...]:
    """Every GDR parameterisation for one nuclide: experiment first, then theory."""
    key = (nuclide.z, nuclide.a)
    out: list[GDREntry] = []
    for shape in ("SLO", "MLO"):
        file = path / "gamma" / f"gdr-parameters&errors-exp-{shape}.dat"
        if file.is_file():
            out.extend(parse_file(parse_gdr_exp, file, f"exp-{shape}").get(key, ()))
    theor_file = path / "gamma" / "gdr-parameters-theor.dat"
    if theor_file.is_file():
        entry = parse_file(parse_gdr_theor, theor_file).get(key)
        if entry is not None:
            out.append(entry)
    if not out:
        raise LookupError(f"no RIPL-3 GDR parameters for {nuclide}")
    return tuple(out)


def _parse_gsf_file(
    text: str,
) -> dict[tuple[int, int], tuple[tuple[float, float], ...]]:
    table: dict[tuple[int, int], tuple[tuple[float, float], ...]] = {}
    for z, a, _header, body in z_blocks(text, lambda line: line.startswith(" Z=")):
        rows = []
        for line in body:
            parts = line.split()
            if len(parts) == 2 and not line.lstrip().startswith("U["):
                rows.append((float(parts[0]), float(parts[1])))
        table[(z, a)] = tuple(rows)
    return table


def load_gsf(path: Path, nuclide: Nuclide) -> GSFTable:
    """The microscopic E1 strength table for one nuclide."""
    file = path / "gamma" / "gamma-strength-micro" / f"z{nuclide.z:03d}.dat"
    require_file(file, "RIPL-3 microscopic strength table")
    rows = parse_file(_parse_gsf_file, file).get((nuclide.z, nuclide.a))
    if not rows:
        raise LookupError(f"no microscopic strength function for {nuclide}")
    return GSFTable(nuclide=nuclide, rows=rows)
