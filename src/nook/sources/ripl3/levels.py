"""The RIPL-3 discrete-levels segment: per-element ``z0NN.dat`` files.

Format per ``levels-readme.html`` in the mirror (R. Capote and M. Verpelli,
2021-12-13).  Each isotope is an identification record ``(a5,6i5,2f12.6)``,
then one level record per level, each followed by its gamma records
``(39x,i4,1x,f10.4,3(1x,e10.3))``.

Traps the readme is explicit about, preserved here rather than smoothed over:

* Energies are in **MeV**; the shared model speaks keV, so values convert at
  parse time and ``raw`` keeps the file's numbers.
* Spin ``-1.0`` and parity ``0`` mean *unknown* -- but for levels below
  ``Nmax`` RIPL **invents** unique values from a spin distribution and, for
  parity, a coin flip.  The estimation method is coded in a flag: ``u``
  means ENSDF offered exactly one compatible spin (possibly tentative, like
  ``(1)+``) and RIPL adopted it; ``g`` means it was inferred from gamma
  transitions to known levels; ``c`` chosen from a listed set by the spin
  distribution; ``n`` no experimental information at all.  A ``g``, ``c``
  or ``n`` assignment is RIPL's, not the evaluator's, and is marked
  ``assumed``; a ``u`` assignment is ENSDF's own single candidate, so only
  its tentativeness survives (in ``tentative``).  The original ENSDF J
  field is kept alongside either way.  (``g`` is undocumented in the
  readme but 5000+ levels in the 2021 cut carry it.)
* Stable levels carry ``T1/2 = -1.0E+0``, not blank.
* ``Pg`` may be 0 when ENSDF gave no branching; ``Pe = Pg * (1 + ICC)`` and
  the ``Pe`` sum over a level is its electromagnetic branching, 1 unless the
  level record lists other decay modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...model import Gamma, Level, LevelScheme
from ...nuclide import Nuclide
from ...quantities import HalfLife, JPi, SpinParity, Uncertain, parse_spin_parity
from ._util import MEV_TO_KEV, field, float_field, int_field, parse_file, require_file

__all__ = ["LevelsParam", "RiplNuclideLevels", "parse_levels_file", "parse_levels_param"]


@dataclass(frozen=True)
class RiplNuclideLevels:
    """One isotope's block from a per-element file, still close to the file."""

    nuclide: Nuclide
    n_levels: int          # Nol
    n_gammas: int          # Nog
    n_complete: int        # Nmax: scheme believed complete up to this level
    n_unique_jpi: int      # Nc: spins and parities unique up to this level
    sn_kev: float | None
    sp_kev: float | None
    levels: tuple[Level, ...] = ()
    gammas: tuple[Gamma, ...] = ()

    def to_scheme(self) -> LevelScheme:
        return LevelScheme(
            nuclide=self.nuclide,
            levels=self.levels,
            gammas=self.gammas,
            source="ripl3-local",
            dataset="RIPL-3 LEVELS",
            metadata={
                "nol": self.n_levels,
                "nog": self.n_gammas,
                "nmax": self.n_complete,
                "nc": self.n_unique_jpi,
                "sn_kev": self.sn_kev,
                "sp_kev": self.sp_kev,
                "cut": "2021",
            },
        )


@dataclass(frozen=True)
class LevelsParam:
    """One row of ``levels-param-2021.data``: the constant-temperature fit."""

    nuclide: Nuclide
    temperature_mev: float | None
    temperature_unc_mev: float | None
    u0_mev: float | None
    u0_unc_mev: float | None
    n_levels: int | None
    n_max: int | None
    n0: int | None
    n_c: int | None
    u_max_mev: float | None
    u_c_mev: float | None


def _half_life(text: str) -> HalfLife:
    """RIPL half-lives are seconds; ``-1.0E+0`` flags a stable level."""
    if not text:
        return HalfLife()
    value = float(text)
    if value < 0:
        return HalfLife(seconds=Uncertain(float("inf"), raw=text), stable=True, raw=text)
    return HalfLife(seconds=Uncertain(value, raw=text), raw=text)


def _spin_parity(spin: float | None, parity: int | None, flag: str, original: str) -> SpinParity:
    """Combine RIPL's unique pick with the original ENSDF J field.

    RIPL's numeric spin wins for :attr:`~nook.quantities.SpinParity.candidates`
    because that is what RIPL asserts; the ENSDF field supplies tentativeness
    and survives in ``raw``.  A level with no experimental information at all
    (flag ``n``, empty J field) still gets numbers from RIPL's spin
    distribution -- callers who care can check ``Level.properties``.

    ``assumed`` marks the flags where the *value* is RIPL's inference (``g``
    from gamma cascades, ``c`` from the spin distribution over a listed set,
    ``n`` pure spin-distribution invention).  ``u`` is excluded on purpose:
    there ENSDF itself offered exactly one candidate -- every firm unique
    ENSDF assignment below Emax is u-flagged, so marking ``u`` assumed would
    label the entire well-known part of a scheme as RIPL's invention.
    """
    if spin is None or spin < 0:
        # RIPL itself gave up; fall back to whatever ENSDF said, if anything.
        return parse_spin_parity(original) if original else SpinParity()
    ensdf = parse_spin_parity(original) if original else SpinParity()
    return SpinParity(
        candidates=(JPi(round(2 * spin), parity if parity else None),),
        tentative=ensdf.tentative,
        assumed=flag in ("g", "c", "n"),
        raw=original,
    )


def _decay_modes(line: str, count: int) -> tuple[tuple[str, Uncertain], ...]:
    """The ``nd`` repeated groups ``(1x,a2,1x,e10.4,1x,a7)`` from column 66."""
    modes = []
    for i in range(count):
        base = 66 + 22 * i
        modifier = field(line, base + 1, base + 3)
        percent = float_field(line, base + 4, base + 14)
        name = field(line, base + 15, base + 22)
        if not name:
            continue
        operator = None if modifier in ("", "=") else modifier
        modes.append((name, Uncertain(percent, operator=operator, raw=line[base:base + 22].strip())))
    return tuple(modes)


def parse_levels_file(text: str) -> tuple[RiplNuclideLevels, ...]:
    """Parse one per-element file into its isotope blocks."""
    nuclides: list[RiplNuclideLevels] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # Identification record: (a5,6i5,2f12.6)
        a = int_field(line, 5, 10)
        z = int_field(line, 10, 15)
        nol = int_field(line, 15, 20)
        nog = int_field(line, 20, 25)
        nmax = int_field(line, 25, 30)
        nc = int_field(line, 30, 35)
        sn = float_field(line, 35, 47)
        sp = float_field(line, 47, 59)
        if a is None or z is None or nol is None:
            raise ValueError(f"not a RIPL identification record: {line!r}")
        i += 1

        levels: list[Level] = []
        gammas: list[Gamma] = []
        for _ in range(nol):
            row = lines[i]
            index = int_field(row, 0, 3)
            energy_mev = float_field(row, 4, 14)
            spin = float_field(row, 15, 20)
            parity = int_field(row, 20, 23)
            half_life_text = field(row, 24, 34)
            n_gammas = int_field(row, 34, 37) or 0
            j_flag = field(row, 38, 39)
            unc_flag = field(row, 40, 44)
            original_jpi = field(row, 45, 63)
            nd = int_field(row, 63, 66) or 0
            shift_mev = float_field(row, 286, 296)
            band = field(row, 297, 303)
            if index is None or energy_mev is None:
                raise ValueError(f"not a RIPL level record: {row!r}")
            i += 1

            properties: dict[str, str] = {}
            if j_flag:
                properties["ripl_spin_flag"] = j_flag
            if unc_flag:
                properties["ripl_energy_flag"] = unc_flag
            if shift_mev:
                properties["ripl_shift_mev"] = f"{shift_mev}"

            levels.append(
                Level(
                    index=index,
                    energy=Uncertain(energy_mev * MEV_TO_KEV, raw=f"{energy_mev:.6f} MeV"),
                    spin_parity=_spin_parity(spin, parity, j_flag, original_jpi),
                    half_life=_half_life(half_life_text),
                    # A set `unc` flag (X, Y, ... SN, SP) says the absolute
                    # position is unknown: the printed energy is band-relative
                    # (or a copy of the previous level's).  That is exactly the
                    # ENSDF floating-level situation, so expose it the same
                    # way; `below()`/`compare` then treat both chains alike.
                    energy_offset=unc_flag or None,
                    band=band or None,
                    decay_modes=_decay_modes(row, nd),
                    dataset="RIPL-3 LEVELS",
                    properties=properties,
                    raw={"ensdf_jpi": original_jpi, "spin": spin, "parity": parity},
                )
            )
            for _ in range(n_gammas):
                grow = lines[i]
                final = int_field(grow, 39, 43)
                eg_mev = float_field(grow, 44, 54)
                pg = float_field(grow, 55, 65)
                pe = float_field(grow, 66, 76)
                icc = float_field(grow, 77, 87)
                i += 1
                gammas.append(
                    Gamma(
                        energy=Uncertain(
                            None if eg_mev is None else eg_mev * MEV_TO_KEV,
                            raw=f"{eg_mev} MeV",
                        ),
                        start_index=index,
                        end_index=final,
                        intensity=None if pg is None else Uncertain(pg, raw=f"{pg}"),
                        conversion_coefficient=(
                            None if icc is None else Uncertain(icc, raw=f"{icc}")
                        ),
                        properties={} if pe is None else {"pe": f"{pe}"},
                        raw={"pg": pg, "pe": pe, "icc": icc},
                    )
                )

        nuclides.append(
            RiplNuclideLevels(
                nuclide=Nuclide(z, a),
                n_levels=nol,
                n_gammas=nog or 0,
                n_complete=nmax or 0,
                n_unique_jpi=nc or 0,
                sn_kev=None if sn is None else sn * MEV_TO_KEV,
                sp_kev=None if sp is None else sp * MEV_TO_KEV,
                levels=tuple(levels),
                gammas=tuple(gammas),
            )
        )
    return tuple(nuclides)


def load_levels(path: Path, nuclide: Nuclide) -> RiplNuclideLevels:
    """The block for one nuclide from its per-element file."""
    file = path / "levels" / f"z{nuclide.z:03d}.dat"
    require_file(file, "RIPL-3 levels file")
    for block in parse_file(parse_levels_file, file):
        if block.nuclide == nuclide:
            return block
    raise LookupError(f"no RIPL-3 levels for {nuclide}")


def parse_levels_param(text: str) -> dict[tuple[int, int], LevelsParam]:
    """Parse ``levels-param-2021.data`` (constant-temperature fit table)."""
    table: dict[tuple[int, int], LevelsParam] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        z, a = int(parts[0]), int(parts[1])
        # parts[2] is the element symbol; trust Z.
        values = parts[3:13]
        table[(z, a)] = LevelsParam(
            nuclide=Nuclide(z, a),
            temperature_mev=float(values[0]),
            temperature_unc_mev=float(values[1]),
            u0_mev=float(values[2]),
            u0_unc_mev=float(values[3]),
            n_levels=int(values[4]),
            n_max=int(values[5]),
            n0=int(values[6]),
            n_c=int(values[7]),
            u_max_mev=float(values[8]),
            u_c_mev=float(values[9]),
        )
    return table
