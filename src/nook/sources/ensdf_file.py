"""Reader for the ENSDF flat files distributed by NNDC.

Livechart exposes only the adopted dataset; these fixed-column card images
carry every evaluation, so this is the backend for anything reaction-specific.

Decodes identification, ``X``, ``Q``, ``H``, ``L``, ``G``, ``P``, ``N``, ``B``,
``E``, ``A``, ``D`` and ``R`` records, plus their data continuations. Comment
records are skipped but counted, so nothing disappears silently.

Files: https://www.nndc.bnl.gov/ensarchivals/, one per mass number
(``ensdf.024``); point ``ENSDF_PATH`` at the directory.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..decay import DecayScheme, Feeding, Normalization, Parent
from ..model import Gamma, Level, LevelScheme
from ..nuclide import Nuclide
from ..quantities import (
    Uncertain,
    parse_half_life,
    parse_spin_parity,
    parse_value_with_uncertainty,
)

__all__ = [
    "ENSDFDataset",
    "ENSDFFileSource",
    "clear_chain_cache",
    "default_ensdf_path",
    "parse_ensdf_text",
    "parse_references",
]


def default_ensdf_path() -> Path:
    """``$ENSDF_PATH``, else ``~/.local/share/ensdf``."""
    env = os.environ.get("ENSDF_PATH")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base).expanduser() / "ensdf"


def _field(line: str, start: int, end: int) -> str:
    """1-based inclusive column slice, as the ENSDF manual numbers them."""
    return line[start - 1 : end].strip()


class ENSDFDataset:
    """One ENSDF dataset: an identification record and its level scheme."""

    __slots__ = (
        "band_definitions",
        "dsid",
        "feedings",
        "gammas",
        "levels",
        "normalization",
        "nuclide",
        "parents",
        "properties",
        "reference",
        "skipped",
        "xref_key",
    )

    def __init__(
        self,
        nuclide: Nuclide,
        dsid: str,
        reference: str,
        levels: Sequence[Level],
        gammas: Sequence[Gamma],
        skipped: int = 0,
        band_definitions: dict[str, str] | None = None,
        xref_key: dict[str, str] | None = None,
        properties: dict[str, object] | None = None,
        parents: Sequence[Parent] = (),
        normalization: Normalization | None = None,
        feedings: Sequence[Feeding] = (),
    ):
        self.nuclide = nuclide
        self.dsid = dsid
        self.reference = reference
        self.levels = tuple(levels)
        self.gammas = tuple(gammas)
        self.skipped = skipped
        self.band_definitions = dict(band_definitions or {})
        self.xref_key = dict(xref_key or {})
        self.properties = dict(properties or {})
        self.parents = tuple(parents)
        self.normalization = normalization
        self.feedings = tuple(feedings)

    @property
    def is_adopted(self) -> bool:
        return self.dsid.upper().startswith("ADOPTED LEVELS")

    def to_scheme(self) -> LevelScheme:
        return LevelScheme(
            nuclide=self.nuclide,
            levels=self.levels,
            gammas=self.gammas,
            source="ensdf-file",
            dataset=self.dsid,
            metadata={
                "reference": self.reference,
                "skipped_records": self.skipped,
                "band_definitions": self.band_definitions,
                "xref_key": self.xref_key,
                **self.properties,
            },
        )

    @property
    def is_decay(self) -> bool:
        """True when the dataset describes a decay (it names a parent)."""
        return bool(self.parents)

    def to_decay_scheme(self) -> DecayScheme:
        return DecayScheme(
            nuclide=self.nuclide,
            dsid=self.dsid,
            levels=self.to_scheme(),
            parents=self.parents,
            normalization=self.normalization,
            feedings=self.feedings,
            metadata={"reference": self.reference, **self.properties},
        )

    def __repr__(self) -> str:
        return (
            f"<ENSDFDataset {self.nuclide} {self.dsid!r} "
            f"levels={len(self.levels)} gammas={len(self.gammas)}>"
        )


#: ENSDF's ASCII markup for Greek letters and super/subscripts.
_GREEK = {
    "a": "\u03b1", "b": "\u03b2", "g": "\u03b3", "d": "\u03b4", "e": "\u03b5",
    "n": "\u03bd", "p": "\u03c0", "m": "\u03bc", "s": "\u03c3", "t": "\u03c4",
    "w": "\u03c9", "x": "\u03be", "l": "\u03bb", "r": "\u03c1", "h": "\u03b7",
    "D": "\u0394", "G": "\u0393", "S": "\u03a3", "W": "\u03a9", "P": "\u03a0",
}


def ensdf_markup_to_text(text: str) -> str:
    """Render ENSDF's ASCII markup as readable Unicode.

    ``|p7/2[404]`` becomes ``\u03c07/2[404]``; ``{+180}Hf`` becomes
    ``180Hf``.  Purely cosmetic, but band configurations are unreadable
    without it.
    """
    out = re.sub(r"\{[+-]([^}]*)\}", r"\1", text)
    out = re.sub(r"\|(.)", lambda m: _GREEK.get(m.group(1), m.group(1)), out)
    return out.strip()


_BAND_RE = re.compile(r"^BAND\(([^)]+)\)\$(.*)$")


def _parse_band_definitions(block: Sequence[str]) -> dict[str, str]:
    """Decode ``CL BAND(x)$...`` documentation records, continuations included."""
    definitions: dict[str, str] = {}
    current: str | None = None
    for raw_line in block:
        line = raw_line.ljust(80)
        if line[6].upper() != "C" or line[7].upper() != "L":
            current = None
            continue
        text = line[9:80].rstrip()
        if line[5] != " " and current is not None:  # continuation record
            definitions[current] = f"{definitions[current]} {text.strip()}".strip()
            continue
        m = _BAND_RE.match(text.strip())
        if m:
            label = m.group(1).strip()
            definitions[label] = m.group(2).strip()
            current = label
        else:
            current = None
    return {k: ensdf_markup_to_text(v) for k, v in definitions.items()}


_CONT_KEY = re.compile(r"^\s*([%A-Za-z][A-Za-z0-9()%+\-/ ]*?)\s*=\s*(.*)$")


def parse_continuation_properties(lines: Iterable[str]) -> dict[str, str]:
    """Decode ``KEY=value$KEY=value`` data-continuation records.

    Continuation records carry everything the fixed 80-column primary record
    has no room for: ``XREF``, moments, decay branchings, flags.  Values are
    left as text because their types vary by key.
    """
    properties: dict[str, str] = {}
    for raw_line in lines:
        for fieldset in raw_line.ljust(80)[9:80].split("$"):
            m = _CONT_KEY.match(fieldset)
            if m:
                key, value = m.group(1).strip().upper(), m.group(2).strip()
                if value:
                    properties[key] = value
    return properties


def parse_xref(value: str, key: Sequence[str] = ()) -> tuple[str, ...]:
    """Expand an ``XREF`` value into dataset symbols.

    Accepts plain letters (``ABCD``), letters flagged as uncertain in
    parentheses (``AB(C)D`` -- kept, since the level was still reported),
    and the ``ALL``/``ALL EXCEPT xy`` shorthands.
    """
    text = value.strip()
    upper = text.upper()
    if upper.startswith("ALL"):
        excluded = set(re.sub(r"[^A-Za-z]", "", upper.partition("EXCEPT")[2]))
        return tuple(sym for sym in key if sym.upper() not in excluded)
    return tuple(re.findall(r"[A-Za-z]", text))


_L_TRANSFER = re.compile(r"\d+")


def parse_transfer_l(value: str) -> tuple[int, ...]:
    """Angular momentum transfers from an ``L`` field like ``"5(+6)"``."""
    return tuple(int(v) for v in _L_TRANSFER.findall(value or ""))


def _split_datasets(lines: Iterable[str]) -> Iterator[list[str]]:
    block: list[str] = []
    for line in lines:
        if not line.strip():
            if block:
                yield block
                block = []
            continue
        block.append(line.rstrip("\n"))
    if block:
        yield block


def _is_primary(line: str, record_type: str) -> bool:
    """A primary (non-continuation, non-comment) record of the given type."""
    if len(line) < 9:
        return False
    return (
        line[5] == " "  # col 6: continuation marker
        and line[6] == " "  # col 7: comment marker
        and line[7].upper() == record_type
    )


@dataclass
class _BlockRecords:
    """Primary records of one dataset, each with its continuation lines.

    Splitting the scan from the decoding keeps a single pass over the block
    -- which is the only part that has to understand ENSDF's column-6/7
    continuation convention -- separate from the per-record-type parsers.
    """

    levels: list[tuple[str, list[str]]] = dc_field(default_factory=list)
    gammas: list[tuple[str, int, list[str]]] = dc_field(default_factory=list)
    feedings: list[tuple[str, str, int | None, list[str]]] = dc_field(default_factory=list)
    parents: list[str] = dc_field(default_factory=list)
    normalization: str | None = None
    q_records: list[str] = dc_field(default_factory=list)
    history: str | None = None
    xref_key: dict[str, str] = dc_field(default_factory=dict)
    skipped: int = 0


def _scan_block(block: Sequence[str]) -> _BlockRecords:
    """Group a dataset's records, attaching continuations to their owner."""
    found = _BlockRecords()
    current_level = -1

    for raw_line in block:
        line = raw_line.ljust(80)
        record = line[7].upper()
        continuation = line[5] != " " and line[6] == " "

        if _is_primary(line, "X"):
            found.xref_key[line[8]] = _field(line, 10, 39)
        elif _is_primary(line, "P"):
            found.parents.append(line)
        elif _is_primary(line, "N"):
            found.normalization = line
        elif _is_primary(line, "Q"):
            found.q_records.append(line)
        elif _is_primary(line, "H"):
            found.history = line
        elif record in _FEEDING_TYPES and _is_primary(line, record):
            # Feeding records follow the level they populate.
            found.feedings.append(
                (line, record, current_level if current_level >= 0 else None, [])
            )
        elif _is_primary(line, "L"):
            current_level += 1
            found.levels.append((line, []))
        elif _is_primary(line, "G"):
            found.gammas.append((line, current_level, []))
        elif continuation and record == "L" and found.levels:
            found.levels[-1][1].append(line)
        elif continuation and record == "G" and found.gammas:
            found.gammas[-1][2].append(line)
        elif continuation and record in _FEEDING_TYPES and found.feedings:
            found.feedings[-1][3].append(line)
        elif record in ("L", "G"):
            found.skipped += 1

    return found


def parse_ensdf_text(text: str) -> list[ENSDFDataset]:
    """Parse an ENSDF card-image file into datasets."""
    datasets: list[ENSDFDataset] = []

    for block in _split_datasets(text.splitlines()):
        header = block[0].ljust(80)
        try:
            nuclide = Nuclide.from_nucid(header[:5])
        except ValueError:
            continue  # not an identification record we understand
        dsid = _field(header, 10, 39)
        found = _scan_block(block[1:])

        properties: dict[str, object] = {}
        if found.q_records:
            # Most adopted datasets carry two Q records: NNDC prepends one
            # from the newest mass evaluation, then a comment saying "current
            # evaluation has used the following Q record", then the older set
            # the evaluator actually worked from. Default to the first, which
            # is the newer mass data; keep the rest for anyone who needs
            # internal consistency with the evaluation's own derived values.
            decoded = [_parse_q_record(line) for line in found.q_records]
            properties["Q"] = decoded[0]
            properties["Q_records"] = decoded
        if found.history is not None:
            properties["history"] = parse_continuation_properties([found.history])

        datasets.append(
            ENSDFDataset(
                nuclide,
                dsid,
                _field(header, 40, 65),
                [
                    _parse_level_record(line, i, dsid, conts, found.xref_key)
                    for i, (line, conts) in enumerate(found.levels)
                ],
                [
                    _parse_gamma_record(line, idx, conts)
                    for line, idx, conts in found.gammas
                ],
                found.skipped,
                band_definitions=_parse_band_definitions(block[1:]),
                xref_key=found.xref_key,
                properties=properties,
                parents=[_parse_parent_record(line) for line in found.parents],
                normalization=(
                    None
                    if found.normalization is None
                    else _parse_normalization_record(found.normalization)
                ),
                feedings=[
                    _parse_feeding_record(line, kind, idx, conts)
                    for line, kind, idx, conts in found.feedings
                ],
            )
        )

    return datasets


#: Offset symbols for levels whose position relative to the ground state is
#: unknown, in the order the manual (V.18) says evaluators should use them.
_OFFSET_MARKERS = ("SN", "SP", "X", "Y", "Z", "U", "V", "W")


def split_energy_offset(text: str) -> tuple[str, str | None]:
    """Separate an ENSDF energy field into its number and offset symbol.

    Manual V.18 allows ``NUM``, ``NUM+A``, ``A+NUM``, ``A`` alone, and the
    ``SN``/``SP`` forms.  The suffix spelling is easy to miss: ``4172.3+X``
    is a real level 4172.3 keV above an unknown reference, and treating the
    whole string as a number silently discards the energy.
    """
    body = text.strip()
    if not body:
        return "", None
    upper = body.upper()
    for marker in _OFFSET_MARKERS:
        if upper.startswith(marker):
            return body[len(marker) :].lstrip().lstrip("+").strip(), marker
    for marker in _OFFSET_MARKERS:
        if upper.endswith("+" + marker):
            return body[: -(len(marker) + 1)].strip(), marker
    return body, None


def _parse_level_record(
    line: str,
    index: int,
    dsid: str,
    continuations: Sequence[str] = (),
    xref_key: dict[str, str] | None = None,
) -> Level:
    energy_text, offset = split_energy_offset(_field(line, 10, 19))

    properties = parse_continuation_properties(continuations)
    key = xref_key or {}
    xref = parse_xref(properties["XREF"], tuple(key)) if "XREF" in properties else ()
    observed = tuple(key[sym] for sym in xref if sym in key)

    spectroscopic = parse_value_with_uncertainty(
        _field(line, 65, 74), _field(line, 75, 76)
    )

    def moment(key: str) -> "Uncertain | None":
        """Continuation moments are written value-then-uncertainty in one field."""
        text = properties.get(key)
        if not text:
            return None
        parts = text.split()
        return parse_value_with_uncertainty(parts[0], parts[1] if len(parts) > 1 else "")

    return Level(
        index=index,
        energy=parse_value_with_uncertainty(energy_text, _field(line, 20, 21)),
        spin_parity=parse_spin_parity(_field(line, 22, 39)),
        half_life=parse_half_life(_field(line, 40, 49), _field(line, 50, 55)),
        energy_offset=offset,
        band=_field(line, 77, 77) or None,
        metastable=bool(_field(line, 78, 79)),
        questionable=_field(line, 80, 80) == "?",
        dataset=dsid,
        xref=xref,
        observed_in=observed,
        transfer_l=parse_transfer_l(_field(line, 56, 64)),
        transfer_l_raw=_field(line, 56, 64) or None,
        spectroscopic_factor=spectroscopic if spectroscopic.value is not None else None,
        magnetic_dipole=moment("MOMM1"),
        electric_quadrupole=moment("MOME2"),
        properties=properties,
        raw={
            "record": line.rstrip(),
            "L": _field(line, 56, 64),
            "S": _field(line, 65, 74),
            "MS": _field(line, 78, 79),
        },
    )


#: Q-record columns: Q(beta-), S(n), S(p), Q(alpha), each with an uncertainty.
_Q_FIELDS = (
    ("q_beta_minus", 10, 19, 20, 21),
    ("s_n", 22, 29, 30, 31),
    ("s_p", 32, 39, 40, 41),
    ("q_alpha", 42, 49, 50, 55),
)


def _parse_q_record(line: str) -> dict[str, object]:
    """Decode the Q record: separation energies and decay Q-values, in keV."""
    out: dict[str, object] = {
        name: parse_value_with_uncertainty(
            _field(line, a, b), _field(line, c, d)
        )
        for name, a, b, c, d in _Q_FIELDS
    }
    out["reference"] = _field(line, 56, 80)
    return out


def _parse_parent_record(line: str) -> Parent:
    """``P`` record. Its NUCID is the *parent*, not the dataset's nuclide."""
    try:
        nuclide = Nuclide.from_nucid(line[:5])
    except ValueError:
        nuclide = None
    return Parent(
        nuclide=nuclide,
        energy=parse_value_with_uncertainty(_field(line, 10, 19), _field(line, 20, 21)),
        spin_parity=parse_spin_parity(_field(line, 22, 39)),
        half_life=parse_half_life(_field(line, 40, 49), _field(line, 50, 55)),
        q_value=parse_value_with_uncertainty(_field(line, 65, 74), _field(line, 75, 76)),
        ionisation=_field(line, 77, 80) or None,
        raw={"record": line.rstrip()},
    )


def _maybe(line: str, a: int, b: int, c: int, d: int):
    """Parse a value/uncertainty column pair, or ``None`` if the field is blank."""
    text = _field(line, a, b)
    if not text:
        return None
    return parse_value_with_uncertainty(text, _field(line, c, d))


def _parse_normalization_record(line: str) -> Normalization:
    """``N`` record: NR, NT, BR, NB, NP with their uncertainties."""
    return Normalization(
        photon=_maybe(line, 10, 19, 20, 21),
        transition=_maybe(line, 22, 29, 30, 31),
        branching=_maybe(line, 32, 39, 40, 41),
        feeding=_maybe(line, 42, 49, 50, 55),
        delayed=_maybe(line, 56, 62, 63, 64),
        raw={"record": line.rstrip()},
    )


def _parse_feeding_record(
    line: str, kind: str, level_index: int | None, continuations: Sequence[str] = ()
) -> Feeding:
    """``B``, ``E``, ``A`` or ``D`` record, attached to the preceding level.

    Column meanings diverge by record type: columns 32-41 hold the EC
    intensity on an ``E`` record, the hindrance factor on an ``A`` record and
    the intermediate-level energy on a ``D`` record.  They are decoded per
    kind rather than pooled.
    """
    width = parse_half_life(_field(line, 40, 49), _field(line, 50, 55))
    return Feeding(
        kind=kind,
        level_index=level_index,
        energy=parse_value_with_uncertainty(_field(line, 10, 19), _field(line, 20, 21)),
        intensity=parse_value_with_uncertainty(
            _field(line, 22, 29), _field(line, 30, 31)
        ),
        ec_intensity=_maybe(line, 32, 39, 40, 41) if kind == "E" else None,
        total_intensity=_maybe(line, 65, 74, 75, 76) if kind == "E" else None,
        log_ft=_maybe(line, 42, 49, 50, 55) if kind in ("B", "E") else None,
        hindrance_factor=_maybe(line, 32, 39, 40, 41) if kind == "A" else None,
        forbiddenness=(_field(line, 78, 79) or None) if kind in ("B", "E") else None,
        particle=(line[8].strip() or None) if kind == "D" else None,
        intermediate_energy=_maybe(line, 32, 39, 40, 41) if kind == "D" else None,
        width=(width.width_mev if width.from_width else None) if kind == "D" else None,
        transfer_l=parse_transfer_l(_field(line, 56, 64)) if kind == "D" else (),
        questionable=_field(line, 80, 80) == "?",
        properties=parse_continuation_properties(continuations),
        raw={"record": line.rstrip()},
    )


_FEEDING_TYPES = ("B", "E", "A", "D")


def parse_references(text: str) -> dict[str, str]:
    """Decode ``R`` records, mapping NSR keynumbers to their citations."""
    references: dict[str, str] = {}
    for block in _split_datasets(text.splitlines()):
        # Skip block[0]: an identification record such as "180    REFERENCES"
        # has an R in column 8 and would otherwise parse as a reference.
        for line in block[1:]:
            padded = line.ljust(80)
            if _is_primary(padded, "R"):
                key = _field(padded, 10, 17)
                if key:
                    references[key] = _field(padded, 19, 80)
    return references


def _parse_gamma_record(
    line: str, start_index: int, continuations: Sequence[str] = ()
) -> Gamma:
    properties = parse_continuation_properties(continuations)
    final_energy = None
    if "FL" in properties:
        try:
            final_energy = float(properties["FL"].split()[0])
        except (ValueError, IndexError):
            final_energy = None

    return Gamma(
        energy=parse_value_with_uncertainty(_field(line, 10, 19), _field(line, 20, 21)),
        start_index=start_index if start_index >= 0 else None,
        end_index=None,  # ENSDF does not state it; infer via place_gammas()
        intensity=parse_value_with_uncertainty(
            _field(line, 22, 29), _field(line, 30, 31)
        ),
        multipolarity=_field(line, 32, 41) or None,
        mixing_ratio=parse_value_with_uncertainty(
            _field(line, 42, 49), _field(line, 50, 55)
        ),
        conversion_coefficient=parse_value_with_uncertainty(
            _field(line, 56, 62), _field(line, 63, 64)
        ),
        final_level_energy=final_energy,
        properties=properties,
        raw={
            "record": line.rstrip(),
            "TI": _field(line, 65, 74),
            "COIN": _field(line, 78, 78),
        },
    )


def place_gammas(scheme: LevelScheme, tolerance_kev: float = 1.0) -> LevelScheme:
    """Assign ``end_index`` to gammas by energy matching.

    Flat-file ``G`` records give the transition energy and the level they
    depopulate, but not the level they feed.  This resolves the final level
    as the one closest to ``E_start - E_gamma`` within ``tolerance_kev``,
    which is what evaluators intend; ambiguous cases are left unplaced.
    """
    energies = {
        lv.index: lv.energy_kev
        for lv in scheme.levels
        if lv.energy_kev is not None and not lv.is_floating
    }

    placed = []
    for gamma in scheme.gammas:
        start_e = (
            None if gamma.start_index is None else energies.get(gamma.start_index)
        )
        if start_e is None or gamma.energy_kev is None:
            placed.append(gamma)
            continue
        stated = gamma.final_level_energy
        target = stated if stated is not None else start_e - gamma.energy_kev
        best, best_diff = None, 0.05 if stated is not None else tolerance_kev
        for idx, energy in energies.items():
            diff = abs(energy - target)
            if diff <= best_diff:
                best, best_diff = idx, diff
        placed.append(
            Gamma(
                energy=gamma.energy,
                start_index=gamma.start_index,
                end_index=best,
                intensity=gamma.intensity,
                multipolarity=gamma.multipolarity,
                mixing_ratio=gamma.mixing_ratio,
                conversion_coefficient=gamma.conversion_coefficient,
                final_level_energy=gamma.final_level_energy,
                properties=gamma.properties,
                raw=gamma.raw,
            )
        )

    return LevelScheme(
        nuclide=scheme.nuclide,
        levels=scheme.levels,
        gammas=tuple(placed),
        source=scheme.source,
        dataset=scheme.dataset,
        metadata=scheme.metadata,
    )


def place_gammas_in(scheme: DecayScheme) -> DecayScheme:
    """Run :func:`place_gammas` on a decay scheme's daughter levels."""
    return DecayScheme(
        nuclide=scheme.nuclide,
        dsid=scheme.dsid,
        levels=place_gammas(scheme.levels),
        parents=scheme.parents,
        normalization=scheme.normalization,
        feedings=scheme.feedings,
        metadata=scheme.metadata,
    )


@lru_cache(maxsize=8)
def _parse_chain_cached(path: str, fingerprint: tuple[int, int]) -> tuple[ENSDFDataset, ...]:
    del fingerprint  # part of the cache key only
    return tuple(parse_ensdf_text(Path(path).read_text(encoding="utf-8", errors="replace")))


def _parse_chain(path: Path) -> tuple[ENSDFDataset, ...]:
    """Parse a mass chain, reusing the result while the file is unchanged.

    A chain file is ~1.5 MB and parsing it takes a few hundred milliseconds,
    which is paid on every lookup otherwise -- four nuclides from one chain
    cost four full parses.  The size and mtime are part of the cache key, so
    editing or replacing the file invalidates it.
    """
    stat = path.stat()
    return _parse_chain_cached(str(path.resolve()), (stat.st_size, stat.st_mtime_ns))


def clear_chain_cache() -> None:
    """Forget every parsed mass chain."""
    _parse_chain_cached.cache_clear()


class ENSDFFileSource:
    """Read level schemes from a local ENSDF archival distribution."""

    name = "ensdf-file"

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else default_ensdf_path()

    def _file_for(self, nuclide: Nuclide) -> Path:
        candidate = self.path / f"ensdf.{nuclide.a:03d}"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"no ENSDF mass chain file at {candidate}. Download the archival "
                f"files from https://www.nndc.bnl.gov/ensarchivals/ and set "
                f"$ENSDF_PATH to their directory."
            )
        return candidate

    def datasets(self, nuclide: Nuclide) -> list[ENSDFDataset]:
        """Every dataset for ``nuclide`` in its mass chain file."""
        return [ds for ds in _parse_chain(self._file_for(nuclide)) if ds.nuclide == nuclide]

    def references(self, mass_number: int) -> dict[str, str]:
        """NSR keynumber -> citation, from the mass chain's REFERENCES dataset."""
        path = self.path / f"ensdf.{mass_number:03d}"
        if not path.is_file():
            raise FileNotFoundError(f"no ENSDF mass chain file at {path}")
        return parse_references(path.read_text(encoding="utf-8", errors="replace"))

    def decay_schemes(self, nuclide: Nuclide) -> list[DecayScheme]:
        """Every dataset for this nuclide that describes a decay into it."""
        return [
            place_gammas_in(ds.to_decay_scheme())
            for ds in self.datasets(nuclide)
            if ds.is_decay
        ]

    def fetch(
        self, nuclide: Nuclide, dataset: str | None = None, place: bool = True
    ) -> LevelScheme:
        """Return one dataset; defaults to ``ADOPTED LEVELS``.

        ``dataset`` is matched case-insensitively as a substring of the DSID,
        so ``"(P,T)"`` or ``"COULOMB"`` are enough.
        """
        found = self.datasets(nuclide)
        if not found:
            raise LookupError(f"no ENSDF datasets for {nuclide}")
        if dataset is None:
            chosen = next((ds for ds in found if ds.is_adopted), found[0])
        else:
            needle = dataset.upper()
            matches = [ds for ds in found if needle in ds.dsid.upper()]
            if not matches:
                names = ", ".join(sorted({ds.dsid for ds in found}))
                raise LookupError(f"no dataset matching {dataset!r}; available: {names}")
            chosen = matches[0]
        scheme = chosen.to_scheme()
        return place_gammas(scheme) if place else scheme
