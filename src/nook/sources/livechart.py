"""IAEA Livechart backend.

Serves adopted levels, gammas and ground-state properties as CSV. The easiest
route in, at the cost of seeing only the adopted dataset -- for reaction
datasets use :mod:`nook.sources.ensdf_file`.

Columns are resolved through an alias table rather than by position: the API
has grown columns over time and the two endpoints spell shared ones
differently. Unrecognised columns survive on ``.raw``.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Sequence

from ..cache import Cache
from ..groundstate import GroundState
from ..model import Gamma, Level, LevelScheme
from ..nuclide import Nuclide
from ..quantities import (
    HalfLife,
    SpinParity,
    Uncertain,
    parse_spin_parity,
    parse_value_with_uncertainty,
)

__all__ = ["LivechartError", "LivechartSource"]

#: The metadata columns are spelled differently on different endpoints:
#: ``levels`` uses ``ENSDF_authors``, ``ground_states`` uses ``ENSDFauthors``.
_CUTOFF_KEYS = ("ENSDF_publication_cut-off", "ENSDFpublicationcut-off")
_AUTHOR_KEYS = ("ENSDF_authors", "ENSDFauthors")

BASE_URL = "https://nds.iaea.org/relnsd/v1/data"

# The service answers 403 to requests without a browser-ish User-Agent.
# This is documented behaviour in the IAEA API guide, not a workaround we
# invented; see https://www-nds.iaea.org/relnsd/vcharthtml/api_v0_guide.html
USER_AGENT = (
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0"
)

#: Numeric bodies the API returns instead of a CSV when a query yields nothing.
_ERROR_CODES = {
    "0": "no data found for this query",
    "1": "invalid or missing 'fields' parameter",
    "2": "invalid nuclide",
    "3": "invalid parameter combination",
}


class LivechartError(RuntimeError):
    """Raised when the Livechart service refuses or cannot answer a query."""


def _first(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] is not None:
            return row[name].strip()
    return default


class LivechartSource:
    """Pull level schemes from the IAEA Livechart data API."""

    name = "iaea-livechart"

    def __init__(
        self,
        base_url: str = BASE_URL,
        cache: Cache | None = None,
        timeout: float = 30.0,
        opener: Any = None,
    ):
        self.base_url = base_url
        self.cache = cache if cache is not None else Cache()
        self.timeout = timeout
        # Injectable for testing: any callable(url, timeout) -> str.
        self._opener = opener or self._fetch_url

    # -- transport ----------------------------------------------------------

    def _fetch_url(self, url: str, timeout: float) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise LivechartError(
                f"Livechart returned HTTP {exc.code} for {url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LivechartError(
                f"cannot reach Livechart at {url}: {exc.reason}"
            ) from exc

    def _query(self, **params: str) -> str:
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        body = self._opener(url, self.timeout)
        self.cache.put(url, body)
        return body

    @staticmethod
    def _rows(body: str) -> list[dict[str, str]]:
        """Parse the CSV body, keeping duplicated column names distinguishable.

        The ``levels`` endpoint emits ``unc_hl`` twice -- once for the
        half-life in its native unit, once for the value in seconds.
        ``csv.DictReader`` would keep only the last, silently losing the
        first.  Repeats are suffixed ``.1``, ``.2`` in header order instead.
        """
        text = body.strip()
        if text in _ERROR_CODES:
            raise LivechartError(_ERROR_CODES[text])
        if not text:
            return []
        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration:
            return []
        names: list[str] = []
        counts: dict[str, int] = {}
        for raw_name in header:
            name = (raw_name or "").strip()
            counts[name] = counts.get(name, 0) + 1
            names.append(name if counts[name] == 1 else f"{name}.{counts[name] - 1}")
        return [
            # not strict: a short or over-long row should not abort the parse
            {
                name: (value or "").strip()
                for name, value in zip(names, row, strict=False)
            }
            for row in reader
            if row
        ]

    # -- public API ---------------------------------------------------------

    def levels(self, nuclide: Nuclide) -> list[dict[str, str]]:
        return self._rows(self._query(fields="levels", nuclides=nuclide.livechart_id()))

    def gammas(self, nuclide: Nuclide) -> list[dict[str, str]]:
        try:
            return self._rows(
                self._query(fields="gammas", nuclides=nuclide.livechart_id())
            )
        except LivechartError:
            return []  # many nuclides simply have no tabulated transitions

    def ground_state(self, nuclide: Nuclide) -> GroundState:
        """Masses, abundance, radius and moments for one nuclide.

        These come from the compilations Livechart aggregates alongside
        ENSDF -- the atomic mass evaluation, NUBASE, the charge-radius
        tables and the moment tables -- not from ENSDF itself.  See
        :data:`~nook.groundstate.EVALUATIONS`.
        """
        rows = self._rows(
            self._query(fields="ground_states", nuclides=nuclide.livechart_id())
        )
        if not rows:
            raise LivechartError(f"no ground-state record for {nuclide}")
        return _ground_state_from_row(rows[0], nuclide)

    def fetch(self, nuclide: Nuclide, with_gammas: bool = True) -> LevelScheme:
        level_rows = self.levels(nuclide)
        gamma_rows = self.gammas(nuclide) if with_gammas else []
        levels = tuple(
            _level_from_row(row, fallback_index=i) for i, row in enumerate(level_rows)
        )
        gammas = tuple(_gamma_from_row(row) for row in gamma_rows)
        metadata = {}
        if level_rows:
            metadata = {
                "ensdf_cutoff": _first(level_rows[0], *_CUTOFF_KEYS),
                "ensdf_authors": _first(level_rows[0], *_AUTHOR_KEYS),
                "extraction_date": _first(
                    level_rows[0], "Extraction_date", "extraction_date"
                ),
            }
        return LevelScheme(
            nuclide=nuclide,
            levels=levels,
            gammas=gammas,
            source=self.name,
            dataset="ADOPTED LEVELS",
            metadata=metadata,
        )


# --------------------------------------------------------------------------
# row -> model
# --------------------------------------------------------------------------


def _uncertain(
    row: dict[str, str], value_keys: Sequence[str], unc_keys: Sequence[str]
) -> Uncertain:
    """Build an :class:`Uncertain` from already-separated CSV columns.

    Unlike the flat files, Livechart gives absolute uncertainties, so the
    significant-digit convention does not apply here.
    """
    value_text = _first(row, *value_keys)
    unc_text = _first(row, *unc_keys)
    try:
        value = float(value_text) if value_text else None
    except ValueError:
        return parse_value_with_uncertainty(value_text, unc_text)
    try:
        sigma = float(unc_text) if unc_text else None
    except ValueError:
        sigma = None
    return Uncertain(value, sigma, sigma, raw=f"{value_text} {unc_text}".strip())


def _decay_modes(row: dict[str, str]) -> tuple[tuple[str, Uncertain], ...]:
    modes = []
    for i in (1, 2, 3):
        name = _first(row, f"decay_{i}")
        if not name:
            continue
        modes.append(
            (
                name,
                _uncertain(row, [f"decay_{i}_%"], [f"unc_{i}"]),
            )
        )
    return tuple(modes)


def _level_from_row(row: dict[str, str], fallback_index: int) -> Level:
    idx_text = _first(row, "idx", "level_idx")
    index = int(float(idx_text)) if idx_text else fallback_index

    energy = _uncertain(row, ["energy"], ["unc_e", "unc", "energy_unc"])
    offset = _first(row, "energy_shift") or None

    half_life = _half_life_from_row(row)
    spin_parity: SpinParity = parse_spin_parity(_first(row, "jp"))

    return Level(
        index=index,
        energy=energy,
        spin_parity=spin_parity,
        half_life=half_life,
        energy_offset=offset,
        metastable=bool(_first(row, "metastable")),
        questionable=_first(row, "questionable").upper() in ("Y", "TRUE", "1"),
        decay_modes=_decay_modes(row),
        dataset="ADOPTED LEVELS",
        magnetic_dipole=_uncertain(
            row, ["magnetic_dipole"], ["unc_mn", "unc_md", "unc_mm"]
        ),
        electric_quadrupole=_uncertain(row, ["electric_quadrupole"], ["unc_eq"]),
        raw=dict(row),
    )


def _half_life_from_row(row: dict[str, str]) -> HalfLife:
    raw = " ".join(
        part
        for part in (
            _first(row, "half_life"),
            _first(row, "unit_hl"),
        )
        if part
    ).strip()
    if raw.upper().startswith("STABLE"):
        return HalfLife(seconds=Uncertain(float("inf"), raw=raw), stable=True, raw=raw)

    seconds = _uncertain(row, ["half_life_sec"], ["unc_hls", "unc_hl.1"])
    operator = _first(row, "operator_hl") or None
    if operator:
        seconds = Uncertain(
            seconds.value, seconds.plus, seconds.minus, operator.upper(), seconds.raw
        )
    unit = _first(row, "unit_hl").upper()
    from_width = unit in ("EV", "KEV", "MEV")
    width = None
    if from_width:
        scale = {"EV": 1e-6, "KEV": 1e-3, "MEV": 1.0}[unit]
        # the first unc_hl belongs to the native-unit value
        base = _uncertain(row, ["half_life"], ["unc_hl"])
        width = Uncertain(
            None if base.value is None else base.value * scale,
            None if base.plus is None else base.plus * scale,
            None if base.minus is None else base.minus * scale,
            raw=raw,
        )
    return HalfLife(seconds=seconds, from_width=from_width, width_mev=width, raw=raw)


def _ground_state_from_row(row: dict[str, str], nuclide: Nuclide) -> GroundState:
    def year() -> int | None:
        text = _first(row, "discovery")
        try:
            return int(float(text)) if text else None
        except ValueError:
            return None

    return GroundState(
        nuclide=nuclide,
        spin_parity=parse_spin_parity(_first(row, "jp")),
        half_life=_half_life_from_row(row),
        mass_excess=_uncertain(row, ["massexcess", "mass_excess"], ["unc_me"]),
        atomic_mass=_uncertain(row, ["atomic_mass"], ["unc_am"]),
        binding_per_nucleon=_uncertain(row, ["binding"], ["unc_ba"]),
        mass_from_systematics=_first(row, "me_systematics").upper().startswith("Y"),
        q_beta_minus=_uncertain(row, ["qbm"], ["unc_qb"]),
        q_beta_minus_n=_uncertain(row, ["qbm_n"], ["unc_qbmn"]),
        q_alpha=_uncertain(row, ["qa"], ["unc_qa"]),
        q_ec=_uncertain(row, ["qec"], ["unc_qec"]),
        s_n=_uncertain(row, ["sn"], ["unc_sn"]),
        s_p=_uncertain(row, ["sp"], ["unc_sp"]),
        abundance=_uncertain(row, ["abundance"], ["unc_a"]),
        discovery_year=year(),
        charge_radius=_uncertain(row, ["radius"], ["unc_r"]),
        # the moment uncertainty column is spelled unc_md on ground_states
        # and unc_mn on levels; accept either rather than picking one.
        magnetic_dipole=_uncertain(
            row, ["magnetic_dipole"], ["unc_md", "unc_mn", "unc_mm"]
        ),
        electric_quadrupole=_uncertain(row, ["electric_quadrupole"], ["unc_eq"]),
        isospin=_first(row, "isospin") or None,
        decay_modes=_decay_modes(row),
        source="iaea-livechart",
        metadata={
            "ensdf_cutoff": _first(row, *_CUTOFF_KEYS),
            "ensdf_authors": _first(row, *_AUTHOR_KEYS),
            "extraction_date": _first(row, "Extraction_date", "extraction_date"),
        },
        raw=dict(row),
    )


def _gamma_from_row(row: dict[str, str]) -> Gamma:
    def maybe_int(*keys: str) -> int | None:
        text = _first(row, *keys)
        try:
            return int(float(text)) if text else None
        except ValueError:
            return None

    return Gamma(
        energy=_uncertain(row, ["energy"], ["unc_e", "unc"]),
        start_index=maybe_int("start_level_idx"),
        end_index=maybe_int("end_level_idx"),
        intensity=_uncertain(row, ["intensity"], ["unc_i"]),
        multipolarity=_first(row, "multipolarity") or None,
        mixing_ratio=_uncertain(row, ["mixing_ratio"], ["unc_mr"]),
        conversion_coefficient=_uncertain(row, ["conversion_coeff"], ["unc_cc"]),
        raw=dict(row),
    )
