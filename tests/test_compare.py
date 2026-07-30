"""Cross-source comparison.

The gated 24Mg test is the independent oracle for the RIPL levels parser:
RIPL's discrete levels derive from ENSDF, and the ENSDF flat-file path is a
completely separate parser, so low-lying levels agreeing to ~1 keV checks
both chains end to end.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import nook
from nook import compare
from nook.compare import _jpi_agree, _match_levels
from nook.model import Level
from nook.quantities import Uncertain, parse_spin_parity
from nook.sources.ripl3 import parse_levels_file

from test_ripl3 import FIXTURE, needs_mirror

DATA = Path(__file__).parent / "data"
CHARACTERIZATION = json.loads((DATA / "ripl3_characterization.json").read_text())

CHAIN_DIR = Path(os.environ.get("ENSDF_PATH", "/nonexistent"))
needs_chain = pytest.mark.skipif(
    not (CHAIN_DIR / "ensdf.024").is_file(),
    reason="set ENSDF_PATH to a directory holding ensdf.024",
)


@pytest.fixture(scope="module")
def mg24():
    blocks = {b.nuclide.a: b for b in parse_levels_file(FIXTURE)}
    return blocks[24].to_scheme()


def _shifted(levels, delta_kev):
    """Every level moved by ``delta_kev`` except the ground state (RIPL
    indexes from 1, so key on energy, not index)."""
    return tuple(
        replace(
            lv,
            energy=Uncertain(
                value=(lv.energy.value or 0.0)
                + (delta_kev if lv.energy.value else 0.0)
            ),
        )
        for lv in levels
    )


# --------------------------------------------------------------------------
# matching machinery, against the fixture scheme
# --------------------------------------------------------------------------


def test_identical_schemes_match_perfectly(mg24):
    pairs, only_a, only_b = _match_levels(mg24.levels, mg24.levels, tolerance_kev=2.0)
    assert len(pairs) == len(mg24.levels)
    assert not only_a and not only_b
    assert all(a.energy_kev == b.energy_kev for a, b in pairs)


def test_small_shift_still_matches(mg24):
    # low-lying levels only: high in the scheme, spacings shrink below the
    # shift and cross-pairing becomes the *correct* nearest-neighbour answer
    low = mg24.levels[:15]
    shifted = _shifted(low, 0.5)
    pairs, _, _ = _match_levels(low, shifted, tolerance_kev=2.0)
    assert len(pairs) == len(low)
    # pairing stays index-faithful, not just count-faithful
    assert all(a.index == b.index for a, b in pairs)


def test_large_shift_orphans_excited_levels(mg24):
    low = mg24.levels[:15]
    shifted = _shifted(low, 500.0)
    pairs, only_a, _ = _match_levels(low, shifted, tolerance_kev=2.0)
    # ground states always anchor to each other ...
    assert any(a.energy_kev == 0.0 and b.energy_kev == 0.0 for a, b in pairs)
    # ... and a 500 keV shift orphans everything else
    assert len(only_a) == len(low) - 1


def test_matching_is_one_to_one(mg24):
    doubled = mg24.levels + mg24.levels
    pairs, _, only_b = _match_levels(mg24.levels, doubled, tolerance_kev=2.0)
    assert len(pairs) == len(mg24.levels)
    assert len(only_b) == len(mg24.levels)


def test_identical_pairs_are_not_significant(mg24):
    pairs, _, _ = _match_levels(mg24.levels, mg24.levels, tolerance_kev=2.0)
    for a, b in pairs:
        m = compare.LevelMatch(
            a=a, b=b, delta_kev=0.0,
            combined_sigma_kev=None, jpi_agree=None,
        )
        assert not m.significant


def test_significance_thresholds():
    def match(delta, sigma):
        gs = Level(index=0, energy=Uncertain(0.0))
        other = Level(index=0, energy=Uncertain(delta))
        return compare.LevelMatch(
            a=gs, b=other, delta_kev=delta, combined_sigma_kev=sigma, jpi_agree=None
        )

    assert not match(0.5, None).significant  # no sigma: 1 keV rule
    assert match(1.5, None).significant
    assert not match(2.0, 1.0).significant  # 2 sigma is fine
    assert match(4.0, 1.0).significant  # 4 sigma is not


def test_jpi_agreement_uses_candidate_intersection():
    def level(jpi):
        return Level(index=0, energy=Uncertain(0.0), spin_parity=parse_spin_parity(jpi))

    assert _jpi_agree(level("2+"), level("2+")) is True
    assert _jpi_agree(level("2+"), level("3+")) is False
    assert _jpi_agree(level("(1,2)+"), level("2+")) is True
    assert _jpi_agree(level("2"), level("2-")) is True  # unknown parity matches
    assert _jpi_agree(level(""), level("2+")) is None  # unknown J judges nothing


# --------------------------------------------------------------------------
# the cross-source oracle: RIPL vs the independent ENSDF parser
# --------------------------------------------------------------------------


@needs_mirror
@needs_chain
def test_compare_24mg_ripl_vs_ensdf_adopted_matches_low_lying():
    result = compare.levels("24Mg", sources=("file", "ripl3"), below=8000.0)
    assert result.source_a == "ensdf-file"
    assert result.source_b == "ripl3-local"
    assert result.n_matched >= 8
    for m in result.matched:
        assert abs(m.delta_kev) <= 1.0, (m.a.energy_kev, m.b.energy_kev)
    assert result.jpi_agreement_fraction is not None
    assert result.jpi_agreement_fraction >= 0.9
    # RIPL's evaluated completeness cutoff rides along
    assert result.cutoff_b == CHARACTERIZATION["24Mg"]["nmax"]


@needs_mirror
@needs_chain
def test_compare_reports_levels_unique_to_each_source():
    result = compare.levels("24Mg", sources=("file", "ripl3"), below=12000.0)
    # ENSDF adopted carries levels RIPL dropped; both counts are small
    assert len(result.only_a) + len(result.only_b) < result.n_matched


# --------------------------------------------------------------------------
# masses
# --------------------------------------------------------------------------


@needs_mirror
def test_compare_masses_180ta():
    result = compare.masses("180Ta", with_livechart=False)
    assert set(result.entries) == {"ripl-exp", "frdm95", "hfb14"}
    assert result.entries["ripl-exp"].value == pytest.approx(-48936, abs=5)
    # theory sits within ~1 MeV of experiment for a well-measured nuclide
    for theory in ("frdm95", "hfb14"):
        residual = result.residual_kev(theory)
        assert residual is not None and abs(residual) < 1000
    assert result.spread_kev is not None and result.spread_kev < 2000


@needs_mirror
def test_mass_table_skips_missing_nuclides():
    table = compare.mass_table(["24Mg", "56Fe", "100C"], with_livechart=False)
    assert set(table) == {(12, 24), (26, 56)}
    for row in table.values():
        assert "ripl-exp" in row.entries


# --------------------------------------------------------------------------
# characterization: freeze the parse of 20 nuclides across the chart
# --------------------------------------------------------------------------


@needs_mirror
@pytest.mark.parametrize("name", sorted(CHARACTERIZATION))
def test_characterization_frozen(name):
    expected = CHARACTERIZATION[name]
    block = nook.Ripl3Source().levels(name)
    assert block.n_levels == expected["nol"]
    assert block.n_gammas == expected["nog"]
    assert block.n_complete == expected["nmax"]
    assert block.n_unique_jpi == expected["nc"]
    assert block.sn_kev == pytest.approx(expected["sn_kev"], abs=0.01)
    if expected["e1_kev"] is not None:
        assert block.levels[1].energy_kev == pytest.approx(expected["e1_kev"], abs=0.01)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@needs_mirror
def test_cli_compare_masses_json(capsys):
    from nook.cli import main

    assert main(["compare", "24Mg", "--what", "masses", "--no-livechart", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "ripl-exp" in payload


@needs_mirror
@needs_chain
def test_cli_compare_levels(capsys):
    from nook.cli import main

    assert main(["compare", "24Mg", "--below", "8000"]) == 0
    out = capsys.readouterr().out
    assert "matched" in out and "ripl3-local" in out
