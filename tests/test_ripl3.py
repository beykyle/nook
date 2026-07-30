"""RIPL-3 backend: discrete levels.

The excerpt fixture is 19Mg-21Mg and 24Mg cut verbatim from the mirrored
``z012.dat`` (2021 cut).  Mirror-wide sweeps are gated on the mirror being
present, the same way the ENSDF chain tests gate on ``ENSDF_PATH``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nook
from nook.sources.ripl3 import parse_levels_file, parse_levels_param

FIXTURE = (Path(__file__).parent / "data" / "ripl3" / "z012_excerpt.dat").read_text()

RIPL_DIR = nook.default_ripl_path()  # honours $RIPL_PATH itself
HAS_MIRROR = (RIPL_DIR / "levels" / "z012.dat").is_file()
needs_mirror = pytest.mark.skipif(
    not HAS_MIRROR,
    reason="populate the mirror with tools/fetch_ripl3.py or set RIPL_PATH",
)


@pytest.fixture(scope="module")
def blocks():
    return {b.nuclide.a: b for b in parse_levels_file(FIXTURE)}


# --------------------------------------------------------------------------
# identification record and bookkeeping
# --------------------------------------------------------------------------


def test_header_counts_match_body(blocks):
    for block in blocks.values():
        assert len(block.levels) == block.n_levels
        assert len(block.gammas) == block.n_gammas


def test_separation_energies_are_kev(blocks):
    assert blocks[24].sn_kev == pytest.approx(16531.222)
    assert blocks[24].sp_kev == pytest.approx(11692.696)


# --------------------------------------------------------------------------
# level records
# --------------------------------------------------------------------------


def test_level_energy_is_converted_to_kev(blocks):
    mg24 = blocks[24]
    assert mg24.levels[1].energy_kev == pytest.approx(1368.672)
    assert "MeV" in mg24.levels[1].energy.raw  # file units preserved


def test_stable_ground_state_from_negative_half_life(blocks):
    gs = blocks[24].levels[0]
    assert gs.half_life.stable and gs.half_life.seconds.value == float("inf")


def test_half_life_in_seconds(blocks):
    assert blocks[24].levels[1].half_life.seconds.value == pytest.approx(1.33e-12)


def test_unknown_spin_gives_empty_candidates(blocks):
    # 21Mg levels 17+ have s = -1.0 and no ENSDF J field
    unknown = blocks[21].levels[16]
    assert unknown.spin_parity.candidates == ()
    assert not unknown.spin_parity.is_known


def test_unique_spin_becomes_jpi(blocks):
    lv = blocks[24].levels[2]
    assert lv.jpi == nook.JPi(8, +1)  # 4+


def test_ripl_estimated_spin_is_flagged_assumed(blocks):
    # 21Mg level 6 is flagged 'c': chosen from (1/2,3/2)- by RIPL, not ENSDF
    lv = blocks[21].levels[5]
    assert lv.properties["ripl_spin_flag"] == "c"
    assert lv.spin_parity.assumed
    assert lv.spin_parity.raw == "(1/2,3/2)-"


def test_u_flag_is_ensdf_own_candidate_not_assumed(blocks):
    # Every firm unique ENSDF assignment below Emax is u-flagged (24Mg's 0+
    # ground state included), so 'u' must NOT be marked assumed...
    gs = blocks[24].levels[0]
    assert gs.properties["ripl_spin_flag"] == "u"
    assert not gs.spin_parity.assumed and not gs.spin_parity.tentative
    # ...but a u-flag over a tentative single candidate keeps `tentative`:
    # RIPL promoted (7/2+) to a firm 7/2+ and the parentheses must survive.
    lv = blocks[21].levels[8]
    assert lv.spin_parity.raw == "(7/2+)"
    assert lv.spin_parity.tentative and not lv.spin_parity.assumed


def test_tentative_ensdf_assignment_survives(blocks):
    # 20Mg level 3: ENSDF says (2+,4+); RIPL picked 2+
    lv = blocks[20].levels[2]
    assert lv.spin_parity.raw == "(2+,4+)"


def test_decay_modes_parsed(blocks):
    gs = blocks[20].levels[0]
    modes = dict(gs.decay_modes)
    assert modes["%EC+%B+"].value == pytest.approx(100.0)
    assert modes["%B+P"].value == pytest.approx(30.3)


# --------------------------------------------------------------------------
# gamma records
# --------------------------------------------------------------------------


def test_gamma_end_index_set_directly(blocks):
    mg24 = blocks[24]
    first = mg24.gammas[0]
    assert (first.start_index, first.end_index) == (2, 1)
    assert first.energy_kev == pytest.approx(1368.6)


def test_gamma_probabilities_and_icc(blocks):
    first = blocks[24].gammas[0]
    assert first.intensity.value == pytest.approx(1.0)
    assert first.conversion_coefficient.value == pytest.approx(1.3e-05)
    assert float(first.properties["pe"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# LevelScheme integration
# --------------------------------------------------------------------------


def test_scheme_metadata_and_source(blocks):
    scheme = blocks[24].to_scheme()
    assert scheme.source == "ripl3-local"
    assert scheme.dataset == "RIPL-3 LEVELS"
    assert scheme.metadata["nmax"] == 28
    assert scheme.metadata["nc"] == 21
    assert scheme.metadata["cut"] == "2021"


def test_scheme_query_api_works(blocks):
    scheme = blocks[24].to_scheme()
    low = scheme.below(5000.0)
    assert [round(lv.energy_kev) for lv in low] == [0, 1369, 4123, 4238]


def test_levels_param_table():
    text = (
        "#Constant Temperature fit results\n"
        "#  Z   A El ...\n"
        "  12  24 Mg   2.00439   0.05683   0.79051   0.24825 186  28  10  21"
        "  7.74538  6.62000    1.30 CTF  Um   0             0.000\n"
    )
    param = parse_levels_param(text)[(12, 24)]
    assert param.temperature_mev == pytest.approx(2.00439)
    assert param.n_max == 28 and param.n_c == 21
    assert param.u_max_mev == pytest.approx(7.74538)


# --------------------------------------------------------------------------
# against the real mirror
# --------------------------------------------------------------------------


@needs_mirror
def test_fetch_via_top_level_helper():
    scheme = nook.level_scheme("24Mg", source="ripl3")
    assert scheme.source == "ripl3-local"
    assert scheme.levels[1].energy_kev == pytest.approx(1368.672)


@needs_mirror
def test_levels_param_via_facade():
    table = nook.Ripl3Source().levels_param()
    assert table[(12, 24)].n_max is not None


@needs_mirror
def test_sn_table_matches_single_lookup():
    source = nook.Ripl3Source()
    table = source.sn_table(zs=(12,))
    assert table[(12, 24)] == source.levels("24Mg").sn_kev
    assert all(z == 12 for z, _a in table)


@needs_mirror
def test_up_to_nmax_truncates():
    source = nook.Ripl3Source()
    full = source.fetch("24Mg")
    cut = source.fetch("24Mg", up_to_nmax=True)
    assert len(cut.levels) == full.metadata["nmax"] == 28
    kept = {lv.index for lv in cut.levels}
    assert all(g.start_index in kept and g.end_index in kept for g in cut.gammas)


@needs_mirror
def test_g_flagged_spins_are_assumed():
    # 'g' (spin inferred from gamma transitions) is in no readme but sits on
    # 5000+ levels of the 2021 cut; it is RIPL's inference, hence assumed.
    source = nook.Ripl3Source()
    flagged = [
        lv
        for block in (source.levels("181Ta"), source.levels("180Ta"))
        for lv in block.levels
        if lv.properties.get("ripl_spin_flag") == "g"
    ]
    assert flagged, "expected g-flagged levels in the tantalum chain"
    assert all(lv.spin_parity.assumed for lv in flagged)


@needs_mirror
def test_uncertain_energy_flag_makes_level_floating():
    # 180Ta band heads with unknown absolute energy carry the `unc` flag
    # (X, Y, ...); they must behave like ENSDF floating levels.
    scheme = nook.level_scheme("180Ta", source="ripl3")
    floating = [lv for lv in scheme if lv.is_floating]
    assert floating
    assert all(lv.energy_offset for lv in floating)
    assert all(lv.properties.get("ripl_energy_flag") for lv in floating)
    # below() must not treat the placeholder energy as an absolute position
    assert not [lv for lv in scheme.below(1e9) if lv.is_floating]


@needs_mirror
def test_cli_json_is_strict_json(capsys):
    from nook.cli import main

    assert main(["levels", "24Mg", "--source", "ripl3", "--json"]) == 0
    out = capsys.readouterr().out
    records = json.loads(out, parse_constant=lambda t: pytest.fail(f"bare {t}"))
    gs = records[0]
    assert gs["half_life_s"] is None  # inf is not JSON
    assert gs["stable"] is True


@needs_mirror
def test_ripl3_rejects_dataset_argument():
    with pytest.raises(ValueError, match="one levels dataset"):
        nook.level_scheme("24Mg", source="ripl3", dataset="ADOPTED")


@needs_mirror
@pytest.mark.parametrize("z", range(1, 119))
def test_every_element_file_parses_with_invariants(z):
    path = RIPL_DIR / "levels" / f"z{z:03d}.dat"
    if not path.is_file():
        pytest.skip(f"no z{z:03d}.dat in this cut")
    blocks = parse_levels_file(path.read_text(encoding="utf-8", errors="replace"))
    assert blocks
    for block in blocks:
        assert len(block.levels) == block.n_levels
        assert len(block.gammas) == block.n_gammas
        assert 0 <= block.n_complete <= block.n_levels
        energies = [lv.energy_kev for lv in block.levels]
        assert energies == sorted(energies)
        for gamma in block.gammas:
            assert gamma.end_index < gamma.start_index
        # per-level electromagnetic branching: sum(Pg) <= 1 + rounding
        by_level: dict[int, float] = {}
        for gamma in block.gammas:
            if gamma.intensity and gamma.intensity.value:
                by_level[gamma.start_index] = (
                    by_level.get(gamma.start_index, 0.0) + gamma.intensity.value
                )
        for total in by_level.values():
            assert total <= 1.02
