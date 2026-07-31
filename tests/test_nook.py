import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

import nook as el
from nook.quantities import (
    HBAR_LN2_MEV_S,
    parse_half_life,
    parse_spin_parity,
    parse_value_with_uncertainty,
)
from nook.sources.ensdf_file import ENSDFFileSource, parse_ensdf_text

# --------------------------------------------------------------------------
# nuclide identifiers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["24Mg", "Mg24", "Mg-24", "mg_24", "24MG", " 24 mg "])
def test_nuclide_parsing_accepts_common_spellings(text):
    assert el.Nuclide.parse(text) == el.Nuclide(12, 24)


def test_nuclide_round_trips_through_nucid():
    nuc = el.Nuclide.parse("208Pb")
    assert nuc.nucid() == "208PB"
    assert el.Nuclide.from_nucid(nuc.nucid()) == nuc
    assert nuc.livechart_id() == "208pb"
    assert nuc.n == 126


def test_nuclide_rejects_nonsense():
    with pytest.raises(ValueError):
        el.Nuclide.parse("unobtainium-5")
    with pytest.raises(ValueError):
        el.Nuclide(12, 4)  # A < Z


# --------------------------------------------------------------------------
# uncertainties
# --------------------------------------------------------------------------


def test_uncertainty_follows_significant_digit_convention():
    q = parse_value_with_uncertainty("12.1", "11")
    assert q.value == pytest.approx(12.1)
    assert q.symmetric == pytest.approx(1.1)

    q = parse_value_with_uncertainty("12.1", "1")
    assert q.symmetric == pytest.approx(0.1)

    q = parse_value_with_uncertainty("1368.672", "5")
    assert q.symmetric == pytest.approx(0.005)


def test_uncertainty_handles_exponents_and_asymmetry():
    q = parse_value_with_uncertainty("1.2E+3", "3")
    assert q.value == pytest.approx(1200.0)
    assert q.symmetric == pytest.approx(300.0)

    q = parse_value_with_uncertainty("5.62", "+13-9")
    assert (q.plus, q.minus) == (pytest.approx(0.13), pytest.approx(0.09))
    assert q.symmetric == pytest.approx(0.11)


def test_uncertainty_records_limits():
    q = parse_value_with_uncertainty("LT 3.0", "")
    assert q.operator == "LT" and q.is_limit and q.value == pytest.approx(3.0)
    assert parse_value_with_uncertainty("", "").value is None


# --------------------------------------------------------------------------
# spin and parity
# --------------------------------------------------------------------------


def test_spin_parity_simple_assignment():
    sp = parse_spin_parity("3/2-")
    assert sp.unique == el.JPi(3, -1)
    assert sp.unique.j_float == 1.5
    assert str(sp.unique) == "3/2-"
    assert not sp.tentative and not sp.assumed


def test_spin_parity_marks_tentative_and_assumed():
    assert parse_spin_parity("(2+)").tentative
    assert parse_spin_parity("[0+]").assumed
    assert parse_spin_parity("(2+)").unique == el.JPi(4, 1)


def test_parity_does_not_leak_across_a_comma():
    """Manual V.20: in '1,2+' the '+' attaches only to the 2."""
    sp = parse_spin_parity("1,2+")
    assert sp.candidates == (el.JPi(2, None), el.JPi(4, +1))
    assert sp.unique is None


def test_parity_after_a_closing_paren_covers_the_group():
    """Manual V.20: '(3,4)-' is J=(3) or (4) with pi=-."""
    sp = parse_spin_parity("(3,4)-")
    assert sp.candidates == (el.JPi(6, -1), el.JPi(8, -1))
    assert sp.tentative


def test_not_excludes_rather_than_asserts():
    """'NOT 3-' says the level is not 3-; reporting it as 3- inverts the data."""
    sp = parse_spin_parity("NOT 3-")
    assert sp.candidates == ()
    assert sp.excluded == (el.JPi(6, -1),)
    assert not sp.allows(6, -1)
    assert sp.allows(4, +1)


@pytest.mark.parametrize(
    "text,expected",
    [
        # Manual V.20, the three range cases, verbatim.
        ("3 TO 6-", [(6, -1), (8, -1), (10, -1), (12, -1)]),
        ("3+ TO 6-", [(6, 1), (8, None), (10, None), (12, -1)]),
        ("3+ TO 6", [(6, 1), (8, None), (10, None), (12, None)]),
    ],
)
def test_range_parity_follows_the_manual(text, expected):
    sp = parse_spin_parity(text)
    assert [(c.two_j, c.parity) for c in sp.candidates] == expected


def test_operator_bounds_spin_but_fixes_parity():
    """Manual V.20: 'LE 5+' means pi=+ and J<=5."""
    sp = parse_spin_parity("LE 5+")
    assert sp.constraint == "LE"
    assert sp.candidates == (el.JPi(10, +1),)


def test_natural_parity_is_a_rule_not_a_value():
    assert parse_spin_parity("NATURAL").natural_parity is True
    assert parse_spin_parity("UNNATURAL").natural_parity is False
    assert parse_spin_parity("NATURAL").candidates == ()


def test_spin_parity_expands_ranges_and_keeps_limits():
    sp = parse_spin_parity("3/2 TO 9/2")
    assert [c.two_j for c in sp.candidates] == [3, 5, 7, 9]
    assert all(c.parity is None for c in sp.candidates)

    sp = parse_spin_parity("GE 4")
    assert sp.constraint == "GE"
    assert sp.candidates == (el.JPi(8, None),)


def test_spin_parity_unknown_is_not_known():
    sp = parse_spin_parity("")
    assert not sp.is_known and sp.unique is None


# --------------------------------------------------------------------------
# half-lives
# --------------------------------------------------------------------------


def test_half_life_time_units():
    hl = parse_half_life("1.33 PS", "7")
    assert hl.seconds.value == pytest.approx(1.33e-12)
    assert hl.seconds.symmetric == pytest.approx(0.07e-12)
    assert not hl.from_width


def test_half_life_stable():
    hl = parse_half_life("STABLE")
    assert hl.stable and math.isinf(hl.seconds.value)


def test_half_life_from_level_width():
    hl = parse_half_life("65 KEV", "5")
    assert hl.from_width
    assert hl.width_mev.value == pytest.approx(0.065)
    assert hl.seconds.value == pytest.approx(HBAR_LN2_MEV_S / 0.065)
    # a larger width means a shorter life: the error branches swap over
    assert hl.seconds.plus == pytest.approx(hl.seconds.value * 0.005 / 0.065)


# --------------------------------------------------------------------------
# flat-file reader
# --------------------------------------------------------------------------

ENSDF_FIXTURE = """\
 24MG    ADOPTED LEVELS, GAMMAS        2014FI09
 24MG  XA24MG(P,P')
 24MG  XB26MG(P,T)
 24MG c  Comment record, must be ignored.
 24MG  L 0.0         0+                STABLE
 24MG2 L XREF=AB$MOMM1=-0.5 2
 24MG cL E$from a least-squares fit.
 24MG  L 1368.672  5 2+                1.33 PS   7
 24MG  G 1368.67   5 100       E2
 24MG  L 4122.889  124+                39 FS     5
 24MG  G 2754.02   11100       E2
 24MG  L 4238.36   5 2+                1.7 FS    3
 24MG  L 7616.47   5 (3-)              18 FS     4
 24MG  L 9299.6    5 (1,2)+            65 KEV    5
 24MG  L X+1440      3/2 TO 9/2

 24MG    24MG(P,T)                     1978AJ03
 24MG  L 0.0         0+
 24MG  L 1368      3 2+
"""


def test_flat_file_splits_datasets():
    datasets = parse_ensdf_text(ENSDF_FIXTURE)
    assert [ds.dsid for ds in datasets] == [
        "ADOPTED LEVELS, GAMMAS",
        "24MG(P,T)",
    ]
    assert datasets[0].is_adopted
    assert datasets[0].reference == "2014FI09"
    assert all(ds.nuclide == el.Nuclide(12, 24) for ds in datasets)


def test_flat_file_decodes_level_columns():
    adopted = parse_ensdf_text(ENSDF_FIXTURE)[0]
    assert len(adopted.levels) == 7
    assert adopted.skipped == 1  # the cL comment record

    gs, first, second = adopted.levels[0], adopted.levels[1], adopted.levels[2]
    assert gs.energy_kev == 0.0 and gs.half_life.stable
    assert gs.spin_parity.unique == el.JPi(0, +1)

    assert first.energy_kev == pytest.approx(1368.672)
    assert first.energy.symmetric == pytest.approx(0.005)
    assert first.half_life.seconds.value == pytest.approx(1.33e-12)

    assert second.energy_kev == pytest.approx(4122.889)
    assert second.spin_parity.unique == el.JPi(8, +1)


def test_flat_file_handles_unplaced_and_unbound_levels():
    adopted = parse_ensdf_text(ENSDF_FIXTURE)[0]
    floating = adopted.levels[-1]
    assert floating.is_floating and floating.energy_offset == "X"
    assert floating.energy_kev == pytest.approx(1440.0)

    resonance = adopted.levels[-2]
    assert resonance.half_life.from_width
    assert resonance.spin_parity.tentative


def test_gamma_placement_by_energy_difference():
    scheme = el.place_gammas(parse_ensdf_text(ENSDF_FIXTURE)[0].to_scheme())
    by_energy = {round(g.energy_kev): g for g in scheme.gammas}
    assert by_energy[1369].start_index == 1 and by_energy[1369].end_index == 0
    assert by_energy[2754].start_index == 2 and by_energy[2754].end_index == 1
    assert by_energy[2754].multipolarity == "E2"


def test_file_source_selects_dataset(tmp_path):
    (tmp_path / "ensdf.024").write_text(ENSDF_FIXTURE)
    source = ENSDFFileSource(tmp_path)

    adopted = source.fetch(el.Nuclide.parse("24Mg"))
    assert adopted.dataset == "ADOPTED LEVELS, GAMMAS"

    transfer = source.fetch(el.Nuclide.parse("24Mg"), dataset="(P,T)")
    assert len(transfer.levels) == 2

    with pytest.raises(LookupError):
        source.fetch(el.Nuclide.parse("24Mg"), dataset="COULOMB EXCITATION")


# --------------------------------------------------------------------------
# scheme queries
# --------------------------------------------------------------------------


@pytest.fixture
def mg24():
    return el.place_gammas(parse_ensdf_text(ENSDF_FIXTURE)[0].to_scheme())


def test_below_drops_floating_levels(mg24):
    sub = mg24.below(5000.0)
    assert [round(lv.energy_kev) for lv in sub] == [0, 1369, 4123, 4238]
    assert all(not lv.is_floating for lv in sub)


def test_filtering_prunes_dangling_gammas(mg24):
    sub = mg24.below(2000.0)
    assert len(sub.levels) == 2
    assert len(sub.gammas) == 1  # the 2754 keV line lost its parent


def test_complete_up_to_truncates_at_first_ambiguity(mg24):
    # stops at the ambiguous (1,2)+ level
    assert len(mg24.complete_up_to().levels) == 5
    # ... or earlier, if tentative assignments are not trusted
    assert len(mg24.complete_up_to(allow_tentative=False).levels) == 4


def test_with_known_jpi_can_exclude_tentative(mg24):
    assert len(mg24.with_known_jpi(allow_tentative=True).levels) == 5
    assert len(mg24.with_known_jpi(allow_tentative=False).levels) == 4


def test_records_export_is_flat_and_typed(mg24):
    rows = mg24.to_records()
    assert rows[1]["two_j"] == 4 and rows[1]["parity"] == 1
    assert rows[0]["stable"] is True
    assert set(rows[0]) >= {"index", "energy_kev", "jpi", "half_life_s"}


# --------------------------------------------------------------------------
# Livechart backend (no network: the transport is injected)
# --------------------------------------------------------------------------

LEVELS_CSV = """\
idx,energy,unc_e,jp,half_life,operator_hl,unc_hl,unit_hl,half_life_sec,unc_hls,\
decay_1,decay_1_%,unc_1,energy_shift,questionable,ENSDF_publication_cut-off
0,0,0,0+,STABLE,,,,,,,,,,,31-Dec-2013
1,1368.672,0.005,2+,1.33,,0.07,ps,1.33E-12,7E-14,,,,,,31-Dec-2013
2,4122.889,0.012,4+,39,,5,fs,3.9E-14,5E-15,,,,,,31-Dec-2013
"""

GAMMAS_CSV = """\
energy,unc_e,intensity,unc_i,multipolarity,start_level_idx,end_level_idx
1368.672,0.005,100,,E2,1,0
2754.007,0.011,100,,E2,2,1
"""


@pytest.fixture
def livechart(tmp_path):
    def opener(url, timeout):
        return GAMMAS_CSV if "gammas" in url else LEVELS_CSV

    return el.LivechartSource(cache=el.Cache(tmp_path), opener=opener)


def test_livechart_builds_a_scheme(livechart):
    scheme = livechart.fetch(el.Nuclide.parse("24Mg"))
    assert len(scheme.levels) == 3 and len(scheme.gammas) == 2
    assert scheme.source == "iaea-livechart"
    assert scheme.levels[1].energy_kev == pytest.approx(1368.672)
    # Livechart gives absolute uncertainties, not ENSDF digit counts
    assert scheme.levels[1].energy.symmetric == pytest.approx(0.005)
    assert scheme.levels[1].spin_parity.unique == el.JPi(4, +1)
    assert scheme.levels[0].half_life.stable
    assert scheme.metadata["ensdf_cutoff"] == "31-Dec-2013"


def test_livechart_keeps_raw_rows(livechart):
    scheme = livechart.fetch(el.Nuclide.parse("24Mg"))
    assert scheme.levels[2].raw["unit_hl"] == "fs"


def test_livechart_caches_responses(tmp_path):
    calls = []

    def opener(url, timeout):
        calls.append(url)
        return LEVELS_CSV

    cache = el.Cache(tmp_path)
    for _ in range(3):
        el.LivechartSource(cache=cache, opener=opener).fetch(
            el.Nuclide.parse("24Mg"), with_gammas=False
        )
    assert len(calls) == 1


def test_livechart_reports_service_error_codes(tmp_path):
    source = el.LivechartSource(
        cache=el.Cache(tmp_path, enabled=False), opener=lambda url, t: "0"
    )
    with pytest.raises(el.LivechartError, match="no data"):
        source.levels(el.Nuclide.parse("24Mg"))


def test_top_level_helper_rejects_dataset_for_api():
    with pytest.raises(ValueError, match="adopted"):
        el.level_scheme("24Mg", source="livechart", dataset="(P,T)")


# --------------------------------------------------------------------------
# band structure (columns 77 and 78-79, plus BAND() documentation records)
# --------------------------------------------------------------------------

BAND_FIXTURE = """\
180TA    ADOPTED LEVELS, GAMMAS                                  15NDS    201506
180TA CL BAND(A)$K|p=1+ band. Configuration=|p7/2[404]|n9/2[624]
180TA2CL continued here.
180TA CL BAND(Q)$K|p=9- band. Configuration=|p9/2[514]|n9/2[624]
180TA  L 0.0         1+                8.154 H   6                          A
180TA  L 39.54     5 2+                                                     A
180TA  L 77.2      129-                7.1E15 Y  GT                         QM1
180TA  L 280.12    1010-                                                    Q
"""


@pytest.fixture
def ta180():
    return parse_ensdf_text(BAND_FIXTURE)[0].to_scheme()


def test_band_flag_is_read_from_column_77(ta180):
    assert [lv.band for lv in ta180.levels] == ["A", "A", "Q", "Q"]


def test_metastable_field_spans_columns_78_79(ta180):
    isomer = ta180.levels[2]
    assert isomer.metastable  # "M1" -- would be missed by reading col 78 alone
    assert isomer.raw["MS"] == "M1"
    assert isomer.half_life.seconds.operator == "GT"
    assert isomer.half_life.seconds.value == pytest.approx(7.1e15 * 3.15576e7)


def test_band_definitions_decode_ensdf_markup(ta180):
    definition = ta180.band_definition("A")
    assert definition.startswith("K\u03c0=1+ band")
    assert "\u03c07/2[404]\u03bd9/2[624]" in definition
    assert definition.endswith("continued here.")  # continuation record merged


def test_bands_group_and_order_by_bandhead(ta180):
    bands = ta180.bands()
    assert list(bands) == ["A", "Q"]
    assert [lv.energy_kev for lv in bands["Q"]] == [77.2, 280.12]
    assert len(bands["A"]) == 2


def test_gallagher_moszkowski_partners_share_a_configuration(ta180):
    """K = Omega_p +/- Omega_n from one two-quasiparticle configuration."""
    a = ta180.band_definition("A").split("Configuration=")[1]
    q = ta180.band_definition("Q").split("Configuration=")[1]
    assert a != q  # different configurations here, but both must parse cleanly
    assert ta180.band("A").levels[0].spin_parity.unique == el.JPi(2, +1)
    assert ta180.band("Q").levels[0].spin_parity.unique == el.JPi(18, -1)


# --------------------------------------------------------------------------
# continuation records, XREF provenance, transfer data
# --------------------------------------------------------------------------

from nook.sources.ensdf_file import (  # noqa: E402
    parse_continuation_properties,
    parse_transfer_l,
    parse_xref,
)


def test_continuation_properties_split_on_dollar():
    line = " 24MG2 L XREF=AB$MOMM1=-0.5 2$%IT=100"
    props = parse_continuation_properties([line])
    assert props == {"XREF": "AB", "MOMM1": "-0.5 2", "%IT": "100"}


def test_xref_expands_shorthands():
    key = ("A", "B", "C", "D")
    assert parse_xref("ABD", key) == ("A", "B", "D")
    assert parse_xref("AB(C)D", key) == ("A", "B", "C", "D")  # uncertain, still seen
    assert parse_xref("ALL", key) == key
    assert parse_xref("ALL EXCEPT BC", key) == ("A", "D")


def test_transfer_l_handles_mixed_assignments():
    assert parse_transfer_l("4") == (4,)
    assert parse_transfer_l("5(+6)") == (5, 6)
    assert parse_transfer_l("") == ()


def test_x_records_resolve_xref_to_dataset_names():
    adopted = parse_ensdf_text(ENSDF_FIXTURE)[0]
    assert adopted.xref_key == {"A": "24MG(P,P')", "B": "26MG(P,T)"}
    gs = adopted.levels[0]
    assert gs.xref == ("A", "B")
    assert gs.observed_in == ("24MG(P,P')", "26MG(P,T)")
    assert gs.properties["MOMM1"] == "-0.5 2"


def test_seen_in_filters_by_provenance():
    scheme = parse_ensdf_text(ENSDF_FIXTURE)[0].to_scheme()
    assert len(scheme.seen_in("(P,T)").levels) == 1
    assert len(scheme.seen_in("COULOMB").levels) == 0


def test_explicit_final_level_beats_energy_matching():
    """An FL= continuation states the final level; don't guess from energies."""
    fixture = ENSDF_FIXTURE.replace(
        " 24MG  G 2754.02   11100       E2\n",
        " 24MG  G 2754.02   11100       E2\n 24MG2 G FL=1368.672\n",
    )
    scheme = el.place_gammas(parse_ensdf_text(fixture)[0].to_scheme())
    g = next(x for x in scheme.gammas if round(x.energy_kev) == 2754)
    assert g.final_level_energy == pytest.approx(1368.672)
    assert g.end_index == 1


# --------------------------------------------------------------------------
# decay datasets
# --------------------------------------------------------------------------

DECAY_FIXTURE = """\
180HF    180LU B- DECAY
180LU  P 0.0         5+                5.7 M     1              3100      70
180HF  N 0.43      1           1.00      1.0
180HF  L 0.0         0+                STABLE
180HF  L 308.541   214+
180HF  G 215.241   1551.3    25E2
180HF  B             5       2           7.42    18
"""


@pytest.fixture
def hf180_decay():
    dataset = parse_ensdf_text(DECAY_FIXTURE)[0]
    return dataset.to_decay_scheme()


def test_dataset_is_recognised_as_a_decay(hf180_decay):
    assert hf180_decay.dsid == "180LU B- DECAY"
    assert hf180_decay.nuclide == el.Nuclide.parse("180Hf")


def test_parent_record_describes_the_decaying_state(hf180_decay):
    parent = hf180_decay.parent
    assert parent.nuclide == el.Nuclide.parse("180Lu")  # not the dataset nuclide
    assert parent.spin_parity.unique == el.JPi(10, +1)
    assert parent.half_life.seconds.value == pytest.approx(5.7 * 60)
    assert parent.q_value.value == pytest.approx(3100)
    assert parent.q_value.symmetric == pytest.approx(70)


def test_normalization_record_fields(hf180_decay):
    norm = hf180_decay.normalization
    assert norm.photon.value == pytest.approx(0.43)
    assert norm.photon.symmetric == pytest.approx(0.01)
    assert norm.branching.value == pytest.approx(1.0)
    assert norm.feeding.value == pytest.approx(1.0)
    assert norm.transition is None  # blank field stays absent, not zero


def test_feeding_attaches_to_the_preceding_level(hf180_decay):
    feeding = hf180_decay.feedings[0]
    assert feeding.kind == "B" and feeding.label == "\u03b2-"
    assert feeding.level_index == 1  # the 308.541 keV level, not the ground state
    assert feeding.intensity.value == pytest.approx(5)
    assert feeding.log_ft.value == pytest.approx(7.42)
    assert feeding.log_ft.symmetric == pytest.approx(0.18)
    assert hf180_decay.feeding_of(1) == (feeding,)
    assert hf180_decay.feeding_of(0) == ()


def test_relative_intensities_become_absolute(hf180_decay):
    gamma = hf180_decay.levels.gammas[0]
    assert gamma.intensity.value == pytest.approx(51.3)
    absolute = hf180_decay.absolute_photon_intensity(gamma)
    assert absolute.value == pytest.approx(51.3 * 0.43)  # RI * NR * BR
    # relative uncertainties add in quadrature
    assert absolute.symmetric == pytest.approx(
        absolute.value * ((2.5 / 51.3) ** 2 + (0.01 / 0.43) ** 2) ** 0.5
    )


def test_missing_normalization_reports_unknown_not_unity():
    """Silently assuming NR=1 would fabricate absolute intensities."""
    stripped = "\n".join(
        ln for ln in DECAY_FIXTURE.splitlines() if not ln.startswith("180HF  N")
    )
    scheme = parse_ensdf_text(stripped + "\n")[0].to_decay_scheme()
    assert scheme.normalization is None
    _, absolute = scheme.absolute_feedings()[0]
    assert absolute.value is None


def test_references_map_keynumbers_to_citations():
    """The 'REFERENCES' identification record also has an R in column 8."""
    text = "180    REFERENCES\n180    R 2012WA38 JOUR CPCHC 36 1603\n"
    assert el.parse_references(text) == {"2012WA38": "JOUR CPCHC 36 1603"}


# --------------------------------------------------------------------------
# error propagation
# --------------------------------------------------------------------------

from nook.quantities import add, divide, multiply  # noqa: E402


def test_product_keeps_the_two_error_branches_apart():
    """Symmetrising an asymmetric input throws away what the evaluator said."""
    a = el.Uncertain(5.62, 0.13, 0.09)
    z = multiply(a, el.Uncertain(2.0, 0.0, 0.0))
    assert z.value == pytest.approx(11.24)
    assert z.plus == pytest.approx(0.26)
    assert z.minus == pytest.approx(0.18)
    assert z.plus != z.minus


def test_division_swaps_the_denominator_branches():
    """A denominator fluctuating up pushes the quotient down."""
    z = divide(el.Uncertain(10.0, 0.0, 0.0), el.Uncertain(2.0, 0.4, 0.2))
    assert z.value == pytest.approx(5.0)
    assert z.plus == pytest.approx(5.0 * 0.1)  # from the -0.2 branch
    assert z.minus == pytest.approx(5.0 * 0.2)  # from the +0.4 branch


@pytest.mark.parametrize(
    "a_op,b_op,expected",
    [
        ("LE", "LE", "LE"),
        ("LT", "LE", "LT"),  # strictness is contagious
        ("GE", "GE", "GE"),
        ("LE", None, "LE"),
        (None, None, None),
    ],
)
def test_limit_directions_combine_under_multiplication(a_op, b_op, expected):
    z = multiply(el.Uncertain(3.0, operator=a_op), el.Uncertain(2.0, operator=b_op))
    assert z.operator == expected
    assert z.value == pytest.approx(6.0)


def test_opposing_limits_give_no_bound():
    """An upper limit times a lower limit constrains nothing."""
    z = multiply(el.Uncertain(3.0, operator="LE"), el.Uncertain(2.0, operator="GE"))
    assert z.value is None and "unbounded" in z.raw
    # dividing by an upper limit also inverts to a lower bound
    assert (
        divide(el.Uncertain(3.0, operator="LE"), el.Uncertain(2.0, operator="LE")).value
        is None
    )


def test_quality_flags_degrade_to_the_weakest_claim():
    z = multiply(el.Uncertain(3.0, operator="AP"), el.Uncertain(2.0, operator="SY"))
    assert z.operator == "SY"  # from systematics is the softer statement


def test_sum_adds_branches_in_quadrature():
    z = add(el.Uncertain(1.0, 0.3, 0.1), el.Uncertain(2.0, 0.4, 0.2))
    assert z.value == pytest.approx(3.0)
    assert z.plus == pytest.approx(0.5)
    assert z.minus == pytest.approx((0.1**2 + 0.2**2) ** 0.5)


def test_totals_apply_a_shared_factor_once(hf180_decay):
    """NR is common to every gamma, so it must not enter the quadrature sum."""
    naive = add(
        *(hf180_decay.absolute_photon_intensity(g) for g in hf180_decay.levels.gammas)
    )
    correct = hf180_decay.total_photon_intensity()
    assert correct.value == pytest.approx(naive.value)
    # one gamma only in the fixture, so the two agree; the relative error
    # must still carry NR's full 1/43 contribution
    assert correct.symmetric / correct.value == pytest.approx(
        ((2.5 / 51.3) ** 2 + (0.01 / 0.43) ** 2) ** 0.5
    )


def test_delayed_particles_normalise_through_np_not_nb():
    """NP is defined against total parent decays; NB would be the wrong scale."""
    fixture = (
        "180PT    181HG ECP DECAY\n"
        "181HG  P 0.0         1/2-              3.6 S     1              6485\n"
        "180PT  N                        1.3E-4 3                1.3E-4 3\n"
        "180PT  L 0.0         0+\n"
        "180PT  DP            50      10\n"
    )
    scheme = parse_ensdf_text(fixture)[0].to_decay_scheme()
    norm = scheme.normalization
    assert norm.delayed.value == pytest.approx(1.3e-4)
    assert norm.feeding is None  # NB is blank here
    feeding, absolute = scheme.absolute_feedings()[0]
    assert feeding.kind == "D" and feeding.particle == "P"
    assert feeding.label == "delayed proton"
    assert absolute.value == pytest.approx(50 * 1.3e-4)  # NP alone, no extra BR


def test_ufloat_bridge_is_lossy_and_says_so():
    pytest.importorskip("uncertainties")
    u = el.Uncertain(5.62, 0.13, 0.09).to_ufloat()
    assert u.nominal_value == pytest.approx(5.62)
    assert u.std_dev == pytest.approx(0.11)  # symmetrised
    with pytest.raises(ValueError, match="bound"):
        el.Uncertain(3.2, operator="LT").to_ufloat()


# --------------------------------------------------------------------------
# ground-state properties (masses, abundance, radii, moments)
# --------------------------------------------------------------------------

# Header and first row exactly as the Livechart ground_states endpoint
# returns them. The neutron is a useful test case: negative radius, blank
# abundance and quadrupole, and a systematics flag.
GROUND_STATES_CSV = (
    "z,n,symbol,radius,unc_r,abundance,unc_a,energy_shift,energy,unc_e,ripl_shift,"
    "jp,half_life,operator_hl,unc_hl,unit_hl,half_life_sec,unc_hls,"
    "decay_1,decay_1_%,unc_1,decay_2,decay_2_%,unc_2,decay_3,decay_3_%,unc_3,"
    "isospin,magnetic_dipole,unc_md,electric_quadrupole,unc_eq,"
    "qbm,unc_qb,qbm_n,unc_qbmn,qa,unc_qa,qec,unc_qec,sn,unc_sn,sp,unc_sp,"
    "binding,unc_ba,atomic_mass,unc_am,massexcess,unc_me,me_systematics,discovery,"
    "ENSDFpublicationcut-off,ENSDFauthors,Extraction_date\n"
    "0,1,n,-0.1149,0.0027,,,,0,,,1/2+,613.9,,6,s,613.9,0.6,B-,100,,,,,,,,,"
    "-1.9130427,0.0000005,,,782.347,0.0004,,,,,,,0.0,0.0,,,0,0.0,"
    "1008664.9159,0.00047,8071.31806,0.00044,N,,31-Oct-2005,BALRAJ SINGH,2023-10-18\n"
)


@pytest.fixture
def neutron(tmp_path):
    source = el.LivechartSource(
        cache=el.Cache(tmp_path), opener=lambda url, t: GROUND_STATES_CSV
    )
    return source.ground_state(el.Nuclide(0, 1))


def test_ground_state_reads_the_mass_evaluation(neutron):
    assert neutron.mass_excess.value == pytest.approx(8071.31806)
    assert neutron.mass_excess.symmetric == pytest.approx(0.00044)
    assert neutron.atomic_mass.value == pytest.approx(1008664.9159)  # micro-u
    assert neutron.q_beta_minus.value == pytest.approx(782.347)
    assert neutron.mass_from_systematics is False


def test_ground_state_reads_moments_and_radius(neutron):
    assert neutron.magnetic_dipole.value == pytest.approx(-1.9130427)
    assert neutron.magnetic_dipole.symmetric == pytest.approx(5e-7)
    assert neutron.charge_radius.value == pytest.approx(-0.1149)


def test_ground_state_reads_ensdf_side(neutron):
    assert neutron.spin_parity.unique == el.JPi(1, +1)
    assert neutron.half_life.seconds.value == pytest.approx(613.9)
    assert neutron.decay_modes[0][0] == "B-"


def test_absent_quantities_stay_unknown_not_zero(neutron):
    """A blank abundance must not read as an abundance of zero."""
    assert neutron.abundance.value is None
    assert neutron.electric_quadrupole.value is None
    assert neutron.discovery_year is None


def test_ground_states_metadata_column_names_differ_from_levels(neutron):
    """levels says ENSDF_authors; ground_states says ENSDFauthors."""
    assert neutron.metadata["ensdf_authors"] == "BALRAJ SINGH"
    assert neutron.metadata["ensdf_cutoff"] == "31-Oct-2005"


def test_provenance_and_units_are_recorded(neutron):
    assert "AME" in neutron.provenance("mass_excess")
    assert neutron.provenance("abundance") == "NUBASE"
    assert "radii" in neutron.provenance("charge_radius")
    assert neutron.units("atomic_mass") == "micro-u"  # not u
    assert neutron.units("electric_quadrupole") == "barn"


def test_total_binding_energy_scales_by_mass_number():
    gs = el.GroundState(
        nuclide=el.Nuclide.parse("180Ta"),
        binding_per_nucleon=el.Uncertain(8000.0, 1.0, 1.0),
    )
    assert gs.total_binding_energy.value == pytest.approx(180 * 8000.0)


def test_moments_are_typed_on_levels_from_flat_files():
    """MOMM1/MOME2 continuations become Uncertain, not leftover strings."""
    fixture = BAND_FIXTURE.replace(
        "180TA  L 77.2      129-                7.1E15 Y  GT"
        "                         QM1\n",
        "180TA  L 77.2      129-                7.1E15 Y  GT"
        "                         QM1\n"
        "180TA2 L MOME2=+4.95 2$MOMM1=4.825 11\n",
    )
    isomer = parse_ensdf_text(fixture)[0].levels[2]
    assert isomer.magnetic_dipole.value == pytest.approx(4.825)
    assert isomer.magnetic_dipole.symmetric == pytest.approx(0.011)
    assert isomer.electric_quadrupole.value == pytest.approx(4.95)
    assert isomer.electric_quadrupole.symmetric == pytest.approx(0.02)


# --------------------------------------------------------------------------
# regressions found in review
# --------------------------------------------------------------------------


def test_identity_filter_preserves_every_gamma(mg24):
    """Requiring the *final* level to survive discarded unplaced transitions."""
    unfiltered = mg24.filter(lambda lv: True)
    assert len(unfiltered.gammas) == len(mg24.gammas)


def test_a_gamma_survives_with_its_parent_not_its_daughter(mg24):
    """Level indices are stable, so a kept gamma may point outside the subset."""
    kept = mg24.filter(lambda lv: lv.index in (0, 2))
    transition = next(g for g in kept.gammas if g.start_index == 2)
    assert transition.end_index == 1
    assert kept.level_by_index(1) is None  # resolves to None, never fabricated


def test_level_lookup_handles_missing_and_none():
    scheme = parse_ensdf_text(ENSDF_FIXTURE)[0].to_scheme()
    assert scheme.level_by_index(0) is scheme.levels[0]
    assert scheme.level_by_index(9999) is None
    assert scheme.level_by_index(None) is None


def test_duplicate_csv_columns_stay_distinguishable(tmp_path):
    """The levels endpoint emits unc_hl twice; DictReader would keep only one."""
    header = (
        "z,n,symbol,idx,energy,unc_e,jp,half_life,operator_hl,unc_hl,unit_hl,"
        "half_life_sec,unc_hl,decay_1"
    )
    row = "73,107,Ta,0,0,0,1+,8.154,,7,h,29354,25,B-"
    source = el.LivechartSource(
        cache=el.Cache(tmp_path), opener=lambda u, t: f"{header}\n{row}\n"
    )
    level = source.fetch(el.Nuclide.parse("180Ta"), with_gammas=False).levels[0]
    assert level.raw["unc_hl"] == "7"  # native units
    assert level.raw["unc_hl.1"] == "25"  # seconds
    assert level.half_life.seconds.symmetric == pytest.approx(25)


def test_ragged_csv_rows_do_not_abort_the_parse(tmp_path):
    source = el.LivechartSource(
        cache=el.Cache(tmp_path),
        opener=lambda u, t: "idx,energy,unc_e,jp\n0,0,0,1+\n1,100\n",
    )
    assert len(source.fetch(el.Nuclide.parse("180Ta"), with_gammas=False).levels) == 2


def test_strict_mode_refuses_to_treat_a_blank_uncertainty_as_zero():
    exact = el.Uncertain(10.0)  # no uncertainty quoted
    measured = el.Uncertain(2.0, 0.2, 0.2)
    assert not exact.uncertainty_known
    assert multiply(exact, measured).symmetric == pytest.approx(2.0)
    assert multiply(exact, measured, strict=True).symmetric is None
    assert add(exact, measured, strict=True).value == pytest.approx(12.0)


def test_chain_cache_is_invalidated_when_the_file_changes(tmp_path):
    from nook.sources.ensdf_file import ENSDFFileSource, clear_chain_cache

    clear_chain_cache()
    path = tmp_path / "ensdf.024"
    path.write_text(ENSDF_FIXTURE)
    source = ENSDFFileSource(tmp_path)
    assert len(source.fetch(el.Nuclide.parse("24Mg")).levels) == 7

    path.write_text(" 24MG    ADOPTED LEVELS\n 24MG  L 0.0         0+\n")
    assert len(source.fetch(el.Nuclide.parse("24Mg")).levels) == 1


def test_cli_runs_end_to_end(tmp_path, capsys):
    from nook.cli import main

    (tmp_path / "ensdf.024").write_text(ENSDF_FIXTURE)
    code = main(
        ["24Mg", "--source", "file", "--path", str(tmp_path), "--below", "5000"]
    )
    assert code == 0
    assert "1368.67" in capsys.readouterr().out

    code = main(
        ["24Mg", "--source", "file", "--path", str(tmp_path), "--dataset", "NOPE"]
    )
    assert code == 1  # a missing dataset is an error, not a traceback


def test_a_band_may_be_used_without_being_defined(ta180):
    """Evaluators do not always document every flag they use."""
    undocumented = (
        "180TA  L 500.0     105+                                                     Z"
        + "\n"
    )
    scheme = parse_ensdf_text(BAND_FIXTURE + undocumented)[0].to_scheme()
    assert "Z" in scheme.bands()
    assert scheme.band_definition("Z") is None
    assert scheme.band_definition("A") is not None


def test_energy_offset_may_be_a_suffix_not_only_a_prefix():
    """Manual V.18 allows NUM+A as well as A+NUM.

    Found by the frozen oracle: '4172.3+X' was being read as no energy at
    all, silently discarding a level's position.
    """
    from nook.sources.ensdf_file import split_energy_offset

    assert split_energy_offset("4172.3+X") == ("4172.3", "X")
    assert split_energy_offset("X+1440") == ("1440", "X")
    assert split_energy_offset("X") == ("", "X")
    assert split_energy_offset("SN+100") == ("100", "SN")
    assert split_energy_offset("1234.5") == ("1234.5", None)


def test_suffix_offset_level_keeps_its_energy():
    card = "180TA  L 4172.3+X    (23,24,25)"
    scheme = parse_ensdf_text(f"180TA    ADOPTED LEVELS\n{card}\n")[0].to_scheme()
    level = scheme.levels[0]
    assert level.energy_kev == pytest.approx(4172.3)
    assert level.energy_offset == "X"
    assert level.is_floating


# --------------------------------------------------------------------------
# physics invariants
# --------------------------------------------------------------------------

# A Nilsson label pi/nu Omega/2[N n_z Lambda]; the orbital parity is (-1)**N.
_NILSSON = re.compile(r"([\u03c0\u03bd])(\d+)/2\[(\d)(\d)(\d)\]")
_BANDHEAD = re.compile(r"K\u03c0=\(?(\d+)\)?([+-])")


def _two_quasiparticle_bands(definitions):
    """Group decoded ``BAND()`` texts by configuration, keeping 2-qp bands."""
    grouped = {}
    for label, text in definitions.items():
        head = _BANDHEAD.search(text)
        if not head or "Configuration=" not in text:
            continue
        config = text.split("Configuration=")[1].split()[0]
        orbitals = [
            (Fraction(int(m.group(2)), 2), (-1) ** int(m.group(3)))
            for m in _NILSSON.finditer(config)
        ]
        if len(orbitals) != 2:
            continue  # 4-quasiparticle bands have no two-body coupling rule
        grouped.setdefault(config, []).append(
            (label, int(head.group(1)), 1 if head.group(2) == "+" else -1, orbitals)
        )
    return grouped


def test_band_assignments_obey_the_coupling_rules():
    """Cross-check band parsing against nuclear structure, not another parser.

    Two quasiparticles couple to K = |Omega_p - Omega_n| or Omega_p + Omega_n
    (the Gallagher-Moszkowski doublet), with parity the product of the two
    orbital parities. Column 77, the ``BAND()`` records and the markup
    rendering are all decoded separately, so if any of them were misread the
    bandheads would stop satisfying these relations.

    This is the only validation available for the band code: the differential
    oracle cannot help, because nudel does not parse bands at all.
    """
    text = (Path(__file__).parent / "data" / "ensdf180_excerpt.ensdf").read_text()
    definitions = parse_ensdf_text(text)[0].band_definitions
    grouped = _two_quasiparticle_bands(definitions)

    doublets = 0
    for config, bands in grouped.items():
        for label, k, parity, ((omega_p, pi_p), (omega_n, pi_n)) in bands:
            allowed = {abs(omega_p - omega_n), omega_p + omega_n}
            assert k in allowed, f"band {label} ({config}): K={k} not in {allowed}"
            assert parity == pi_p * pi_n, f"band {label} ({config}): wrong parity"
        if len(bands) == 2:
            doublets += 1
            assert {b[1] for b in bands} == {
                abs(bands[0][3][0][0] - bands[0][3][1][0]),
                bands[0][3][0][0] + bands[0][3][1][0],
            }, f"{config}: the two bands should be the K-minus/K-plus pair"

    assert doublets >= 4, f"only {doublets} doublets found; the check is vacuous"


def test_band_parsing_failure_would_break_the_invariant():
    """The invariant must actually be sensitive to a mis-decode."""
    corrupted = {"X": "K\u03c0=5+ band. Configuration=\u03c07/2[404]\u03bd9/2[624]"}
    grouped = _two_quasiparticle_bands(corrupted)
    label, k, parity, ((wp, _), (wn, _)) = grouped["\u03c07/2[404]\u03bd9/2[624]"][0]
    assert k not in {abs(wp - wn), wp + wn}  # 5 is neither 1 nor 8
