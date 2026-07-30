"""RIPL-3 resonances, gamma and fission segments."""

from __future__ import annotations

import pytest

import nook
from nook.sources.ripl3.fission import _parse_empirical
from nook.sources.ripl3.gamma import parse_gdr_exp, parse_gdr_theor
from nook.sources.ripl3.resonances import parse_resonances

from test_ripl3 import needs_mirror

# Rows cut verbatim from the mirrored files.
RESONANCE_ROWS = """\
# Z El  A   Io     Bn        D0        dD      S0    dS    Gg   dG   Com.
 12 Mg  24  0.    7.331  4.80E+02  7.00E+01   0.00  0.00    -     -  *07I
 13 Al  27  2.5   7.725  5.50E+01  6.00E+00   2.30  0.50  1600  400  07I
"""

GDR_EXP_ROWS = """\
#-----------------------------------------------------------------------
   6  12 C   0 22.79   15.7   3.62                       13.0 - 25.0  63Bur
                0.07    0.7   0.28
  73 181 Ta  0 12.19  262.6   2.93 14.99  317.5   5.13   10.8 - 18.8  81Gur
                0.29   74.6   1.12  0.52   65.8   0.87
"""

GDR_THEOR_ROWS = """\
#--------------------------------------------------
  14  35 Si  1.000  21.27   7.76  21.27   7.76
  14  36 Si  1.216  18.63   5.41  22.29   9.55
"""

BARRIER_ROWS = """\
#--------------------------------------------------------------
  80 196 Hg    S    16.9
  92 238 U     GA    6.30    1.00   MA    5.50    0.60    0.818
"""


# --------------------------------------------------------------------------
# resonances
# --------------------------------------------------------------------------


def test_resonance_row_with_all_fields():
    entry = parse_resonances(RESONANCE_ROWS, wave=0)[(13, 27)]
    assert entry.spacing_kev.value == pytest.approx(55.0)
    assert entry.spacing_kev.symmetric == pytest.approx(6.0)
    assert entry.strength_1e4.value == pytest.approx(2.30)
    assert entry.gamma_width_mev_milli.value == pytest.approx(1600.0)
    assert entry.bn_kev == pytest.approx(7725.0)
    assert not entry.estimated
    assert entry.reference == "07I"


def test_resonance_dashes_mean_absent_and_star_means_estimated():
    entry = parse_resonances(RESONANCE_ROWS, wave=0)[(12, 24)]
    assert entry.gamma_width_mev_milli.value is None
    assert entry.estimated


# Verbatim from resonances1.dat: RIPL files Ar-40 with Z=20 (a transcription
# error), which would collide with -- and silently lose to -- the real Ca-40
# row two lines later.  The element symbol is authoritative.
P_WAVE_ROWS = """\
# Z El  A    Io    Bn        D1        dD      S1   dS    Gg    dG    Ref
 20 Ar  40   .0   6.100  1.60E+01  3.00E+00   0.44  0.08              07I
 20 Ca  40   .0   8.363  1.60E+01  1.00E+00   0.32  0.05   360    90  06M
"""


def test_resonance_z_column_yields_to_element_symbol():
    table = parse_resonances(P_WAVE_ROWS, wave=1)
    argon, calcium = table[(18, 40)], table[(20, 40)]
    assert argon.bn_kev == pytest.approx(6100.0)
    assert argon.spacing_kev.symmetric == pytest.approx(3.0)
    assert calcium.bn_kev == pytest.approx(8363.0)
    assert calcium.gamma_width_mev_milli.value == pytest.approx(360.0)


# --------------------------------------------------------------------------
# gamma
# --------------------------------------------------------------------------


def test_gdr_single_lorentzian_with_errors():
    entries = parse_gdr_exp(GDR_EXP_ROWS, "exp-SLO")[(6, 12)]
    (peak,) = entries[0].peaks
    energy, width, sigma = peak
    assert energy.value == pytest.approx(22.79)
    assert energy.symmetric == pytest.approx(0.07)
    assert width.value == pytest.approx(3.62)
    assert width.symmetric == pytest.approx(0.28)
    assert sigma.value == pytest.approx(15.7)
    assert entries[0].fit_range_mev == (13.0, 25.0)


def test_gdr_deformed_nucleus_has_two_lorentzians():
    entries = parse_gdr_exp(GDR_EXP_ROWS, "exp-SLO")[(73, 181)]
    assert len(entries[0].peaks) == 2
    second = entries[0].peaks[1]
    assert second[0].value == pytest.approx(14.99)
    assert second[0].symmetric == pytest.approx(0.52)


def test_gdr_theor_collapses_degenerate_peaks():
    table = parse_gdr_theor(GDR_THEOR_ROWS)
    assert len(table[(14, 35)].peaks) == 1  # spherical: E1 == E2
    assert len(table[(14, 36)].peaks) == 2
    assert table[(14, 36)].eta == pytest.approx(1.216)


# --------------------------------------------------------------------------
# fission
# --------------------------------------------------------------------------


def test_empirical_barriers_full_actinide_row():
    entry = _parse_empirical(BARRIER_ROWS)[(92, 238)]
    assert entry.inner.height_mev.value == pytest.approx(6.30)
    assert entry.inner.symmetry == "GA"
    assert entry.outer.height_mev.value == pytest.approx(5.50)
    assert entry.outer.hw_mev.value == pytest.approx(0.60)
    assert entry.outer.symmetry == "MA"
    assert entry.pairing_gap_mev == pytest.approx(0.818)


def test_empirical_barriers_ragged_light_row():
    entry = _parse_empirical(BARRIER_ROWS)[(80, 196)]
    assert entry.inner.height_mev.value == pytest.approx(16.9)
    assert entry.inner.hw_mev.value is None
    assert entry.outer is None


# --------------------------------------------------------------------------
# against the real mirror
# --------------------------------------------------------------------------


@needs_mirror
def test_real_mirror_p_wave_file_keeps_all_targets():
    from nook.sources.ripl3._util import read_file

    source = nook.Ripl3Source()
    table = parse_resonances(
        read_file(source.path / "resonances" / "resonances1.dat"), wave=1
    )
    assert len(table) == 119  # the readme's count; the Ar/Ca collision loses one
    argon = source.resonances("40Ar", wave=1)
    assert argon.bn_kev == pytest.approx(6100.0)


@needs_mirror
def test_real_mirror_resonances_gdr_fission_gsf():
    source = nook.Ripl3Source()

    aluminium = source.resonances("27Al")
    assert aluminium.spacing_kev.value == pytest.approx(55.0)

    tantalum = source.gdr("181Ta")
    assert any(len(fit.peaks) == 2 for fit in tantalum if fit.kind == "exp-SLO")
    assert any(fit.kind == "theor" for fit in tantalum)

    u238 = source.fission_barriers("238U")
    assert u238.inner.height_mev.value == pytest.approx(6.3, abs=0.5)
    hfb = source.fission_barriers("238U", model="hfb")
    assert hfb.inner.height_mev.value == pytest.approx(5.8, abs=0.1)

    gsf = source.gsf("21Mg")
    assert gsf.rows[0] == pytest.approx((0.1, 7.677e-06))
    energies = [e for e, _ in gsf.rows]
    assert energies == sorted(energies)


@needs_mirror
def test_gdr_exp_files_parse_completely():
    source = nook.Ripl3Source()
    from nook.sources.ripl3.gamma import parse_gdr_exp as parse
    from nook.sources.ripl3._util import read_file

    for shape in ("SLO", "MLO"):
        file = source.path / "gamma" / f"gdr-parameters&errors-exp-{shape}.dat"
        table = parse(read_file(file), f"exp-{shape}")
        assert len(table) > 50
        for entries in table.values():
            for entry in entries:
                for energy, width, sigma in entry.peaks:
                    assert energy.value and 5.0 < energy.value < 40.0
                    assert width.value and width.value > 0


@needs_mirror
def test_resonance_table_matches_single_lookups():
    source = nook.Ripl3Source()
    table = source.resonance_table()
    assert len(table) == 300  # every s-wave target
    assert table[(26, 56)] == source.resonances("56Fe")
    assert len(source.resonance_table(wave=1)) == 119
