"""RIPL-3 masses segment: parsing, GroundState projection, provenance."""

from __future__ import annotations

import pytest

import nook
from nook.sources.ripl3 import clear_ripl_cache

from test_ripl3 import needs_mirror

# Verbatim rows from the mirrored files (formats (2i4,1x,a2,1x,i1,...)).
FRDM_ROWS = """\
#
#  Z   A s fl     Mexp      Err       Mth      Emic    beta2   beta3   beta4   beta6
  12  24 MG 2   -13.934     0.000   -14.080     3.540   0.374          -0.053  -0.010
   8  16 O  2    -4.737     0.000    -4.840     2.140   0.021          -0.108  -0.003
   3   3 Li 1    28.667     2.000
 118 310 Ex 0                        90.500     3.030   0.080           0.020   0.000
"""

HFB_ROWS = """\
#
#  Z   A s fl     Mexp      Err       Mth    beta2   beta4     rhon     rn       an       rhop     rp       ap
  12  24 Mg 2   -13.934     0.000   -13.600   0.570   0.170   0.0745   3.0549   0.4772   0.0716   3.0743   0.4952
  10  42 Ne 0                       121.770   0.170   0.030   0.1083   3.6212   0.7911   0.0535   3.3409   0.4653
"""

ABUNDANCE_ROWS = """\
#      Natural abundances
  12  24 Mg  78.990000  0.040000
"""


@pytest.fixture
def mass_dir(tmp_path):
    masses = tmp_path / "masses"
    masses.mkdir()
    (masses / "mass-frdm95.dat").write_text(FRDM_ROWS)
    (masses / "mass-hfb14.dat").write_text(HFB_ROWS)
    (masses / "abundance.dat").write_text(ABUNDANCE_ROWS)
    clear_ripl_cache()
    return tmp_path


def test_mass_entry_merges_all_three_tables(mass_dir):
    entry = nook.Ripl3Source(path=mass_dir).masses("24Mg")
    assert entry.mass_excess_exp.value == pytest.approx(-13934.0)
    assert entry.mass_excess_exp.symmetric == pytest.approx(0.0)
    assert entry.mass_excess_frdm95.value == pytest.approx(-14080.0)
    assert entry.mass_excess_hfb14.value == pytest.approx(-13600.0)
    assert entry.emic_frdm95 == pytest.approx(3540.0)
    assert entry.beta2_frdm95 == pytest.approx(0.374)
    assert entry.beta3_frdm95 is None  # blank column, not zero
    assert entry.beta2_hfb14 == pytest.approx(0.570)
    assert entry.neutron_density_params == pytest.approx((0.0745, 3.0549, 0.4772))
    assert entry.abundance.value == pytest.approx(78.99)


def test_theory_only_row_has_no_experimental_mass(mass_dir):
    entry = nook.Ripl3Source(path=mass_dir).masses((118, 310))
    assert entry.mass_excess_exp.value is None
    assert entry.mass_excess_frdm95.value == pytest.approx(90500.0)


def test_hfb_only_nuclide_loads(mass_dir):
    # 710 nuclides exist only in mass-hfb14.dat; the HFB row must supply the
    # (empty) experimental columns rather than KeyError-ing on "mexp".
    entry = nook.Ripl3Source(path=mass_dir).masses("42Ne")
    assert entry.mass_excess_exp.value is None
    assert entry.mass_excess_frdm95 is None
    assert entry.mass_excess_hfb14.value == pytest.approx(121770.0)
    assert entry.beta2_hfb14 == pytest.approx(0.170)


def test_recommended_flag_marks_systematics(mass_dir):
    entry = nook.Ripl3Source(path=mass_dir).masses("3Li")
    assert entry.recommended_only
    assert entry.to_ground_state().mass_from_systematics


def test_ground_state_projection(mass_dir):
    gs = nook.ground_state("24Mg", source="ripl3", path=mass_dir)
    assert gs.source == "ripl3-local"
    assert gs.mass_excess.value == pytest.approx(-13934.0)
    assert gs.abundance.value == pytest.approx(78.99)
    assert gs.metadata["beta2_frdm95"] == pytest.approx(0.374)


def test_provenance_and_units(mass_dir):
    entry = nook.Ripl3Source(path=mass_dir).masses("24Mg")
    assert "FRDM" in entry.provenance("mass_excess_frdm95")
    assert "Audi" in entry.provenance("mass_excess_exp")
    assert entry.units("mass_excess_exp") == "keV"
    assert entry.units("abundance") == "% (mole fraction)"


# --------------------------------------------------------------------------
# against the real mirror
# --------------------------------------------------------------------------


@needs_mirror
def test_real_mirror_hfb_only_nuclides_load():
    source = nook.Ripl3Source()
    for nuclide in ("42Ne", (100, 222)):
        entry = source.masses(nuclide)
        assert entry.mass_excess_hfb14 is not None
        assert entry.mass_excess_exp.value is None


@needs_mirror
def test_real_mirror_mass_and_density():
    source = nook.Ripl3Source()
    fe56 = source.masses("56Fe")
    assert fe56.mass_excess_exp.value == pytest.approx(-60601, abs=5)
    density = source.matter_density("208Pb")
    assert len(density.rows) == 200
    radii = [r for r, _, _ in density.rows]
    assert radii == sorted(radii)


@needs_mirror
def test_ripl_mass_consistent_with_ensdf_derived_q_values():
    """The AME vintages differ, so ask only for a few-keV agreement."""
    livechart_style = {  # AME2020 values, keV
        "24Mg": -13933.6,
        "56Fe": -60607.1,
        "208Pb": -21748.7,
    }
    source = nook.Ripl3Source()
    for name, expected in livechart_style.items():
        got = source.masses(name).mass_excess_exp.value
        assert got == pytest.approx(expected, abs=10.0), name


@needs_mirror
def test_mass_table_matches_single_lookups():
    source = nook.Ripl3Source()
    table = source.mass_table()
    assert len(table) > 5000
    assert table[(12, 24)] == source.masses("24Mg")
    # the HFB-only nuclides are present too, not just the FRDM ones
    assert (100, 222) in table
