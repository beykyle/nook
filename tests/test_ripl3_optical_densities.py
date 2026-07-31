"""RIPL-3 optical potentials and level densities."""

from __future__ import annotations

import pytest

import nook
from nook.sources.ripl3.optical import _fortran_float, _parse_entry
from test_ripl3 import needs_mirror

# The first archive entry, verbatim (P.G.Young & E.D.Arthur, n + 237Np).
OMP_ENTRY = """\
    1
P.G.Young and E.D.Arthur
Proc.Int.Conf.Nucl.Data Sci.and Tech.,Julich (1992) p894
modification of potential by Haouat,Lachkar,Lagrange et al, NSE 81, (1982),
especially for higher energy


      .001   30.0000
   93   93
  237  237
    1    0    1    0    0
    1
    30.000
     1.26000   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
               .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
      .63000   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
               .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
    46.20000 -3.00000-1   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
               .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
               .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
               .00000+0   .00000+0   .00000+0   .00000+0   .00000+0   .00000+0
    0
    0
    0
    0
    0
    0
"""


def test_fortran_float_spellings():
    assert _fortran_float(".00000+0") == 0.0
    assert _fortran_float("-3.00000-1") == pytest.approx(-0.3)
    assert _fortran_float("1.00000-1") == pytest.approx(0.1)
    assert _fortran_float("46.20000") == pytest.approx(46.2)


def test_omp_entry_parses_header_and_ranges():
    entry = _parse_entry(OMP_ENTRY)
    assert entry.ripl_id == 1
    assert entry.projectile == "n"
    assert entry.author.startswith("P.G.Young")
    assert entry.e_range_mev == (0.001, 30.0)
    assert entry.z_range == (93, 93) and entry.a_range == (237, 237)
    assert entry.model == 1  # rigid rotor


def test_omp_component_coefficients():
    entry = _parse_entry(OMP_ENTRY)
    real_volume = entry.components[0]
    assert not real_volume.as_volume_integral
    (rng,) = real_volume.ranges
    assert rng.e_max_mev == pytest.approx(30.0)
    assert rng.radius_coefficients[0] == pytest.approx(1.26)
    assert rng.diffuseness_coefficients[0] == pytest.approx(0.63)
    assert rng.strength_coefficients[0] == pytest.approx(46.2)
    assert rng.strength_coefficients[1] == pytest.approx(-0.3)
    assert len(rng.radius_coefficients) == 13
    assert len(rng.strength_coefficients) == 25
    # remaining five components empty in this entry
    assert all(not c.ranges for c in entry.components[1:])


def test_omp_applicability_filter():
    entry = _parse_entry(OMP_ENTRY)
    assert entry.applies_to(nook.Nuclide(93, 237), energy_mev=14.0)
    assert not entry.applies_to(nook.Nuclide(93, 237), energy_mev=40.0)
    assert not entry.applies_to(nook.Nuclide(26, 56))
    # the energy filter must bite even without a nuclide filter
    assert entry.applies_to(energy_mev=14.0)
    assert not entry.applies_to(energy_mev=40.0)


# A>=100 loses the space after "A=" in the header; MeV-1 densities verbatim
# from the z026.tab layout.
HFB_TAB = """\
                    *********************************************************************************
                    *  Z= 82 A=170: Positive-Parity Spin-dependent Level Density [MeV-1] for PB170  *
                    *********************************************************************************
 U[MeV]  T[MeV]  NCUMUL   RHOOBS   RHOTOT     J=00     J=01     J=02
   0.25  0.256  1.00E+00 1.46E+00 3.31E+00 1.00E+00 0.00E+00 4.63E-01
                    *********************************************************************************
    *  Z= 82 A=170: Negative-parity Spin-dependent Level Density [MeV-1] for PB170  *
 U[MeV]  T[MeV]  NCUMUL   RHOOBS   RHOTOT     J=00     J=01     J=02
   0.25  0.300  1.00E+00 2.00E-01 1.00E+00 2.00E-01 0.00E+00 0.00E+00
"""


@pytest.fixture
def hfb_dir(tmp_path):
    from nook.sources.ripl3 import clear_ripl_cache

    tables = tmp_path / "densities" / "level-densities-hfb"
    tables.mkdir(parents=True)
    (tables / "z082.tab").write_text(HFB_TAB)
    clear_ripl_cache()
    return tmp_path


def test_hfb_header_without_space_after_a_parses(hfb_dir):
    table = nook.Ripl3Source(path=hfb_dir).hfb_density((82, 170))
    assert sorted(table.grids) == [-1, 1]


def test_rho_total_is_sum_of_spin_columns_not_rhotot(hfb_dir):
    table = nook.Ripl3Source(path=hfb_dir).hfb_density((82, 170))
    # per parity: RHOOBS, not the 2J+1-weighted RHOTOT
    assert table.rho(0.25, parity=1) == pytest.approx(1.46)
    assert table.rho(0.25, parity=-1) == pytest.approx(0.2)
    assert table.rho(0.25) == pytest.approx(1.66)
    # and it equals the sum over the spin columns
    assert table.rho(0.25, parity=1) == pytest.approx(
        sum(table.rho(0.25, two_j=two_j, parity=1) for two_j in (0, 2, 4)),
        rel=0.01,
    )


def test_rho_rejects_impossible_spin_for_even_a(hfb_dir):
    table = nook.Ripl3Source(path=hfb_dir).hfb_density((82, 170))
    with pytest.raises(ValueError, match="impossible"):
        table.rho(0.25, two_j=1)  # half-integer J on an even-A nucleus


def test_commented_out_parameter_row_is_skipped(tmp_path):
    from nook.sources.ripl3 import clear_ripl_cache

    densities = tmp_path / "densities"
    densities.mkdir()
    (densities / "level-densities-egsm.dat").write_text(
        "#26  57 Fe  0.5  7.646  2.54E-02  5.0E-03  1.4  0.1  6.0  0.2\n"
        " 27  58 Co  2.0  8.573  1.30E-03  2.0E-04  1.2  0.1  6.5  0.2\n"
    )
    clear_ripl_cache()
    source = nook.Ripl3Source(path=tmp_path)
    assert source.level_density_params("58Co").spacing_kev.value == pytest.approx(
        1.3e-3
    )
    with pytest.raises(LookupError):
        source.level_density_params("57Fe")  # the commented row must not crash


# --------------------------------------------------------------------------
# against the real mirror
# --------------------------------------------------------------------------


@needs_mirror
def test_full_archive_parses():
    potentials = nook.Ripl3Source().optical_potentials()
    assert len(potentials) == 566
    for entry in potentials:
        assert entry.e_range_mev[0] <= entry.e_range_mev[1]
        assert entry.z_range[0] <= entry.z_range[1]


@needs_mirror
def test_energy_filter_applies_without_nuclide():
    source = nook.Ripl3Source()
    # no archive entry claims validity at 2 GeV
    assert source.optical_potentials(projectile="n", energy_mev=2000.0) == ()
    at_14 = source.optical_potentials(projectile="n", energy_mev=14.0)
    assert at_14
    assert all(e.e_range_mev[0] <= 14.0 <= e.e_range_mev[1] for e in at_14)


@needs_mirror
def test_koning_delaroche_global_neutron_potential():
    # RIPL id 2405 is the Koning-Delaroche global spherical neutron OMP
    (kd,) = nook.Ripl3Source().optical_potentials(ripl_id=2405)
    assert "Koning" in kd.author
    assert kd.projectile == "n"
    assert kd.z_range[0] <= 26 <= kd.z_range[1]
    assert kd.model == 0


@needs_mirror
def test_deformations_for_carbon():
    rows = nook.Ripl3Source().deformations("12C")
    assert any(
        row.energy_kev == pytest.approx(4438.9, abs=1.0) and row.multipole == 2
        for row in rows
    )


@needs_mirror
def test_level_density_units_agree_across_models():
    """EGSM ships D0 in keV, the others in eV; normalised they must agree."""
    source = nook.Ripl3Source()
    values = {
        model: source.level_density_params("57Fe", model=model).spacing_kev.value
        for model in ("egsm", "bfm", "ctm", "hfm")
    }
    reference = values["egsm"]
    for model, value in values.items():
        assert value == pytest.approx(reference, rel=0.01), model


@needs_mirror
def test_hfb_density_reads_heavy_nuclides():
    # regression: "A=170" (no space) headers broke every file with A >= 100
    table = nook.Ripl3Source().hfb_density("208Pb")
    assert sorted(table.grids) == [-1, 1]
    assert table.rho(5.0) > 0


@needs_mirror
def test_hfb_density_monotone_and_d0_consistent():
    source = nook.Ripl3Source()
    table = source.hfb_density("57Fe")
    assert sorted(table.grids) == [-1, 1]
    for parity in (1, -1):
        totals = [row[4] for row in table.grids[parity] if row[0] >= 4.0]
        assert all(b >= a * 0.9 for a, b in zip(totals, totals[1:]))
    # the queryable total is the sum of the spin-resolved level densities
    row = table.grids[1][40]
    assert table.rho(row[0], parity=1) == pytest.approx(sum(row[5:]), rel=0.02)

    # characterization: 1/D0 vs rho(Bn, J = I0 +/- 1/2, s-wave parity), x3
    params = source.level_density_params("57Fe", model="egsm")
    d0_mev = params.spacing_kev.value * 1e-3
    rho = table.rho(params.bn_mev, two_j=1, parity=1)
    assert 1 / (rho * d0_mev) < 3.0
    assert 1 / (rho * d0_mev) > 1 / 3.0
