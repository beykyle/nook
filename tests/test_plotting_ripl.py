"""RIPL-3 figures: structure and measured layout, not pixels.

Same philosophy as test_plotting.py -- count what should exist, measure that
no two labels overlap and that no bare label sits on a stroke -- with data
from the committed mirror, so most of these need no external files at all.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import nook  # noqa: E402
from nook.compare import LevelComparison, LevelMatch  # noqa: E402
from nook.model import Level  # noqa: E402
from nook.plotting import (  # noqa: E402
    PALETTE,
    plot_chart_panels,
    plot_deformation_chart,
    plot_fission_barriers,
    plot_gdr,
    plot_gsf,
    plot_level_comparison,
    plot_level_density,
    plot_mass_residuals,
    plot_matter_density,
    plot_resonance_charts,
)
from nook.plotting.ripl import _ChartState  # noqa: E402
from nook.quantities import Uncertain, parse_spin_parity  # noqa: E402
from nook.sources.ripl3 import (  # noqa: E402
    DensityTable,
    FissionBarriers,
    GDREntry,
    GSFTable,
    MatterDensity,
    Ripl3Source,
)
from test_plotting import _bare_labels_over_strokes, _overlapping_labels  # noqa: E402
from test_ripl3 import needs_mirror  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def source():
    return Ripl3Source()


NUC = nook.Nuclide(26, 57)


# --------------------------------------------------------------------------
# empty input refuses cleanly (no mirror needed)
# --------------------------------------------------------------------------


def test_gdr_refuses_empty():
    with pytest.raises(ValueError, match="no GDR entries"):
        plot_gdr(())


def test_gdr_refuses_theor_only():
    theor = GDREntry(nuclide=NUC, kind="theor",
                     peaks=((Uncertain(16.0), Uncertain(5.0), None),))
    with pytest.raises(ValueError, match="cross-sections"):
        plot_gdr([theor])


def test_gsf_refuses_empty():
    with pytest.raises(ValueError, match="no strength-function points"):
        plot_gsf(GSFTable(nuclide=NUC, rows=()))


def test_level_density_refuses_empty():
    with pytest.raises(ValueError, match="no level-density grid"):
        plot_level_density(DensityTable(nuclide=NUC, grids={}))


def test_level_density_refuses_impossible_spin():
    table = DensityTable(nuclide=NUC, grids={1: ((1.0, 0.5, 1.0, 1.0, 2.0),)})
    with pytest.raises(ValueError, match="impossible"):
        plot_level_density(table, two_js=(2,))  # even 2J for odd A


def test_matter_density_refuses_empty():
    with pytest.raises(ValueError, match="no density profile"):
        plot_matter_density(MatterDensity(nuclide=NUC, beta2=None, rows=()))


def test_fission_refuses_empty():
    with pytest.raises(ValueError, match="no fission barriers"):
        plot_fission_barriers(
            FissionBarriers(nuclide=NUC, model="empirical", barriers=())
        )


def test_mass_residuals_refuses_empty():
    with pytest.raises(ValueError, match="no nuclides"):
        plot_mass_residuals({})


def test_deformation_chart_refuses_empty():
    with pytest.raises(ValueError, match="no nuclides"):
        plot_deformation_chart({})


def test_resonance_charts_refuses_empty():
    with pytest.raises(ValueError, match="no resonance entries"):
        plot_resonance_charts({})


def test_chart_panels_refuses_empty():
    with pytest.raises(ValueError, match="no panels"):
        plot_chart_panels([], colour_by="s_n")
    states = [_state(12, 24, s_n=16531.0)]
    with pytest.raises(ValueError, match="no beta2 values"):
        plot_chart_panels([("a", states)], colour_by="beta2")


def _state(z, a, **quantity):
    return _ChartState(nuclide=nook.Nuclide(z, a), z=z, n=a - z, **quantity)


def test_chart_panels_share_one_scale_and_bar():
    panel_a = [_state(12, 24, s_n=16531.0), _state(12, 25, s_n=7331.0)]
    panel_b = [_state(12, 24, s_n=16400.0), _state(12, 26, s_n=11093.0)]
    fig, axes = plot_chart_panels(
        [("ensdf", panel_a), ("ripl3", panel_b)], colour_by="s_n",
    )
    assert len(axes) == 2
    clims = [ax.collections[0].get_clim() for ax in axes]
    assert clims[0] == clims[1]
    assert len(fig.axes) == 3  # two panels + one shared colourbar
    labels = {t.get_text() for ax in axes for t in ax.texts}
    assert {"ensdf", "ripl3"} <= labels


def test_plot_chart_colorbar_can_be_skipped():
    from nook.plotting import plot_chart

    fig, ax = plot_chart([_state(12, 24, s_n=16531.0)], colour_by="s_n",
                         log=False, colorbar=False)
    assert len(fig.axes) == 1


def test_level_comparison_refuses_empty():
    empty = LevelComparison(nuclide=NUC, source_a="a", source_b="b",
                            matched=(), only_a=(), only_b=())
    with pytest.raises(ValueError, match="no levels to compare"):
        plot_level_comparison(empty)


# --------------------------------------------------------------------------
# a synthetic comparison (no mirror needed)
# --------------------------------------------------------------------------


def _level(index, energy, jpi=None, assumed=False):
    from dataclasses import replace

    from nook.quantities import SpinParity

    spin_parity = parse_spin_parity(jpi) if jpi else SpinParity()
    if assumed:
        spin_parity = replace(spin_parity, assumed=True)
    return Level(index=index, energy=Uncertain(energy), spin_parity=spin_parity)


def _synthetic_comparison():
    a1, b1 = _level(1, 0.0, "0+"), _level(1, 0.0, "0+")
    a2, b2 = _level(2, 1368.7, "2+"), _level(2, 1368.6, "2+", assumed=True)
    a3 = _level(3, 4122.9, "4+")
    b4 = _level(3, 5100.0, "3-")
    return LevelComparison(
        nuclide=nook.Nuclide(12, 24), source_a="ensdf", source_b="ripl3-local",
        matched=(
            LevelMatch(a=a1, b=b1, delta_kev=0.0, combined_sigma_kev=None,
                       jpi_agree=True),
            LevelMatch(a=a2, b=b2, delta_kev=0.1, combined_sigma_kev=0.2,
                       jpi_agree=True),
        ),
        only_a=(a3,), only_b=(b4,),
        cutoff_a=3, cutoff_b=2,
    )


def test_level_comparison_draws_matches_and_ticks():
    comparison = _synthetic_comparison()
    fig, ax = plot_level_comparison(comparison)
    # two connectors, one per matched pair, crossing the gutter
    connectors = [
        line for line in ax.lines
        if len(line.get_xdata()) == 2
        and list(line.get_xdata()) == [0.42, 0.58]
    ]
    assert len(connectors) == comparison.n_matched
    # outward ticks for the one-sided levels, in the isomer hue
    ticks = [line for line in ax.lines
             if line.get_color() == PALETTE.isomer]
    assert len(ticks) == len(comparison.only_a) + len(comparison.only_b)
    # both cutoff hairlines present
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(dashed) == 2


def test_level_comparison_greys_assumed_spins():
    fig, ax = plot_level_comparison(_synthetic_comparison())
    labels = {t.get_text(): t.get_color() for t in ax.texts}
    assert labels["$2^{+}$"] == PALETTE.unknown or (
        # both sides carry a 2+ label; the RIPL one (assumed) must be grey,
        # so grey must appear among the 2+ labels
        PALETTE.unknown in [
            t.get_color() for t in ax.texts if t.get_text() == "$2^{+}$"
        ]
    )
    fig.canvas.draw()
    assert _overlapping_labels(fig, ax) == []
    assert _bare_labels_over_strokes(fig, ax) == []


# --------------------------------------------------------------------------
# against the real mirror
# --------------------------------------------------------------------------


@needs_mirror
def test_gdr_181ta_layout(source):
    entries = source.gdr("181Ta")
    fig, ax = plot_gdr(entries)
    slo = [e for e in entries if e.kind == "exp-SLO"]
    components = len(slo[0].peaks)
    # every extra evaluation is one thin curve; the primary adds its
    # components plus the summed curve
    assert len(ax.lines) == (len(slo) - 1) + components + 1
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(dashed) == components
    assert _overlapping_labels(fig, ax) == []
    assert _bare_labels_over_strokes(fig, ax) == []


@needs_mirror
def test_gsf_56fe_layout(source):
    table = source.gsf("56Fe")
    fig, ax = plot_gsf(table, sn_kev=source.levels("56Fe").sn_kev)
    assert ax.get_yscale() == "log"
    hairlines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(hairlines) == 1
    assert _overlapping_labels(fig, ax) == []


@needs_mirror
def test_level_density_57fe_layout(source):
    scheme = source.fetch("57Fe")
    fig, (ax_rho, ax_cum) = plot_level_density(
        source.hfb_density("57Fe"), scheme=scheme,
        ct=source.levels_param()[(26, 57)],
    )
    assert ax_rho.get_yscale() == "log" and ax_cum.get_yscale() == "log"
    # both parity hues drawn in the density panel
    colours = {line.get_color() for line in ax_rho.lines}
    assert PALETTE.positive in colours and PALETTE.negative in colours
    # the Nmax hairline spans both panels
    for panel in (ax_rho, ax_cum):
        assert any(line.get_linestyle() == "--" and len(line.get_xdata()) == 2
                   and line.get_xdata()[0] == line.get_xdata()[1]
                   for line in panel.lines) or any(
            line.get_linestyle() == "--" for line in panel.lines)
    # the staircase covers the drawable levels
    steps = [line for line in ax_cum.lines if line.get_drawstyle() != "default"]
    assert steps and len(steps[0].get_xdata()) > 50
    for panel in (ax_rho, ax_cum):
        assert _overlapping_labels(fig, panel) == []
        assert _bare_labels_over_strokes(fig, panel) == []


@needs_mirror
def test_matter_density_208pb_layout(source):
    density = source.matter_density("208Pb")
    fig, ax = plot_matter_density(density)
    principal = [line for line in ax.lines if len(line.get_xdata()) > 10]
    assert len(principal) == 2
    assert _overlapping_labels(fig, ax) == []
    assert _bare_labels_over_strokes(fig, ax) == []


@needs_mirror
def test_fission_238u_layout(source):
    barriers = source.fission_barriers("238U")
    fig, ax = plot_fission_barriers(
        barriers, overlay=source.fission_barriers("238U", model="hfb")
    )
    curves = [line for line in ax.lines if len(line.get_xdata()) > 10]
    assert len(curves) == 2
    # the solid landscape peaks at the tallest empirical barrier
    empirical = max(curves, key=lambda line: line.get_linewidth())
    assert max(empirical.get_ydata()) == pytest.approx(
        barriers.inner.height_mev.value, abs=0.05
    )
    text = {t.get_text() for t in ax.texts}
    assert any("GA" in t for t in text) and any("MA" in t for t in text)
    assert _overlapping_labels(fig, ax) == []
    assert _bare_labels_over_strokes(fig, ax) == []


@needs_mirror
def test_mass_residual_chart(source):
    table = source.mass_table()
    assert len(table) > 5000
    fig, axes = plot_mass_residuals(table)
    assert len(axes) == 2  # frdm95 and hfb14 panels
    clims = [ax.collections[0].get_clim() for ax in axes]
    assert clims[0] == clims[1]  # one shared scale, or the comparison is void
    lo, hi = clims[0]
    assert lo == pytest.approx(-hi)  # symmetric about zero
    assert len(fig.axes) == 3  # two panels + exactly one shared colourbar
    for ax in axes:
        collection = ax.collections[0]
        assert len(collection.get_paths()) > 1000
        # stable (abundance-bearing) nuclides keep the dark outline convention
        outlined = [c for c in ax.collections if c is not collection
                    and len(c.get_paths())]
        assert outlined


@needs_mirror
def test_deformation_chart(source):
    fig, axes = plot_deformation_chart(source.mass_table())
    assert len(axes) == 2
    clims = [ax.collections[0].get_clim() for ax in axes]
    assert clims[0] == clims[1]
    lo, hi = clims[0]
    assert lo == pytest.approx(-hi)


@needs_mirror
def test_resonance_charts(source):
    table = source.resonance_table()
    fig, axes = plot_resonance_charts(table)
    assert len(axes) == 3
    assert len(axes[0].collections[0].get_paths()) == len(
        [e for e in table.values() if e.spacing_kev.value is not None]
    )
    assert len(fig.axes) == 6  # three panels + three colourbars
    # only the top panel carries the title and only the bottom the x label
    assert axes[0].get_title(loc="left") and not axes[1].get_title(loc="left")
    assert not axes[0].get_xlabel() and axes[2].get_xlabel()


@needs_mirror
def test_level_comparison_real_ripl_vs_itself(source):
    # "file" needs ENSDF, so exercise the real-data path RIPL-vs-RIPL: the
    # comparison machinery is source-agnostic and the layout is what matters
    comparison = nook.compare.levels(
        "24Mg", sources=("ripl3", "ripl3"), below=10000.0
    )
    fig, ax = plot_level_comparison(comparison)
    assert comparison.n_matched > 3
    fig.canvas.draw()
    assert _overlapping_labels(fig, ax) == []
    assert _bare_labels_over_strokes(fig, ax) == []
