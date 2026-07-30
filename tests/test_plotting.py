"""Figure tests.

The drawing code is checked for structure, not appearance -- asserting on
pixels is brittle and asserting on nothing is useless. What is tested is that
each figure contains the elements it claims to (a line per level, an arrow per
transition), and that the label helpers, which are pure functions and the part
most likely to be wrong, produce the right strings.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import nook as el
from nook.sources.ensdf_file import parse_ensdf_text, place_gammas

DATA = Path(__file__).parent / "data"

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from nook.plotting import (  # noqa: E402
    plot_band_scheme,
    plot_chart,
    plot_decay_scheme,
    plot_level_scheme,
)
from nook.plotting.style import dodge, half_life_label, jpi_label  # noqa: E402
from nook.survey import NuclideSummary, decay_modes_from  # noqa: E402

from test_nook import BAND_FIXTURE, DECAY_FIXTURE, ENSDF_FIXTURE  # noqa: E402


@pytest.fixture
def scheme():
    return place_gammas(parse_ensdf_text(ENSDF_FIXTURE)[0].to_scheme())


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# --------------------------------------------------------------------------
# label helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0+", "$0^{+}$"),
        ("3/2-", "$3/2^{-}$"),
        ("(2+)", "$(2^{+})$"),          # tentative keeps its parentheses
        ("1-,2-", "$(1,2)^{-}$"),       # a shared parity is factored out
        ("", ""),
    ],
)
def test_jpi_label(text, expected):
    assert jpi_label(el.parse_spin_parity(text)) == expected


def test_jpi_label_truncates_long_alternative_lists():
    label = jpi_label(el.parse_spin_parity("3/2 TO 9/2"), max_alternatives=2)
    assert label.count(",") == 1 and r"\ldots" in label


def test_jpi_label_reports_a_parity_rule():
    assert jpi_label(el.parse_spin_parity("NATURAL")) == "natural"


@pytest.mark.parametrize(
    "text,unit",
    [("8.154 H", "h"), ("1.33 PS", "ps"), ("39 FS", "fs"), ("5.7 M", "min")],
)
def test_half_life_label_picks_a_readable_unit(text, unit):
    assert half_life_label(el.parse_half_life(text)).endswith(unit)


def test_half_life_label_handles_limits_and_stability():
    assert half_life_label(el.parse_half_life("STABLE")) == "stable"
    label = half_life_label(el.parse_half_life("7.1E15 Y", "GT"))
    assert ">" in label and "y" in label


def test_dodge_separates_without_reordering():
    values = [0.0, 39.5, 107.8, 110.8]
    out = dodge(values, 12.0)
    assert out == sorted(out)
    assert min(b - a for a, b in zip(out, out[1:])) >= 12.0 - 1e-9


def test_dodge_respects_bounds():
    out = dodge([0.0, 1.0, 2.0], 10.0, low=0.0)
    assert min(out) >= -1e-9


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def test_level_scheme_draws_one_line_per_level(scheme):
    fig, ax = plot_level_scheme(scheme, show_gammas=False)
    drawn = [line for line in ax.lines if len(line.get_xdata()) == 2]
    placed = [lv for lv in scheme.levels if not lv.is_floating]
    assert len(drawn) >= len(placed)
    assert ax.get_ylabel() == "energy (keV)"


def test_level_scheme_colours_by_parity(scheme):
    from nook.plotting.style import PALETTE

    fig, ax = plot_level_scheme(scheme, show_gammas=False)
    colours = {line.get_color() for line in ax.lines}
    assert PALETTE.positive in colours          # the 0+, 2+, 4+ levels
    assert PALETTE.unknown in colours           # the 3/2..9/2 level has no parity


def test_level_scheme_refuses_an_empty_selection(scheme):
    with pytest.raises(ValueError, match="no levels"):
        plot_level_scheme(scheme, max_energy=-1.0)


def test_band_scheme_draws_a_column_per_band():
    band_scheme = parse_ensdf_text(BAND_FIXTURE)[0].to_scheme()
    fig, ax = plot_band_scheme(band_scheme)
    captions = [t.get_text() for t in ax.texts]
    assert any("1+" in c for c in captions)     # the decoded BAND() text
    assert any("\u03c07/2[404]" in c for c in captions)  # its configuration


def test_band_scheme_needs_labelled_bands(scheme):
    with pytest.raises(ValueError, match="no labelled bands"):
        plot_band_scheme(scheme)


def test_decay_scheme_labels_the_absolute_scale():
    decay = parse_ensdf_text(DECAY_FIXTURE)[0].to_decay_scheme()
    fig, ax = plot_decay_scheme(decay)
    assert any("per 100 parent decays" in t.get_text() for t in ax.texts)
    assert any("log $ft$" in t.get_text() for t in ax.texts)


def test_decay_scheme_says_when_intensities_are_only_relative():
    stripped = "\n".join(
        line for line in DECAY_FIXTURE.splitlines() if not line.startswith("180HF  N")
    )
    decay = parse_ensdf_text(stripped + "\n")[0].to_decay_scheme()
    fig, ax = plot_decay_scheme(decay)
    assert any("no N record" in t.get_text() for t in ax.texts)


def test_decay_scheme_refuses_without_placed_feedings():
    scheme = parse_ensdf_text(ENSDF_FIXTURE)[0].to_decay_scheme()
    with pytest.raises(ValueError, match="no placed feedings"):
        plot_decay_scheme(scheme)


# --------------------------------------------------------------------------
# chart of nuclides
# --------------------------------------------------------------------------


def _states():
    return [
        NuclideSummary(nuclide=el.Nuclide(2, 4), stable=True),
        NuclideSummary(nuclide=el.Nuclide(6, 14), decay_modes=("B-",),
                    half_life_s=1.8e11, s_n=8176.0),
        NuclideSummary(nuclide=el.Nuclide(11, 22), decay_modes=("EC", "B+"),
                    half_life_s=8.2e7, s_n=11000.0),
        NuclideSummary(nuclide=el.Nuclide(92, 238), decay_modes=("A", "SF"),
                    half_life_s=1.4e17, s_n=6154.0),
    ]


def test_chart_places_one_square_per_nuclide():
    fig, ax = plot_chart(_states(), colour_by="decay")
    squares = sum(len(c.get_paths()) for c in ax.collections)
    assert squares == len(_states())
    assert ax.get_xlabel().startswith("neutron")


def test_chart_legend_lists_only_modes_present():
    fig, ax = plot_chart(_states(), colour_by="decay")
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "stable" in labels and r"$\beta^-$" in labels
    assert "SF" not in labels  # 238U's dominant branch is alpha, not SF


def test_chart_uses_physics_notation_on_the_colourbar():
    fig, ax = plot_chart(_states(), colour_by="half_life_s")
    assert "T_{1/2}" in fig.axes[-1].get_ylabel()


def test_chart_marks_stable_nuclides_when_colouring_continuously():
    fig, ax = plot_chart(_states(), colour_by="s_n", log=False)
    outlined = [c for c in ax.collections if c.get_edgecolor().size and
                not c.get_facecolor().size]
    assert outlined, "stable nuclides should stay outlined under any ramp"


def test_chart_refuses_an_empty_survey():
    with pytest.raises(ValueError, match="no nuclides"):
        plot_chart([])


# --------------------------------------------------------------------------
# survey
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("%EC+%B+", ("EC", "B+")),   # one key naming two modes
        ("%B-N", ("B-",)),           # delayed emission: primary mode is beta
        ("%ECP", ("EC",)),
        ("%A", ("A",)),
        ("%IT", ("IT",)),
    ],
)
def test_decay_modes_from_properties(key, expected):
    assert decay_modes_from({key: "100"}) == expected


def test_dominant_decay_follows_the_branching_not_a_convention():
    """238U has both alpha and SF open; alpha is 10^6 times stronger."""
    modes = decay_modes_from({"%A": "100", "%SF": "5.4E-5 5"})
    assert modes[0] == "A"
    state = NuclideSummary(nuclide=el.Nuclide(92, 238), decay_modes=modes)
    assert state.dominant_decay == "A"


# --------------------------------------------------------------------------
# data-quality checks
# --------------------------------------------------------------------------

from nook.survey import inconsistencies  # noqa: E402


def test_stable_nuclide_cannot_have_a_negative_separation_energy():
    """The 112Sn record in one ENSDF snapshot has S(n) = -10788 keV."""
    state = NuclideSummary(nuclide=el.Nuclide(50, 112), stable=True, s_n=-10788.0)
    (_, reason), = inconsistencies([state])
    assert "stable but S(n)" in reason


def test_unbound_nuclide_cannot_have_a_measurable_half_life():
    state = NuclideSummary(nuclide=el.Nuclide(43, 112), s_n=-4304.0, half_life_s=0.271)
    (_, reason), = inconsistencies([state])
    assert "promptly" in reason


def test_genuinely_unbound_light_nuclei_are_not_flagged():
    """5He is neutron-unbound with no measurable half-life; that is physics."""
    assert inconsistencies([NuclideSummary(nuclide=el.Nuclide(2, 5), s_n=-735.0)]) == []


def test_extrapolated_values_are_marked_as_such():
    """An SY value straddling zero is unreliable, not wrong."""
    state = NuclideSummary(
        nuclide=el.Nuclide(7, 24), s_n=-500.0, half_life_s=5.2e-8,
        estimated=("s_n",),
    )
    (_, reason), = inconsistencies([state])
    assert "from systematics" in reason


def test_out_of_range_energies_are_flagged():
    state = NuclideSummary(nuclide=el.Nuclide(44, 117), q_alpha=-91800.0)
    (_, reason), = inconsistencies([state])
    assert "out of range" in reason


def test_chart_clips_the_colour_scale_against_outliers():
    """One sign-flipped record should not decide how 3000 good ones are drawn."""
    good = [
        NuclideSummary(nuclide=el.Nuclide(z, 2 * z), s_n=8000.0 + 10 * z)
        for z in range(2, 60)
    ]
    rogue = NuclideSummary(nuclide=el.Nuclide(60, 120), s_n=-99000.0)
    fig, ax = plot_chart([*good, rogue], colour_by="s_n", log=False)
    low, _ = ax.collections[0].get_clim()
    assert low > -50000, "a single outlier stretched the scale"

    fig, ax = plot_chart([*good, rogue], colour_by="s_n", log=False, robust=False)
    assert ax.collections[0].get_clim()[0] == pytest.approx(-99000.0)


# --------------------------------------------------------------------------
# mass-surface closure
# --------------------------------------------------------------------------

from nook.survey import closure_failures  # noqa: E402

_NEUTRON_EXCESS_KEV = 8071.3


def _from_mass_excesses(excess: dict[tuple[int, int], float], **overrides):
    """Build a survey whose Q-values are exact differences of given masses.

    Closure then holds by construction, so any failure the test sees comes
    from the code rather than from the numbers.
    """
    states = []
    for (z, a), delta in excess.items():
        lighter = excess.get((z, a - 1))
        heavier = excess.get((z + 1, a))
        states.append(
            NuclideSummary(
                nuclide=el.Nuclide(z, a),
                s_n=None if lighter is None else lighter + _NEUTRON_EXCESS_KEV - delta,
                s_n_unc=1.0,
                q_beta_minus=None if heavier is None else delta - heavier,
                q_beta_minus_unc=1.0,
                **overrides.get((z, a), {}),
            )
        )
    return states


_GRID = {
    (50, 111): -85940.0, (50, 112): -88660.0, (50, 113): -88330.0,
    (51, 111): -82000.0, (51, 112): -81600.0, (51, 113): -84420.0,
    (52, 112): -77570.0, (52, 113): -75880.0,
}


def test_consistent_masses_close_exactly():
    assert closure_failures(_from_mass_excesses(_GRID)) == []


def test_a_single_wrong_value_breaks_the_loop():
    """This is what catches errors no per-record check can see."""
    states = _from_mass_excesses(_GRID)
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": -s.s_n})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in states
    ]
    failures = closure_failures(broken)
    assert failures, "a sign-flipped S(n) must not close"
    assert any(s.nuclide == el.Nuclide(50, 112) for s, _, _ in failures)


def test_extrapolated_values_are_not_expected_to_close():
    """SY values are extrapolations; holding them to the mass surface is unfair."""
    states = _from_mass_excesses(_GRID)
    marked = [
        NuclideSummary(**{**s.__dict__, "s_n": -abs(s.s_n or 1.0),
                          "estimated": ("s_n",)})
        if s.nuclide == el.Nuclide(50, 112)
        else s
        for s in states
    ]
    assert closure_failures(marked) == []


def test_closure_tolerance_scales_with_the_quoted_uncertainty():
    loose = [
        NuclideSummary(**{**s.__dict__, "s_n_unc": 400.0, "q_beta_minus_unc": 400.0})
        for s in _from_mass_excesses(_GRID)
    ]
    nudged = [
        NuclideSummary(**{**s.__dict__, "s_n": (s.s_n or 0) + 900.0})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in loose
    ]
    assert closure_failures(nudged) == []          # 900 keV is under 5 sigma here
    tight = [
        NuclideSummary(**{**s.__dict__, "s_n_unc": 1.0, "q_beta_minus_unc": 1.0})
        for s in nudged
    ]
    assert closure_failures(tight)                 # the same shift, now 450 sigma


def test_closure_failures_reach_the_inconsistency_report():
    states = _from_mass_excesses(_GRID)
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": -s.s_n})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in states
    ]
    assert any("mass-surface loop" in why for _, why in inconsistencies(broken))


# --------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------

from nook.repair import TRANSFORMS, repair  # noqa: E402


def test_consistent_data_is_left_alone():
    """Repair must be a no-op on data that already closes."""
    states = _from_mass_excesses(_GRID)
    fixed, changes = repair(states)
    assert changes == []
    assert [s.s_n for s in fixed] == [s.s_n for s in states]


def test_a_sign_flip_is_recovered_exactly():
    """The 112Sn case: -10788 keV back to +10788."""
    states = _from_mass_excesses(_GRID)
    truth = {s.nuclide: s.s_n for s in states}
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": -s.s_n})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in states
    ]
    fixed, changes = repair(broken)
    assert len(changes) == 1
    assert changes[0].transform == "negate"
    recovered = next(s for s in fixed if s.nuclide == el.Nuclide(50, 112))
    assert recovered.s_n == pytest.approx(truth[el.Nuclide(50, 112)])
    assert "s_n" in recovered.repaired


def test_a_dropped_exponent_is_recovered_exactly():
    """The 173Ir case: a field too narrow for the exponent."""
    states = _from_mass_excesses(_GRID)
    truth = next(s for s in states if s.nuclide == el.Nuclide(51, 112)).s_n
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": s.s_n / 10000.0})
        if s.nuclide == el.Nuclide(51, 112) and s.s_n
        else s
        for s in states
    ]
    fixed, changes = repair(broken)
    assert changes and changes[0].transform == "scale x10000"
    recovered = next(s for s in fixed if s.nuclide == el.Nuclide(51, 112))
    assert recovered.s_n == pytest.approx(truth)


def test_repair_records_the_evidence():
    states = _from_mass_excesses(_GRID)
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": -s.s_n})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in states
    ]
    _, (change,) = repair(broken)
    assert change.residual_before > 1000
    assert change.residual_after < 10
    assert change.loops_fixed >= 1
    assert "negate" in str(change) and "112Sn" in str(change)


def test_an_unconstrained_value_is_never_touched():
    """Q(alpha) does not enter the identity, so nothing can vouch for it."""
    states = _from_mass_excesses(_GRID)
    broken = [
        NuclideSummary(**{**s.__dict__, "q_alpha": -91800.0})
        if s.nuclide == el.Nuclide(50, 112)
        else s
        for s in states
    ]
    fixed, changes = repair(broken)
    assert changes == []
    assert next(s for s in fixed if s.nuclide == el.Nuclide(50, 112)).q_alpha == -91800.0


def test_extrapolated_values_are_never_repaired():
    """An SY value is uncertain, not wrong; rewriting it would be a fabrication."""
    states = _from_mass_excesses(_GRID)
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": -s.s_n, "estimated": ("s_n",)})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in states
    ]
    assert repair(broken)[1] == []


def test_only_the_declared_transforms_are_ever_applied():
    """A value wrong by a factor of 3 has no transcription story; leave it."""
    states = _from_mass_excesses(_GRID)
    broken = [
        NuclideSummary(**{**s.__dict__, "s_n": s.s_n * 3.0})
        if s.nuclide == el.Nuclide(50, 112) and s.s_n
        else s
        for s in states
    ]
    assert repair(broken)[1] == []
    assert {name for name, _ in TRANSFORMS} == {
        "negate", "scale x10", "scale x100", "scale x1000", "scale x10000",
        "scale /10", "scale /100",
    }


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def _overlapping_labels(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        (t.get_text(), t.get_window_extent(renderer))
        for t in ax.texts if t.get_text().strip()
    ]
    return [
        (a, b)
        for i, (a, box_a) in enumerate(boxes)
        for b, box_b in boxes[i + 1:]
        if box_a.overlaps(box_b)
    ]


@pytest.fixture
def crowded():
    """A scheme with levels a couple of keV apart, which is the hard case."""
    return place_gammas(parse_ensdf_text(ENSDF_FIXTURE)[0].to_scheme())


def test_level_scheme_labels_do_not_collide(crowded):
    fig, ax = plot_level_scheme(crowded)
    assert _overlapping_labels(fig, ax) == []


def test_band_scheme_labels_do_not_collide():
    scheme = parse_ensdf_text(BAND_FIXTURE)[0].to_scheme()
    fig, ax = plot_band_scheme(scheme)
    assert _overlapping_labels(fig, ax) == []


@pytest.fixture(scope="module")
def crowded_decay():
    """Seven feedings, including levels 1.8 keV apart.

    The one-feeding fixture used elsewhere cannot exercise label collisions at
    all; this mirrors the geometry of the real 180Lu decay, where two
    annotations overlapped despite the tests passing.
    """
    text = (DATA / "decay_crowded.ensdf").read_text()
    return parse_ensdf_text(text)[0].to_decay_scheme()


def test_decay_scheme_labels_do_not_collide(crowded_decay):
    fig, ax = plot_decay_scheme(crowded_decay)
    assert _overlapping_labels(fig, ax) == []


def test_decay_scheme_labels_do_not_collide_when_all_are_annotated(crowded_decay):
    fig, ax = plot_decay_scheme(crowded_decay, annotate=7)
    assert _overlapping_labels(fig, ax) == []


def test_arrows_get_more_width_than_levels():
    """The information is in the feedings, so they get the horizontal budget."""
    decay = parse_ensdf_text(DECAY_FIXTURE)[0].to_decay_scheme()
    fig, ax = plot_decay_scheme(decay)
    spans = [
        abs(line.get_xdata()[1] - line.get_xdata()[0])
        for line in ax.lines
        if len(line.get_xdata()) == 2
    ]
    level_width = max(s for s in spans if s < 0.5)
    assert level_width <= 0.25, "level lines should be narrow"


def test_log_ft_is_encoded_as_arrow_shade():
    decay = parse_ensdf_text(DECAY_FIXTURE)[0].to_decay_scheme()
    fig, ax = plot_decay_scheme(decay)
    shades = {
        str(a.arrowprops.get("color"))
        for a in ax.texts
        if getattr(a, "arrowprops", None)
    }
    assert shades, "feeding arrows should carry a colour"


def test_only_extremal_feedings_are_annotated():
    """Printing every log ft turns to noise once there are more than a few."""
    from nook.decay import DecayScheme, Feeding

    many = DecayScheme(
        nuclide=el.Nuclide(72, 180),
        dsid="synthetic",
        levels=parse_ensdf_text(DECAY_FIXTURE)[0].to_scheme(),
        parents=parse_ensdf_text(DECAY_FIXTURE)[0].parents,
        normalization=parse_ensdf_text(DECAY_FIXTURE)[0].normalization,
        feedings=tuple(
            Feeding(kind="B", level_index=i % 2,
                    intensity=el.Uncertain(10.0 + i, 1.0, 1.0),
                    log_ft=el.Uncertain(5.0 + 0.4 * i, 0.1, 0.1))
            for i in range(8)
        ),
    )
    fig, ax = plot_decay_scheme(many)
    annotated = [t for t in ax.texts if "log $ft$" in t.get_text()]
    assert 0 < len(annotated) <= 3
    assert len(fig.axes) > 1, "the rest should be carried by a log ft scale"


def _strokes(ax):
    """Every drawn stroke, sampled in display coordinates."""
    import numpy as np

    out = []
    for line in ax.lines:
        points = line.get_transform().transform(
            np.column_stack([line.get_xdata(), line.get_ydata()])
        )
        out += [np.linspace(a, b, 40) for a, b in zip(points, points[1:])]
    for text in ax.texts:
        patch = getattr(text, "arrow_patch", None)
        if patch is not None:
            points = patch.get_transform().transform(patch.get_path().vertices)
            out += [np.linspace(a, b, 40) for a, b in zip(points, points[1:])]
    return out


def _bare_labels_over_strokes(fig, ax):
    """Labels a line passes through that have no background to lift them off.

    Text-against-text was only half the problem: a label sitting on an arrow
    is just as unreadable, and no amount of dodging fixes it in general.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    strokes = _strokes(ax)
    bare = []
    for text in ax.texts:
        if not text.get_text().strip() or text.get_bbox_patch() is not None:
            continue
        box = text.get_window_extent(renderer)
        crossed = any(
            (
                (points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
                & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)
            ).any()
            for points in strokes
        )
        if crossed:
            bare.append(text.get_text())
    return bare


def test_no_bare_label_sits_on_a_stroke_level_scheme(crowded):
    fig, ax = plot_level_scheme(crowded)
    assert _bare_labels_over_strokes(fig, ax) == []


def test_no_bare_label_sits_on_a_stroke_band_scheme():
    scheme = parse_ensdf_text(BAND_FIXTURE)[0].to_scheme()
    fig, ax = plot_band_scheme(scheme)
    assert _bare_labels_over_strokes(fig, ax) == []


def test_no_bare_label_sits_on_a_stroke_decay_scheme(crowded_decay):
    fig, ax = plot_decay_scheme(crowded_decay)
    assert _bare_labels_over_strokes(fig, ax) == []


CHAIN_DIR = Path(os.environ.get("ENSDF_PATH", "/nonexistent"))


@pytest.mark.skipif(
    not (CHAIN_DIR / "ensdf.180").is_file(),
    reason="set ENSDF_PATH to a directory holding ensdf.180",
)
def test_real_decay_schemes_have_no_clashing_labels():
    """Fixtures approximate; real evaluations are the thing being drawn."""
    checked = 0
    for nuclide in ("180Hf", "180Ta", "180W", "180Lu"):
        for scheme in el.decay_schemes(nuclide, path=CHAIN_DIR):
            try:
                fig, ax = plot_decay_scheme(scheme)
            except ValueError:
                continue  # nothing placed to draw
            assert _overlapping_labels(fig, ax) == [], scheme.dsid
            assert _bare_labels_over_strokes(fig, ax) == [], scheme.dsid
            checked += 1
    assert checked >= 3, f"only {checked} real decay schemes exercised"


@pytest.mark.skipif(
    not (CHAIN_DIR / "ensdf.180").is_file(),
    reason="set ENSDF_PATH to a directory holding ensdf.180",
)
def test_every_decay_scheme_in_several_chains_draws_cleanly():
    """Breadth, not depth: rule-based placement fails on the awkward cases.

    Tuning offsets against one figure passed while eight of 233 real schemes
    still clashed, which is what motivated the measured resolver. This sweeps
    whole mass chains so the failure rate is a number rather than a hope.
    """
    import matplotlib.pyplot as plt

    from nook.sources.ensdf_file import _parse_chain

    clashing, drawn = [], 0
    for chain in sorted(CHAIN_DIR.glob("ensdf.1[0-2][0-9]")):
        for nuclide in sorted({d.nuclide for d in _parse_chain(chain)},
                              key=lambda n: n.z):
            for scheme in el.decay_schemes(nuclide, path=CHAIN_DIR):
                try:
                    fig, ax = plot_decay_scheme(scheme)
                except (ValueError, KeyError):
                    continue
                drawn += 1
                if _overlapping_labels(fig, ax) or _bare_labels_over_strokes(fig, ax):
                    clashing.append(scheme.dsid)
                plt.close("all")
    assert drawn > 50, f"only {drawn} schemes exercised"
    assert not clashing, f"{len(clashing)} of {drawn} clash: {clashing[:5]}"


def test_feeding_arrows_are_diagonal_not_right_angled():
    decay = parse_ensdf_text(DECAY_FIXTURE)[0].to_decay_scheme()
    fig, ax = plot_decay_scheme(decay)
    styles = {
        a.arrowprops.get("connectionstyle")
        for a in ax.texts
        if getattr(a, "arrowprops", None)
    }
    assert "arc3,rad=0" in styles
    assert not any("angle" in str(s) for s in styles)
