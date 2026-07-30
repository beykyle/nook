"""Figures for the RIPL-3 segments.

The segments are statistical-model inputs, so the figures show them the way a
reaction calculation consumes them: strength functions and level densities on
log axes against excitation energy, barriers as the double-humped potential
they parameterise, bulk tables as charts of nuclides. Everything draws from
the committed mirror -- no external data, unlike the ENSDF gallery.

Two segments are deliberately absent. Optical potentials are coefficient
records and nook does not evaluate V(r), so there is nothing honest to draw;
per-level coupled-channel deformations are niche input whose story the
ground-state beta_2 chart already tells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..nuclide import Nuclide
from .chart import AXIS_LABELS, plot_chart
from .style import (
    HALO,
    PALETTE,
    _require_matplotlib,
    figure_style,
    jpi_label,
    separate_labels,
)

__all__ = [
    "plot_chart_panels",
    "plot_deformation_chart",
    "plot_fission_barriers",
    "plot_gdr",
    "plot_gsf",
    "plot_level_comparison",
    "plot_level_density",
    "plot_mass_residuals",
    "plot_matter_density",
    "plot_resonance_charts",
]


def _data_labels(ax) -> list:
    """The axes' non-empty texts placed in data coordinates.

    ``separate_labels`` moves texts through ``transData``, so subtitles and
    headers pinned in axes coordinates must stay out of its hands.
    """
    return [
        t for t in ax.texts
        if t.get_text().strip() and t.get_transform() is ax.transData
    ]


def _slo(e: float, e0: float, w: float, sigma0: float) -> float:
    """The standard Lorentzian photoabsorption shape.

    sigma(E) = sigma0 E^2 Gamma^2 / ((E^2 - E0^2)^2 + E^2 Gamma^2) -- the
    form the exp-SLO parameters are fitted with, so drawing them through it
    adds nothing that is not already in the record.  (MLO widths are
    temperature-dependent and theor entries carry no sigma0, which is why
    only exp-SLO entries are drawn as curves.)
    """
    return sigma0 * e**2 * w**2 / ((e**2 - e0**2) ** 2 + e**2 * w**2)


def _grid(lo: float, hi: float, points: int = 400) -> list[float]:
    return [lo + i * (hi - lo) / points for i in range(points + 1)]


def plot_gdr(
    entries,
    kind: str = "exp-SLO",
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 4.4),
):
    """Draw giant-dipole-resonance Lorentzian fits.

    The first ``kind`` entry gets the full treatment -- components dashed,
    sum solid, peaks annotated; any further entries (other references) appear
    as thin sum curves so the evaluation spread stays visible without
    dominating the figure.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    entries = list(entries)
    if not entries:
        raise ValueError("no GDR entries to draw")
    drawable = [
        e for e in entries
        if e.kind == kind and e.peaks
        and all(cs is not None and cs.value for (_e, _w, cs) in e.peaks)
    ]
    if not drawable:
        raise ValueError(f"no {kind} GDR fits with peak cross-sections to draw")

    ranges = [e.fit_range_mev for e in drawable if e.fit_range_mev]
    lo = min((r[0] for r in ranges), default=5.0)
    hi = max((r[1] for r in ranges), default=30.0)
    pad = 0.15 * (hi - lo)
    xs = _grid(max(0.0, lo - pad), hi + pad)

    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)

        def total(entry, e: float) -> float:
            return sum(
                _slo(e, pe.value, pw.value, pcs.value)
                for pe, pw, pcs in entry.peaks
            )

        for entry in drawable[1:]:
            ax.plot(xs, [total(entry, e) for e in xs], color=PALETTE.rule,
                    linewidth=0.9, zorder=1)

        primary = drawable[0]
        hues = (PALETTE.positive, PALETTE.negative, PALETTE.isomer)
        for i, (pe, pw, pcs) in enumerate(primary.peaks):
            ax.plot(xs, [_slo(e, pe.value, pw.value, pcs.value) for e in xs],
                    color=hues[i % len(hues)], linewidth=1.1, linestyle="--",
                    zorder=2)
            peak_y = total(primary, pe.value)
            ax.text(pe.value, peak_y * 1.03,
                    rf"$E_0={pe.value:.1f}$, $\Gamma={pw.value:.1f}$ MeV",
                    ha="center", va="bottom", fontsize=8,
                    family="sans-serif", color=PALETTE.ink, bbox=HALO, zorder=5)
        ax.plot(xs, [total(primary, e) for e in xs], color=PALETTE.ink,
                linewidth=1.8, zorder=3)

        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(bottom=0)
        top = max(total(primary, e) for e in xs)
        ax.set_ylim(top=top * 1.22)
        ax.set_xlabel("photon energy (MeV)")
        ax.set_ylabel(r"$\sigma_{\mathrm{abs}}$  (mb)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title or f"{primary.nuclide}  GDR", loc="left", pad=12)
        note = f"{kind}, {primary.reference}"
        if len(drawable) > 1:
            note += f"  (+{len(drawable) - 1} more evaluations)"
        ax.text(0, 1.005, note, transform=ax.transAxes, fontsize=8,
                color=PALETTE.unknown, family="sans-serif")
        separate_labels(fig, ax, _data_labels(ax))
    return fig, ax


def plot_gsf(
    table,
    sn_kev: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 4.4),
):
    """Draw a microscopic E1 strength function.

    The neutron separation energy hairline marks where capture calculations
    actually sample the curve -- the region the GDR peak towers over.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    rows = [(e, f) for e, f in table.rows if f > 0]
    if not rows:
        raise ValueError("no strength-function points to draw")

    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)
        ax.semilogy([e for e, _ in rows], [f for _, f in rows],
                    color=PALETTE.ink, linewidth=1.6, zorder=3)
        if sn_kev is not None:
            sn = sn_kev / 1000.0
            ax.axvline(sn, color=PALETTE.rule, linewidth=0.8,
                       linestyle="--", zorder=1)
            ax.text(sn, rows[-1][1], r"$S_n$", ha="left", va="bottom",
                    fontsize=9, color=PALETTE.unknown, bbox=HALO, zorder=5)
        ax.set_xlim(left=0)
        ax.set_xlabel("photon energy (MeV)")
        ax.set_ylabel(r"$f_{E1}$  (mb/MeV)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title or f"{table.nuclide}  E1 strength", loc="left", pad=12)
        ax.text(0, 1.005, "HFB+QRPA microscopic table", transform=ax.transAxes,
                fontsize=8, color=PALETTE.unknown, family="sans-serif")
    return fig, ax


def _column_totals(grids, column: int) -> list[tuple[float, float]]:
    """Sum one grid column over parities, keyed by excitation energy."""
    totals: dict[float, float] = {}
    for grid in grids.values():
        for row in grid:
            if column < len(row):
                totals[row[0]] = totals.get(row[0], 0.0) + row[column]
    return sorted(totals.items())


def plot_level_density(
    table,
    scheme=None,
    ct=None,
    two_js: tuple[int, ...] = (1, 5, 11),
    max_energy_mev: float = 20.0,
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 7.6),
):
    """Draw the level density and the cumulative level count, one story.

    Two panels on a shared excitation-energy axis.  The top panel is the HFB
    density by parity (the one figure where the parity palette is literally
    its own semantics) with a few spin-resolved curves; the bottom is the
    discrete-level staircase against the constant-temperature fit (``ct``, a
    :class:`~nook.sources.ripl3.LevelsParam`) and the HFB cumulative count.
    A shared hairline at RIPL's Nmax ties the two regimes together: where the
    staircase falls away from the models above the line, levels are missing,
    not physics.

    Returns ``(fig, (ax_density, ax_cumulative))``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not table.grids:
        raise ValueError("no level-density grid to draw")
    a_parity = table.nuclide.a % 2
    for two_j in two_js:
        if (two_j - a_parity) % 2:
            raise ValueError(
                f"two_j={two_j} is impossible for A={table.nuclide.a}"
            )

    with figure_style():
        fig, (ax_rho, ax_cum) = plt.subplots(
            2, 1, sharex=True, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        )

        # -- density panel ---------------------------------------------------
        for parity in sorted(table.grids, reverse=True):
            rows = [
                (row[0], row[3]) for row in table.grids[parity]
                if row[3] > 0 and row[0] <= max_energy_mev
            ]
            if not rows:
                continue
            colour = PALETTE.for_parity(parity)
            ax_rho.semilogy([u for u, _ in rows], [r for _, r in rows],
                            color=colour, linewidth=1.6, zorder=3)
            sign = "+" if parity > 0 else "-"
            ax_rho.text(rows[-1][0], rows[-1][1], rf"  $\pi={sign}$",
                        ha="left", va="center", fontsize=9, color=colour,
                        bbox=HALO, zorder=5)
        for two_j in two_js:
            column = 5 + (two_j - a_parity) // 2
            rows = [
                (u, r) for u, r in _column_totals(table.grids, column)
                if r > 0 and u <= max_energy_mev
            ]
            if not rows:
                continue
            ax_rho.semilogy([u for u, _ in rows], [r for _, r in rows],
                            color=PALETTE.unknown, linewidth=0.8, zorder=2)
            spin = f"{two_j}/2" if a_parity else f"{two_j // 2}"
            ax_rho.text(rows[-1][0], rows[-1][1], rf"  $J={spin}$",
                        ha="left", va="center", fontsize=8,
                        color=PALETTE.unknown, bbox=HALO, zorder=5)
        ax_rho.set_ylabel(r"$\rho$  (MeV$^{-1}$)")

        # -- cumulative panel ------------------------------------------------
        nmax_energy = None
        if scheme is not None:
            levels = [
                lv for lv in scheme.levels
                if lv.energy_kev is not None and not lv.is_floating
            ]
            if levels:
                energies = sorted(lv.energy_kev / 1000.0 for lv in levels)
                steps = [e for e in energies if e <= max_energy_mev]
                ax_cum.step(steps, range(1, len(steps) + 1), where="post",
                            color=PALETTE.ink, linewidth=1.4, zorder=3)
                nmax = scheme.metadata.get("nmax")
                if nmax and nmax <= len(energies):
                    nmax_energy = energies[nmax - 1]
        if ct is not None and ct.temperature_mev:
            u0 = ct.u0_mev or 0.0
            xs = _grid(max(0.0, u0), max_energy_mev, 200)
            ax_cum.semilogy(
                xs, [math.exp((u - u0) / ct.temperature_mev) for u in xs],
                color=PALETTE.isomer, linewidth=1.2, linestyle="--", zorder=2,
            )
            ax_cum.text(xs[-1], math.exp((xs[-1] - u0) / ct.temperature_mev),
                        "  CT", ha="left", va="center", fontsize=8,
                        color=PALETTE.isomer, bbox=HALO, zorder=5)
        hfb_cum = [
            (u, n) for u, n in _column_totals(table.grids, 2)
            if n > 0 and u <= max_energy_mev
        ]
        if hfb_cum:
            ax_cum.semilogy([u for u, _ in hfb_cum], [n for _, n in hfb_cum],
                            color=PALETTE.unknown, linewidth=1.2,
                            linestyle=":", zorder=2)
            ax_cum.text(hfb_cum[-1][0], hfb_cum[-1][1], "  HFB",
                        ha="left", va="center", fontsize=8,
                        color=PALETTE.unknown, bbox=HALO, zorder=5)
        if nmax_energy is not None:
            for panel in (ax_rho, ax_cum):
                panel.axvline(nmax_energy, color=PALETTE.rule, linewidth=0.8,
                              linestyle="--", zorder=1)
            ax_cum.text(nmax_energy, 1.15, r"  complete to $N_{max}$",
                        ha="left", va="bottom", fontsize=8,
                        color=PALETTE.unknown, family="sans-serif",
                        bbox=HALO, zorder=5)

        ax_cum.set_xlim(0, max_energy_mev * 1.12)
        ax_cum.set_xlabel("excitation energy (MeV)")
        ax_cum.set_ylabel("cumulative count $N$")
        for panel in (ax_rho, ax_cum):
            for side in ("top", "right"):
                panel.spines[side].set_visible(False)
        ax_rho.set_title(title or f"{table.nuclide}  level density",
                         loc="left", pad=12)
        ax_rho.text(0, 1.005, "HFB+combinatorial table, by parity",
                    transform=ax_rho.transAxes, fontsize=8,
                    color=PALETTE.unknown, family="sans-serif", bbox=HALO)
        for panel in (ax_rho, ax_cum):
            separate_labels(fig, panel, _data_labels(panel))
    return fig, (ax_rho, ax_cum)


def plot_matter_density(
    density,
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 4.4),
):
    """Draw the HFB-14 radial matter-density profile, neutrons and protons.

    Neutrons in blue, protons in red -- the assignment every nuclear-physics
    reader expects -- with inline end labels instead of a legend.  Small ticks
    on the baseline mark each species' half-central-density radius, so a
    neutron skin is a visible offset between two ticks rather than a number.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    rows = density.rows
    if not rows:
        raise ValueError("no density profile to draw")

    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)
        radii = [r for r, _n, _p in rows]
        central = rows[0][1] + rows[0][2]
        edge = next(
            (r for r, n, p in rows if r > 1.0 and n + p < central * 1e-3),
            radii[-1],
        )
        for label, colour, offset, values in (
            (r"$\rho_n$", PALETTE.negative, +1, [n for _r, n, _p in rows]),
            (r"$\rho_p$", PALETTE.positive, -1, [p for _r, _n, p in rows]),
        ):
            ax.plot(radii, values, color=colour, linewidth=1.6, zorder=3)
            # label over the flat interior, clear of the surface fall-off
            anchor_r = 0.25 * edge
            anchor_v = values[min(range(len(radii)),
                                  key=lambda i: abs(radii[i] - anchor_r))]
            ax.text(anchor_r, anchor_v + offset * central * 0.02, label,
                    ha="center", va="bottom" if offset > 0 else "top",
                    fontsize=10, color=colour, bbox=HALO, zorder=5)
            half = next(
                (r for r, v in zip(radii, values) if v < values[0] / 2), None
            )
            if half is not None:
                ax.plot([half], [0], marker=2, markersize=7, color=colour,
                        clip_on=False, zorder=4)

        ax.set_xlim(0, edge * 1.15)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("radius (fm)")
        ax.set_ylabel(r"$\rho$  (fm$^{-3}$)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title or f"{density.nuclide}  matter density",
                     loc="left", pad=12)
        note = "HFB-14, spherically averaged"
        if density.beta2 is not None:
            note += rf"  ($\beta_2 = {density.beta2:.2f}$)"
        ax.text(0, 1.005, note, transform=ax.transAxes, fontsize=8,
                color=PALETTE.unknown, family="sans-serif")
        separate_labels(fig, ax, _data_labels(ax))
    return fig, ax


def _barrier_landscape(barriers, saddle_x=None):
    """A schematic double-humped potential through the recorded saddles.

    Heights are exact; widths scale with 1/(h-bar omega); everything else --
    well depths, the outgoing tail -- is presentation, which is why the x
    axis is labelled schematic.  ``saddle_x`` pins saddles onto another
    landscape's positions (matched by order, extras spaced as usual), so an
    overlaid model reads as a height comparison rather than a shape one.
    """
    heights = [b.height_mev.value or 0.0 for b in barriers.barriers]
    hws = [b.hw_mev.value or 1.0 for b in barriers.barriers]
    mean_hw = sum(hws) / len(hws)
    # clamp so one outlying curvature cannot stretch the whole landscape
    widths = [min(1.8, max(0.55, mean_hw / hw)) for hw in hws]

    positions: list[float] = []
    x = 0.0
    for i, width in enumerate(widths):
        x += 0.9 * width if i == 0 else 0.9 * (widths[i - 1] + 3 * width) / 2
        if saddle_x is not None and i < len(saddle_x):
            floor = positions[-1] + 0.3 if positions else 0.3
            x = max(saddle_x[i], floor)
        positions.append(x)

    # extrema: ground state, then saddle/well alternation, then the tail
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for i, (sx, height) in enumerate(zip(positions, heights)):
        points.append((sx, height))
        if i + 1 < len(heights):
            well = max(0.5, min(height, heights[i + 1]) - 2.0)
            points.append(((sx + positions[i + 1]) / 2, well))
    points.append((positions[-1] + 1.1 * widths[-1], -1.5))

    xs: list[float] = []
    ys: list[float] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        for t in _grid(0.0, 1.0, 40):
            blend = (1 - math.cos(math.pi * t)) / 2  # smooth at both extrema
            xs.append(x0 + (x1 - x0) * t)
            ys.append(y0 + (y1 - y0) * blend)
    saddles = points[1::2][: len(heights)]
    return xs, ys, saddles, positions


def plot_fission_barriers(
    barriers,
    overlay=None,
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 4.4),
):
    """Draw fission barriers as the double-humped potential they describe.

    Barrier heights and curvatures are the evaluated numbers; the deformation
    axis is schematic and says so.  ``overlay`` draws a second model family
    (typically HFB) dashed over the first.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not barriers.barriers:
        raise ValueError("no fission barriers to draw")

    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)
        xs, ys, saddles, positions = _barrier_landscape(barriers)
        ax.plot(xs, ys, color=PALETTE.ink, linewidth=1.8, zorder=3,
                label=barriers.model)
        if overlay is not None and overlay.barriers:
            oxs, oys, _saddles, _pos = _barrier_landscape(overlay, positions)
            ax.plot(oxs, oys, color=PALETTE.unknown, linewidth=1.2,
                    linestyle="--", zorder=2, label=overlay.model)
            ax.legend(loc="upper right")

        for i, ((x, height), barrier) in enumerate(zip(saddles, barriers.barriers)):
            name = chr(ord("A") + i)
            ax.text(x, height + 0.45, rf"$E_{name} = {height:.1f}$ MeV",
                    ha="center", va="bottom", fontsize=9, color=PALETTE.ink,
                    bbox=HALO, zorder=5)
            if barrier.symmetry:
                ax.text(x, height + 0.05, barrier.symmetry, ha="center",
                        va="bottom", fontsize=7, color=PALETTE.unknown,
                        family="sans-serif", bbox=HALO, zorder=5)

        ax.axhline(0.0, color=PALETTE.rule, linewidth=0.6, zorder=0)
        top = max(b.height_mev.value or 0.0 for b in barriers.barriers)
        ax.set_ylim(-2.2, top + 2.4)
        ax.set_xticks([])
        ax.set_xlabel("elongation (schematic)")
        ax.set_ylabel("energy (MeV)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title or f"{barriers.nuclide}  fission barriers",
                     loc="left", pad=12)
        if barriers.pairing_gap_mev is not None:
            ax.text(0, 1.005,
                    rf"pairing gap $\Delta_f = {barriers.pairing_gap_mev:.2f}$ MeV",
                    transform=ax.transAxes, fontsize=8,
                    color=PALETTE.unknown, family="sans-serif")
        separate_labels(fig, ax, _data_labels(ax))
    return fig, ax


@dataclass(frozen=True)
class _ChartState:
    """Duck-typed stand-in for the survey rows :func:`plot_chart` consumes."""

    nuclide: Nuclide
    z: int
    n: int
    stable: bool = False
    dominant_decay: str | None = None
    residual_kev: float | None = None
    beta2: float | None = None
    spacing_kev: float | None = None
    strength_1e4: float | None = None
    gamma_width_mev_milli: float | None = None
    s_n: float | None = None


def _symmetric_limit(values) -> float:
    """A robust symmetric colour limit: the 99th percentile of |value|."""
    ordered = sorted(abs(v) for v in values)
    return ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))]


def plot_chart_panels(
    panels,
    colour_by: str,
    log: bool = False,
    symmetric: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (8.6, 10.0),
):
    """Stacked chart-of-nuclides panels of the *same* quantity, one per source.

    ``panels`` is a sequence of ``(label, states)`` pairs.  Every panel shares
    a single colour scale and one colourbar -- computed over the union of all
    panels' values -- because a side-by-side source comparison on independent
    scales would compare nothing.  ``symmetric=True`` centres the shared scale
    on zero (for signed quantities like residuals); otherwise the robust
    1st-99th percentile of the union is used.

    Returns ``(fig, axes)`` with one axes per panel.
    """
    _require_matplotlib()
    import math as _math

    import matplotlib.pyplot as plt

    panels = [(label, list(states)) for label, states in panels]
    if not panels:
        raise ValueError("no panels to draw")
    values = [
        v for _label, states in panels for s in states
        for v in [getattr(s, colour_by, None)]
        if v is not None and not (log and v <= 0)
    ]
    if not values:
        raise ValueError(f"no {colour_by} values to draw")
    if log:
        values = [_math.log10(v) for v in values]
    if symmetric:
        limit = _symmetric_limit(values)
        vmin, vmax = -limit, limit
    else:
        ordered = sorted(values)
        margin = max(1, round(0.01 * len(ordered))) if len(ordered) >= 5 else 0
        vmin, vmax = ordered[margin], ordered[-1 - margin]

    with figure_style():
        fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True,
                                 squeeze=False)
        axes = tuple(axes.flat)
        for ax, (label, states) in zip(axes, panels):
            plot_chart(states, colour_by=colour_by, log=log, ax=ax,
                       vmin=vmin, vmax=vmax, colorbar=False)
            ax.set_title("", loc="left")
            ax.text(0, 1.005, label, transform=ax.transAxes, fontsize=8,
                    color=PALETTE.unknown, family="sans-serif")
        for ax in axes[:-1]:
            ax.set_xlabel("")
        mappable = axes[0].collections[0]
        bar = fig.colorbar(mappable, ax=list(axes), pad=0.015, fraction=0.03)
        bar_label = AXIS_LABELS.get(colour_by, colour_by.replace("_", " "))
        if log and colour_by not in AXIS_LABELS:
            bar_label = rf"$\log_{{10}}$ {bar_label}"
        bar.set_label(bar_label, fontsize=9)
        bar.outline.set_visible(False)  # type: ignore[operator]
        axes[0].set_title(title or f"{colour_by.replace('_', ' ')} by source",
                          loc="left", pad=12)
    return fig, axes


def plot_mass_residuals(table, theories=("frdm95", "hfb14"),
                        title: str | None = None, **panel_kwargs):
    """Panels of theory-minus-experiment mass residuals, one per mass model.

    Only measured masses count as experiment (``recommended_only`` entries are
    systematics, not data).  The panels share one colour scale, symmetric
    about zero so the diverging ramp reads as signed error, and the shell
    closures stand out on the magic-number gridlines the chart already draws.

    Returns ``(fig, axes)`` with one axes per theory.
    """
    panels = []
    for theory in theories:
        states = []
        for entry in table.values():
            theoretical = getattr(entry, f"mass_excess_{theory}")
            exp = entry.mass_excess_exp.value
            if theoretical is None or exp is None or entry.recommended_only:
                continue
            nuclide = entry.nuclide
            states.append(_ChartState(
                nuclide=nuclide, z=nuclide.z, n=nuclide.a - nuclide.z,
                stable=entry.abundance is not None,
                residual_kev=theoretical.value - exp,
            ))
        if states:
            panels.append((f"{theory}  ({len(states)} measured nuclides)",
                           states))
    if not panels:
        raise ValueError("no nuclides with both measured and model masses to draw")
    return plot_chart_panels(
        panels, colour_by="residual_kev", symmetric=True,
        title=title or "mass residuals against experiment",
        **panel_kwargs,
    )


def plot_deformation_chart(table, theories=("frdm95", "hfb14"),
                           title: str | None = None, **panel_kwargs):
    """Panels of ground-state quadrupole deformation, one per mass model.

    Returns ``(fig, axes)`` with one axes per theory.
    """
    panels = []
    for theory in theories:
        states = []
        for entry in table.values():
            beta2 = getattr(entry, f"beta2_{theory}")
            if beta2 is None:
                continue
            nuclide = entry.nuclide
            states.append(_ChartState(
                nuclide=nuclide, z=nuclide.z, n=nuclide.a - nuclide.z,
                stable=entry.abundance is not None, beta2=beta2,
            ))
        if states:
            panels.append((theory, states))
    if not panels:
        raise ValueError("no nuclides with a model deformation to draw")
    return plot_chart_panels(
        panels, colour_by="beta2", symmetric=True,
        title=title or "ground-state deformation by mass model",
        **panel_kwargs,
    )


def plot_resonance_charts(
    table,
    title: str | None = None,
    figsize: tuple[float, float] = (7.2, 12.6),
):
    """Three chart-of-nuclides panels: D0, S0 and the radiative width.

    One panel per quantity because they answer different questions -- spacing
    is a level-density measurement, the strength function a doorway-state
    average, the radiative width the gamma cascade's temperature.  D0 spans
    orders of magnitude and gets a log colour scale.

    Returns ``(fig, axes)`` with one axes per panel.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not table:
        raise ValueError("no resonance entries to draw")
    entries = list(table.values())

    panels = (
        ("spacing_kev", True, [e.spacing_kev.value for e in entries]),
        ("strength_1e4", False, [e.strength_1e4.value for e in entries]),
        ("gamma_width_mev_milli", False,
         [e.gamma_width_mev_milli.value for e in entries]),
    )
    with figure_style():
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        for ax, (quantity, log, values) in zip(axes, panels):
            states = [
                _ChartState(
                    nuclide=e.nuclide, z=e.nuclide.z,
                    n=e.nuclide.a - e.nuclide.z,
                    **{quantity: value},
                )
                for e, value in zip(entries, values) if value is not None
            ]
            plot_chart(states, colour_by=quantity, log=log, ax=ax,
                       label_elements=False)
            ax.set_title("", loc="left")
        for ax in axes[:-1]:
            ax.set_xlabel("")
        wave = entries[0].wave
        axes[0].set_title(
            title or f"{'sp'[wave]}-wave resonance systematics "
                     f"({len(entries)} targets)",
            loc="left", pad=12,
        )
    return fig, axes


def _ladder(values, gap: float, low: float, high: float) -> list[float]:
    """Waterfall label placement: on the level when there is room.

    ``dodge`` recentres a crowded block on its mean, which is right for a
    dozen labels but drags *isolated* labels off their levels once one dense
    run dominates the drift.  Here sparse labels stay exactly on their level;
    only crowded runs fan upward, then get pulled back under the ceiling.
    ``values`` must be ascending.
    """
    out: list[float] = []
    prev = low - gap
    for value in values:
        prev = max(value, prev + gap)
        out.append(prev)
    ceiling = high
    for i in range(len(out) - 1, -1, -1):
        out[i] = min(out[i], ceiling)
        ceiling = out[i] - gap
    return out


def _comparison_side(matched_levels, only_levels, max_energy):
    levels = sorted(
        (lv for lv in [*matched_levels, *only_levels]
         if lv.energy_kev is not None),
        key=lambda lv: lv.energy_kev,
    )
    if max_energy is not None:
        levels = [lv for lv in levels if lv.energy_kev <= max_energy]
    return levels


def _level_colour(level) -> str:
    """Parity colour, except that RIPL-invented spins never wear it.

    A spin the RIPL evaluators *assumed* (flagged ``g``/``c``/``n``) must not
    be rendered as if it were an evaluated assignment, so it stays grey.
    """
    if level.spin_parity.assumed:
        return PALETTE.unknown
    return PALETTE.for_parity(
        level.spin_parity.unique.parity if level.spin_parity.unique else None
    )


def plot_level_comparison(
    comparison,
    max_energy: float | None = None,
    limit: int | None = 30,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 8.0),
):
    """Draw one nuclide's level scheme from two sources, side by side.

    Matched pairs are joined across the gutter; levels only one source has
    get an outward tick, so a dropped level looks dropped rather than merely
    unmatched.  Dashed hairlines mark each side's completeness cutoff -- the
    heuristic estimate against RIPL's own Nmax, which is the comparison
    ``nook.compare`` exists to make.

    Energies are labelled on side a only: matched energies agree to a keV or
    better, so repeating them on side b doubles the label load for no
    information.  Side b labels its J-pi -- the quantity that genuinely can
    differ -- and the energies of levels side a lacks.

    ``limit`` caps each column from the bottom up, for the same reason
    :func:`~nook.plotting.plot_level_scheme` caps its single column: past
    thirty rules the labels have nowhere left to go.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    a_levels = _comparison_side(
        [m.a for m in comparison.matched], comparison.only_a, max_energy)
    b_levels = _comparison_side(
        [m.b for m in comparison.matched], comparison.only_b, max_energy)
    if limit:
        a_levels = a_levels[:limit]
        b_levels = b_levels[:limit]
    if not a_levels and not b_levels:
        raise ValueError("no levels to compare")
    kept = {id(lv) for lv in [*a_levels, *b_levels]}

    columns = {"a": (0.16, 0.42), "b": (0.58, 0.84)}
    top = max(lv.energy_kev for lv in [*a_levels, *b_levels])
    span = top or 1.0

    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.06 * span, 1.06 * span)

        for m in comparison.matched:
            if id(m.a) not in kept or id(m.b) not in kept:
                continue
            ax.plot([columns["a"][1], columns["b"][0]],
                    [m.a.energy_kev, m.b.energy_kev],
                    color=PALETTE.rule, linewidth=0.7, zorder=1)

        only = {
            "a": {id(lv) for lv in comparison.only_a},
            "b": {id(lv) for lv in comparison.only_b},
        }
        for side, levels in (("a", a_levels), ("b", b_levels)):
            left, right = columns[side]
            outward = -1 if side == "a" else 1
            edge = left if side == "a" else right
            # side a reads energy-then-Jpi outward; side b leads with Jpi,
            # its energies (only_b levels) sitting past it
            label_x = left - 0.035 if side == "a" else 0.98
            jpi_x = 0.02 if side == "a" else right + 0.035
            align = "right"
            jpi_align = "left"
            leader_x = (label_x if side == "a" else right + 0.03)

            label_y = _ladder([lv.energy_kev for lv in levels], span * 0.032,
                              low=-0.05 * span, high=1.03 * span)
            for level, y in zip(levels, label_y):
                energy = level.energy_kev
                colour = _level_colour(level)
                ax.plot([left, right], [energy, energy], color=colour,
                        linewidth=1.6, solid_capstyle="butt", zorder=3)
                if id(level) in only[side]:
                    ax.plot([edge, edge + outward * 0.025], [energy, energy],
                            color=PALETTE.isomer, linewidth=2.4,
                            solid_capstyle="butt", zorder=4)
                if abs(y - energy) > span * 0.004:
                    ax.plot([leader_x + outward * 0.005, edge + outward * 0.008],
                            [y, energy], color=PALETTE.rule, linewidth=0.5,
                            zorder=1)
                if side == "a" or id(level) in only[side]:
                    ax.text(label_x, y, f"{energy:.1f}", va="center", ha=align,
                            fontsize=8, color=PALETTE.ink, family="sans-serif",
                            bbox=HALO, zorder=5)
                label = jpi_label(level.spin_parity)
                if label:
                    ax.text(jpi_x, y, label, va="center", ha=jpi_align,
                            fontsize=9, color=colour, bbox=HALO, zorder=5)

        cutoffs = (
            ("a", comparison.cutoff_a, a_levels, "complete (heuristic)"),
            ("b", comparison.cutoff_b, b_levels, r"$N_{max}$"),
        )
        for side, cutoff, levels, note in cutoffs:
            if not cutoff or cutoff > len(levels):
                continue
            energy = levels[cutoff - 1].energy_kev
            left, right = columns[side]
            ax.plot([left - 0.02, right + 0.02], [energy, energy],
                    color=PALETTE.rule, linewidth=0.9, linestyle="--", zorder=2)
            ax.text((left + right) / 2, energy + 0.012 * span, note,
                    ha="center", va="bottom", fontsize=7,
                    color=PALETTE.unknown, family="sans-serif",
                    bbox=HALO, zorder=5)

        for side, name in (("a", comparison.source_a), ("b", comparison.source_b)):
            left, right = columns[side]
            ax.text((left + right) / 2, 1.005, name, transform=ax.transAxes,
                    ha="center", fontsize=8, color=PALETTE.unknown,
                    family="sans-serif")

        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_position(("axes", -0.02))
        ax.set_ylabel("energy (keV)")
        ax.set_title(title or f"{comparison.nuclide}", loc="left", pad=14)
        summary = [f"{comparison.n_matched} matched"]
        if comparison.rms_delta_kev is not None:
            summary.append(rf"rms $\Delta E$ {comparison.rms_delta_kev:.2g} keV")
        if comparison.jpi_agreement_fraction is not None:
            summary.append(
                rf"$J^\pi$ agree {100 * comparison.jpi_agreement_fraction:.0f}%"
            )
        ax.text(1.0, -0.015, "   ".join(summary), transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=PALETTE.unknown,
                bbox=HALO)
        separate_labels(fig, ax, _data_labels(ax))
    return fig, ax
