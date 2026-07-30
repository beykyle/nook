"""Level and band scheme drawings.

The layout follows the Nuclear Data Sheets vernacular rather than a generic
plot: levels are horizontal rules on an energy axis, transitions are arrows
that run *downward* from the depopulated level, and arrow width carries
intensity. Readers of these figures already know that language, so borrowing
it costs nothing and saves a legend.

For band schemes the columns are the structure: one column per rotational
band, ordered by bandhead energy, so a K-isomer sits visibly apart from the
ground-state band instead of being buried in a single crowded ladder.
"""

from __future__ import annotations

import math
from typing import Sequence

from .style import (
    HALO,
    PALETTE,
    _require_matplotlib,
    dodge,
    figure_style,
    half_life_label,
    jpi_label,
)

__all__ = ["plot_level_scheme", "plot_band_scheme"]


def _sorted_levels(scheme, limit: int | None, max_energy: float | None):
    levels = [
        lv for lv in scheme.levels
        if lv.energy_kev is not None and not lv.is_floating
    ]
    if max_energy is not None:
        levels = [lv for lv in levels if lv.energy_kev <= max_energy]
    levels.sort(key=lambda lv: lv.energy_kev)
    return levels[:limit] if limit else levels


def _intensity_widths(gammas, lo: float = 0.6, hi: float = 3.4):
    """Map relative intensity onto a line width, on a square-root scale.

    Intensities span orders of magnitude; a linear map makes everything but
    the strongest transition invisible, and a log map exaggerates the weakest.
    """
    values = [
        g.intensity.value for g in gammas
        if g.intensity is not None and g.intensity.value
    ]
    if not values:
        return {id(g): lo for g in gammas}
    strongest = max(values)
    widths = {}
    for gamma in gammas:
        value = gamma.intensity.value if gamma.intensity else None
        if not value:
            widths[id(gamma)] = lo
        else:
            widths[id(gamma)] = lo + (hi - lo) * math.sqrt(value / strongest)
    return widths


def plot_level_scheme(
    scheme,
    max_energy: float | None = None,
    limit: int | None = 20,
    show_gammas: bool = True,
    title: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (5.2, 7.0),
):
    """Draw a single-column level scheme.

    ``limit`` caps the number of levels drawn from the bottom up, because a
    scheme with 240 levels is a solid black rectangle rather than a figure.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    levels = _sorted_levels(scheme, limit, max_energy)
    if not levels:
        raise ValueError("no levels with a definite energy to draw")

    with figure_style():
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        span = levels[-1].energy_kev - levels[0].energy_kev or 1.0
        left, right = 0.16, 0.60

        gammas = [
            g for g in scheme.gammas
            if show_gammas and g.start_index is not None and g.end_index is not None
        ]
        drawn = {lv.index for lv in levels}
        gammas = [g for g in gammas if g.start_index in drawn and g.end_index in drawn]
        widths = _intensity_widths(gammas)
        energies = {lv.index: lv.energy_kev for lv in levels}

        # fan the arrows across the level width so a cascade reads as
        # several transitions rather than one thick overprinted line
        lanes = sorted({g.start_index for g in gammas})
        for gamma in gammas:
            top, bottom = energies[gamma.start_index], energies[gamma.end_index]
            offset = lanes.index(gamma.start_index) / max(len(lanes), 1)
            x = left + (right - left) * (0.18 + 0.62 * offset)
            colour = PALETTE.rule if gamma.multipolarity is None else PALETTE.ink
            ax.annotate(
                "",
                xy=(x, bottom), xytext=(x, top),
                arrowprops={
                    "arrowstyle": "-|>", "color": colour,
                    "linewidth": widths[id(gamma)],
                    "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9,
                    "alpha": 0.85,
                },
            )

        energy_values = [lv.energy_kev for lv in levels]
        label_y = dodge(energy_values, span * 0.032)

        for level, y in zip(levels, label_y):
            energy = level.energy_kev
            colour = PALETTE.for_parity(
                level.spin_parity.unique.parity if level.spin_parity.unique else None
            )
            ax.plot([left, right], [energy, energy], color=colour, linewidth=1.6,
                    solid_capstyle="butt", zorder=3)
            if level.metastable:
                ax.plot([left, right], [energy, energy], color=PALETTE.isomer,
                        linewidth=4.5, alpha=0.35, solid_capstyle="butt", zorder=2)

            # leader line wherever the label had to move off its level
            if abs(y - energy) > span * 0.004:
                ax.plot([left - 0.055, left - 0.015], [y, energy],
                        color=PALETTE.rule, linewidth=0.5, zorder=1)
            ax.text(left - 0.06, y, f"{energy:.1f}", va="center", ha="right",
                    fontsize=8, color=PALETTE.ink, family="sans-serif",
                    bbox=HALO, zorder=5)

            label = jpi_label(level.spin_parity)
            if label:
                ax.text(right + 0.03, y, label, va="center", ha="left",
                        fontsize=10, color=colour, bbox=HALO, zorder=5)
            life = half_life_label(level.half_life)
            if life:
                emphasis = PALETTE.isomer if level.metastable else PALETTE.unknown
                ax.text(right + 0.22, y, life, va="center", ha="left",
                        fontsize=8, color=emphasis, family="sans-serif",
                        bbox=HALO, zorder=5)

        ax.set_ylim(levels[0].energy_kev - 0.06 * span,
                    levels[-1].energy_kev + 0.06 * span)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_position(("axes", -0.02))
        ax.set_ylabel("energy (keV)")
        ax.set_title(title or f"{scheme.nuclide}", loc="left", pad=14)
        if scheme.dataset:
            ax.text(0, 1.005, scheme.dataset, transform=ax.transAxes,
                    fontsize=8, color=PALETTE.unknown, family="sans-serif")
    return fig, ax


def plot_band_scheme(
    scheme,
    bands: Sequence[str] | None = None,
    max_bands: int = 5,
    max_energy: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 6.5),
):
    """Draw rotational bands side by side, one column each.

    Columns are ordered by bandhead energy. Each is tinted and labelled with
    its decoded ``BAND()`` text, so a K-isomer reads as a separate structure
    rather than as levels interleaved with the ground-state band.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    grouped = scheme.bands()
    labels = list(bands) if bands else list(grouped)[:max_bands]
    labels = [label for label in labels if label in grouped]
    if not labels:
        raise ValueError("no labelled bands in this scheme")

    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)
        column_width, gap = 0.68, 1.0
        top = 0.0
        captions = []

        for column, label in enumerate(labels):
            band = grouped[label]
            levels = _sorted_levels(band, None, max_energy)
            if not levels:
                continue
            x0 = column * gap
            x1 = x0 + column_width
            top = max(top, levels[-1].energy_kev)

            ax.axvspan(x0 - 0.09, x1 + 0.30, color=PALETTE.wash, zorder=0)

            # bandheads crowd together at the bottom of a rotational band,
            # so the spin labels need the same treatment as everywhere else
            extent = levels[-1].energy_kev - levels[0].energy_kev or 1.0
            label_y = dodge([lv.energy_kev for lv in levels], extent * 0.055)
            for level, y in zip(levels, label_y):
                energy = level.energy_kev
                parity = (
                    level.spin_parity.unique.parity
                    if level.spin_parity.unique else None
                )
                colour = PALETTE.for_parity(parity)
                ax.plot([x0, x1], [energy, energy], color=colour, linewidth=1.7,
                        solid_capstyle="butt", zorder=3)
                if level.metastable:
                    ax.plot([x0, x1], [energy, energy], color=PALETTE.isomer,
                            linewidth=5, alpha=0.35, solid_capstyle="butt", zorder=2)
                spin = jpi_label(level.spin_parity)
                if spin:
                    if abs(y - energy) > extent * 0.006:
                        ax.plot([x1 + 0.01, x1 + 0.03], [energy, y],
                                color=PALETTE.rule, linewidth=0.5, zorder=1)
                    ax.text(x1 + 0.04, y, spin, va="center", ha="left",
                            fontsize=9, color=colour, bbox=HALO, zorder=5)

            # in-band cascade
            energies = {lv.index: lv.energy_kev for lv in levels}
            inband = [
                g for g in band.gammas
                if g.start_index in energies and g.end_index in energies
            ]
            widths = _intensity_widths(inband, 0.5, 2.4)
            for gamma in inband:
                ax.annotate(
                    "",
                    xy=(x0 + column_width * 0.30, energies[gamma.end_index]),
                    xytext=(x0 + column_width * 0.30, energies[gamma.start_index]),
                    arrowprops={
                        "arrowstyle": "-|>", "color": PALETTE.ink,
                        "linewidth": widths[id(gamma)],
                        "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8,
                        "alpha": 0.75,
                    },
                )

            captions.append((x0 + column_width / 2, label, levels[0]))

        # one caption baseline for every column, so they read as a header row
        for x, label, head in captions:
            definition = scheme.band_definition(label) or ""
            title_text = definition.split(".")[0] if definition else f"band {label}"
            ax.text(x, -0.055 * top, title_text, ha="center", va="top",
                    fontsize=9, bbox=HALO, zorder=5)
            if "Configuration=" in definition:
                config = definition.split("Configuration=")[1].split()[0]
                ax.text(x, -0.105 * top, config, ha="center", va="top",
                        fontsize=8, color=PALETTE.unknown, bbox=HALO, zorder=5)
            ax.plot([x], [head.energy_kev], marker="o", markersize=3,
                    color=PALETTE.ink, zorder=4)

        ax.set_xlim(-0.25, len(labels) * gap - gap + column_width + 0.55)
        ax.set_ylim(-0.20 * top, top * 1.06)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("energy (keV)")
        ax.set_title(title or f"{scheme.nuclide} rotational bands",
                     loc="left", pad=14)
    return fig, ax
