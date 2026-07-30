"""Decay scheme drawings.

The classic layout: the parent sits above and to one side, the daughter's
levels form a ladder, and feeding arrows run from the parent onto the levels
they populate. Feeding intensity sets arrow width and is printed with the
log *ft*, because those two together are what a reader checks -- a low
log *ft* onto a level two units of spin away is a red flag.

Intensities are drawn on the absolute scale (per 100 parent decays) whenever
the dataset carries an ``N`` record, since relative intensities are
meaningless outside their own dataset. The axis label says which you got.

The two quantities get separate visual channels: arrow *width* is intensity,
arrow *darkness* is log *ft*. Printing every log *ft* beside its arrow works
for four or five branches and turns to noise beyond that, so only the extremes
are labelled -- the fastest and slowest transitions, and the strongest branch.
Those are the ones a reader checks anyway.
"""

from __future__ import annotations

import math

from .style import (
    HALO,
    PALETTE,
    _require_matplotlib,
    dodge,
    figure_style,
    half_life_label,
    jpi_label,
    separate_labels,
)

__all__ = ["plot_decay_scheme"]


#: log ft spans roughly 3 (superallowed) to 9 (second-forbidden). Dark means
#: fast, which is the direction attention should go.
_LOGFT_RANGE = (4.0, 9.0)


def _logft_ramp():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "logft", [PALETTE.negative, "#5E7C91", "#A9BAC6", "#D6DEE4"]
    )


def _feeding_rows(scheme, max_levels: int):
    """Fed levels with their absolute intensity, strongest first."""
    rows = []
    for feeding, absolute in scheme.absolute_feedings():
        level = scheme.levels.level_by_index(feeding.level_index)
        if level is None or level.energy_kev is None or level.is_floating:
            continue
        value = (
            absolute.value
            if absolute.value is not None
            else (feeding.intensity.value or 0.0)
        )
        rows.append((level, feeding, absolute, value))
    rows.sort(key=lambda row: -row[3])
    return rows[:max_levels]


def _renderer(fig):
    """The renderer for a drawn figure; not on the abstract canvas type."""
    fig.canvas.draw()
    return fig.canvas.get_renderer()  # type: ignore[attr-defined]


def plot_decay_scheme(
    scheme,
    max_levels: int = 10,
    show_gammas: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (8.6, 6.4),
    annotate: int | None = None,
):
    """Draw a decay scheme: parent, feedings and the daughter's levels.

    ``annotate`` caps how many feedings are labelled in place. The default
    labels all of them when there are five or fewer, and otherwise only the
    extremes: fastest log *ft*, slowest log *ft*, and strongest branch.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _feeding_rows(scheme, max_levels)
    if not rows:
        raise ValueError("no placed feedings to draw")

    absolute_scale = any(a.value is not None for _, _, a, _ in rows)
    ceiling = max(row[0].energy_kev for row in rows)
    parent = scheme.parent
    q_value = (
        parent.q_value.value
        if parent is not None and parent.q_value.value
        else ceiling * 1.35
    )
    # a decay that feeds only the ground state has no vertical extent at all;
    # give it a nominal one rather than a singular transform
    top = max(q_value, ceiling * 1.15, 1.0)

    logft_values = [
        row[1].log_ft.value
        for row in rows
        if row[1].log_ft is not None and row[1].log_ft.value is not None
    ]
    if annotate is None:
        annotate = len(rows) if len(rows) <= 5 else 3

    # the levels need only enough width to carry a label; the arrows are
    # where the information is, so they get the rest
    with figure_style():
        fig, ax = plt.subplots(figsize=figsize)
        left, right = 0.60, 0.80
        parent_x0, parent_x1 = 0.02, 0.16

        # Fix the axes limits before anything is placed. Several labels are
        # positioned by measuring what has already been drawn, and changing
        # the limits afterwards would move the ground under those
        # measurements -- which is exactly how two of them ended up
        # overlapping despite the checks passing.
        ax.set_xlim(0, 1.35)
        ax.set_ylim(-top * 0.15, top * 1.18)

        # -- parent ------------------------------------------------------
        ax.plot(
            [parent_x0, parent_x1],
            [q_value, q_value],
            color=PALETTE.ink,
            linewidth=2.0,
            solid_capstyle="butt",
        )
        if parent is not None and parent.nuclide is not None:
            name = f"$^{{{parent.nuclide.a}}}$much".replace(
                "much", parent.nuclide.symbol
            )
            ax.text(
                parent_x0,
                q_value + top * 0.085,
                name,
                fontsize=13,
                ha="left",
                bbox=HALO,
                zorder=6,
            )
            meta = "   ".join(
                bit
                for bit in (
                    jpi_label(parent.spin_parity),
                    half_life_label(parent.half_life),
                )
                if bit
            )
            meta_text = ax.text(
                parent_x0,
                q_value + top * 0.028,
                meta,
                fontsize=9,
                ha="left",
                color=PALETTE.unknown,
                bbox=HALO,
                zorder=6,
            )
            if parent.q_value.value:
                # the parent's Jpi and half-life sit just above; how far they
                # reach depends on the data, so measure before placing Q
                edge = meta_text.get_window_extent(_renderer(fig)).x1
                q_x = max(
                    parent_x1 + 0.015,
                    ax.transData.inverted().transform((edge, 0))[0] + 0.02,
                )
                ax.text(
                    q_x,
                    q_value,
                    f"$Q$ = {parent.q_value.value:.0f} keV",
                    fontsize=8,
                    va="center",
                    ha="left",
                    color=PALETTE.unknown,
                    family="sans-serif",
                    bbox=HALO,
                    zorder=6,
                )

        movable: list = []
        strongest = max(row[3] for row in rows) or 1.0
        ramp = _logft_ramp()
        low, high = _LOGFT_RANGE
        if logft_values:
            low = min(low, min(logft_values))
            high = max(high, max(logft_values))

        def shade(feeding):
            """Darkness carries log ft; unknown stays neutral grey."""
            if feeding.log_ft is None or feeding.log_ft.value is None:
                return PALETTE.unknown
            fraction = (feeding.log_ft.value - low) / (high - low or 1.0)
            return ramp(min(max(fraction, 0.0), 1.0))

        # Which feedings earn an in-place label. Keyed on the feeding, not on
        # the level: several branches can populate the same level, and keying
        # on the level would label every one of them.
        labelled: set[int] = set()
        if logft_values:
            ranked = sorted(
                (
                    r
                    for r in rows
                    if r[1].log_ft is not None and r[1].log_ft.value is not None
                ),
                key=lambda r: r[1].log_ft.value,
            )
            for candidate in (ranked[0], ranked[-1], max(rows, key=lambda r: r[3])):
                if len(labelled) < annotate:
                    labelled.add(id(candidate[1]))
        for row in sorted(rows, key=lambda r: -r[3]):
            if len(labelled) < annotate:
                labelled.add(id(row[1]))

        # -- feedings ----------------------------------------------------
        # each feeding drops down its own lane, so a fan of arrows reads as
        # several branches instead of one overprinted bundle
        by_energy = sorted(rows, key=lambda row: row[0].energy_kev)
        # Labels ride their own diagonal, so separating them means choosing a
        # y for each and then solving back for the x on that arrow. Dodging
        # only the annotated ones keeps the spread as small as possible.
        # Each annotated label sits at a different fraction along its own
        # arrow, ordered so the steepest (highest-energy) branches keep their
        # labels near the parent and the shallow ones carry theirs further
        # down. Solving from a dodged height instead runs into arrows that
        # simply do not reach it, and the clamp then piles them back up.
        annotated = [row for row in reversed(by_energy) if id(row[1]) in labelled]
        span = 0.58
        spread = {
            id(row[1]): 0.30 + span * (i / max(len(annotated) - 1, 1))
            for i, row in enumerate(annotated)
        }

        for lane, (level, feeding, absolute, value) in enumerate(by_energy):
            width = 0.6 + 3.0 * math.sqrt(max(value, 0.0) / strongest)
            colour = shade(feeding)
            # Each branch leaves the parent bar at its own point and runs
            # straight to its level. Because both the departure point and the
            # arrival energy increase together, the diagonals fan without
            # crossing, which right-angled routing could not manage.
            drop = parent_x1 + 0.02 + 0.26 * (lane / max(len(by_energy) - 1, 1))
            ax.plot(
                [parent_x1, drop],
                [q_value, q_value],
                color=colour,
                linewidth=width,
                solid_capstyle="butt",
            )
            ax.annotate(
                "",
                xy=(left, level.energy_kev),
                xytext=(drop, q_value),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "linewidth": width,
                    "shrinkA": 0,
                    "shrinkB": 1,
                    "mutation_scale": 9,
                    "connectionstyle": "arc3,rad=0",
                },
            )
            if id(feeding) not in labelled:
                continue
            shown = absolute.value if absolute.value is not None else value
            note = f"{shown:.3g}{'%' if absolute_scale else ''}"
            if feeding.log_ft is not None and feeding.log_ft.value is not None:
                note += f"  log $ft$ {feeding.log_ft.value:.2f}"
            # Ride the diagonal rather than queue in a column beside the
            # levels. The label sits at the point on its own arrow whose
            # height is the separated one, clamped so it stays on the visible
            # run; the halo covers whatever it still crosses.
            fraction = spread.get(id(feeding), 0.62)
            movable.append(
                ax.text(
                    drop + fraction * (left - drop),
                    q_value + fraction * (level.energy_kev - q_value),
                    note,
                    fontsize=8,
                    ha="center",
                    va="center",
                    color=PALETTE.ink,
                    family="sans-serif",
                    bbox=HALO,
                    zorder=6,
                )
            )

        # -- daughter levels ---------------------------------------------
        by_index = {row[0].index: row[0] for row in rows}
        ladder = sorted(by_index.values(), key=lambda lv: lv.energy_kev)
        ground = min(
            (lv for lv in scheme.levels.levels if lv.energy_kev is not None),
            key=lambda lv: lv.energy_kev,
        )
        if ground.index not in by_index:
            ladder.insert(0, ground)

        spin_y = dodge([lv.energy_kev for lv in ladder], top * 0.062)
        spin_texts, energy_labels = [], []
        for level, y in zip(ladder, spin_y):
            parity = (
                level.spin_parity.unique.parity if level.spin_parity.unique else None
            )
            colour = PALETTE.for_parity(parity)
            ax.plot(
                [left, right],
                [level.energy_kev, level.energy_kev],
                color=colour,
                linewidth=1.6,
                solid_capstyle="butt",
                zorder=3,
            )
            if abs(y - level.energy_kev) > top * 0.004:
                ax.plot(
                    [right + 0.005, right + 0.014],
                    [level.energy_kev, y],
                    color=PALETTE.rule,
                    linewidth=0.5,
                    zorder=1,
                )
            spin = jpi_label(level.spin_parity)
            if spin:
                spin_texts.append(
                    ax.text(
                        right + 0.018,
                        y,
                        spin,
                        fontsize=9,
                        va="center",
                        ha="left",
                        color=colour,
                        bbox=HALO,
                        zorder=5,
                    )
                )
            energy_labels.append((y, f"{level.energy_kev:.1f}"))

        # The spin column's width depends on the data -- "(2,3+,1-)" is far
        # wider than "0+" -- so measure it and start the energies clear of the
        # widest one rather than trusting a fixed offset.
        energy_x = right + 0.085
        if spin_texts:
            renderer = _renderer(fig)
            widest = max(t.get_window_extent(renderer).x1 for t in spin_texts)
            in_data = ax.transData.inverted().transform((widest, 0))[0]
            energy_x = max(energy_x, in_data + 0.022)
        for y, label in energy_labels:
            ax.text(
                energy_x,
                y,
                label,
                fontsize=8,
                va="center",
                ha="left",
                color=PALETTE.unknown,
                family="sans-serif",
                bbox=HALO,
                zorder=5,
            )

        # -- gammas between drawn levels ---------------------------------
        if show_gammas:
            positions = {lv.index: lv.energy_kev for lv in ladder}
            for gamma in scheme.levels.gammas:
                if gamma.start_index in positions and gamma.end_index in positions:
                    ax.annotate(
                        "",
                        xy=(left + 0.06, positions[gamma.end_index]),
                        xytext=(left + 0.06, positions[gamma.start_index]),
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": PALETTE.ink,
                            "linewidth": 0.8,
                            "shrinkA": 0,
                            "shrinkB": 0,
                            "mutation_scale": 8,
                            "alpha": 0.55,
                        },
                    )

        daughter = scheme.nuclide
        ax.text(
            right,
            -top * 0.10,
            f"$^{{{daughter.a}}}${daughter.symbol}",
            fontsize=13,
            ha="right",
        )

        # a slim scale for the darkness channel
        if len(logft_values) > annotate:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            mappable = ScalarMappable(Normalize(low, high), ramp)
            bar = fig.colorbar(mappable, ax=ax, fraction=0.022, pad=0.02, aspect=28)
            bar.set_label("log $ft$", fontsize=9)
            bar.outline.set_visible(False)  # type: ignore[operator]
            bar.ax.invert_yaxis()

        separate_labels(fig, ax, [t for t in ax.texts if t.get_text().strip()])

        # Tightening at the very end is safe: shrinking the x-range can only
        # widen the pixel gaps that were measured against the wider one.
        renderer = _renderer(fig)
        rightmost = max(
            (
                ax.transData.inverted().transform(
                    (t.get_window_extent(renderer).x1, 0)
                )[0]
                for t in ax.texts
                if t.get_text().strip()
            ),
            default=1.0,
        )
        ax.set_xlim(0, min(1.35, rightmost + 0.03))
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("energy (keV)")
        ax.set_title(title or scheme.dsid, loc="left", pad=16)
        ax.text(
            0,
            1.005,
            "intensity per 100 parent decays"
            if absolute_scale
            else "relative intensity (no N record)",
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE.unknown,
            family="sans-serif",
        )
    return fig, ax
