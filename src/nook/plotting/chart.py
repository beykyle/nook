"""The chart of nuclides.

One square per nuclide on an N-Z grid. The chart is a map, so it is drawn like
one: magic numbers as ruled lines because they are the coastline everyone
navigates by, the valley of stability picked out in black, and everything else
coloured by whatever property is being asked about.

Two colouring modes, because they answer different questions. ``"decay"`` uses
the field's own categorical convention, so a reader who has seen a Karlsruhe
chart recognises it instantly. Any numeric attribute -- ``half_life_s``,
``s_n``, ``q_alpha`` -- gets a continuous ramp built from the package palette,
which keeps the chart visually of a piece with the level schemes.
"""

from __future__ import annotations

import math

from typing import Sequence

from .style import PALETTE, _require_matplotlib, figure_style

__all__ = ["AXIS_LABELS", "DECAY_COLOURS", "MAGIC_NUMBERS", "plot_chart"]

MAGIC_NUMBERS = (2, 8, 20, 28, 50, 82, 126)

#: Categorical colours, following the chart-of-nuclides convention but
#: re-hued to match the rest of the package.
DECAY_COLOURS = {
    "stable": PALETTE.ink,
    "B-": PALETTE.negative,
    "EC": PALETTE.positive,
    "B+": PALETTE.positive,
    "A": PALETTE.isomer,
    "IT": "#6E4E8F",
    "P": "#2E7D6B",
    "N": "#7A8B3D",
    "SF": "#8C5A3C",
}

#: Proper notation for the quantities a chart is usually coloured by.
AXIS_LABELS = {
    "half_life_s": r"$\log_{10}\,(T_{1/2}\,/\,\mathrm{s})$",
    "s_n": r"$S_n$  (keV)",
    "s_p": r"$S_p$  (keV)",
    "q_alpha": r"$Q_\alpha$  (keV)",
    "q_beta_minus": r"$Q_{\beta^-}$  (keV)",
    "two_j": r"ground-state $2J$",
    "residual_kev": r"$\Delta_{\mathrm{th}}-\Delta_{\mathrm{exp}}$  (keV)",
    "beta2": r"$\beta_2$",
    "spacing_kev": r"$\log_{10}\,(D_0\,/\,\mathrm{keV})$",
    "strength_1e4": r"$S_0$  $(10^{-4})$",
    "gamma_width_mev_milli": r"$\langle\Gamma_\gamma\rangle$  (meV)",
}

_DECAY_LABELS = {
    "stable": "stable", "B-": r"$\beta^-$", "EC": r"$\varepsilon/\beta^+$",
    "A": r"$\alpha$", "IT": "IT", "P": "$p$", "N": "$n$", "SF": "SF",
}


def _ramp():
    """A sequential colormap in the package's own hues."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "ensdf",
        [PALETTE.positive, PALETTE.isomer, "#E8E2D2", "#6E93A8", PALETTE.negative],
    )


def _decay_key(state) -> str:
    if state.stable:
        return "stable"
    mode = state.dominant_decay
    if mode in ("B+", "EC"):
        return "EC"
    return mode if mode in DECAY_COLOURS else "other"


def plot_chart(
    states,
    colour_by: str = "decay",
    log: bool | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 7.0),
    label_elements: bool = True,
    robust: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    mark: "Sequence | None" = None,
    mark_label: str = "flagged",
    ax=None,
    colorbar: bool = True,
):
    """Draw a chart of nuclides from :func:`nook.survey.survey` output.

    ``colour_by`` is ``"decay"`` for the categorical convention, or the name of
    any numeric :class:`~nook.survey.NuclideSummary` attribute.  ``log``
    defaults to true for half-lives, which span some thirty decades.

    ``ax`` draws into an existing axes -- the hook multi-panel figures such as
    :func:`~nook.plotting.plot_resonance_charts` use; ``figsize`` is ignored
    then.  ``colorbar=False`` skips the per-axes bar so panel figures sharing
    one scale can draw a single shared bar instead.

    ``robust`` clips the colour scale to the 1st-99th percentile. Evaluated
    files contain the odd transcription error -- a sign-flipped separation
    energy is enough to stretch the scale until the real structure washes
    out -- and one bad record should not decide how 3000 good ones are drawn.
    Pass ``vmin``/``vmax`` to set the range yourself, or ``robust=False`` for
    the full extent. See :func:`~nook.survey.inconsistencies`.

    ``mark`` rings a set of nuclides -- handy for showing which squares are
    flagged as suspect, or simply the ones you care about.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    states = [s for s in states if s.nuclide.z >= 0]
    if not states:
        raise ValueError("no nuclides to draw")
    if log is None:
        log = colour_by == "half_life_s"

    with figure_style():
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)

        max_n = max(s.n for s in states)
        max_z = max(s.z for s in states)
        for magic in MAGIC_NUMBERS:
            if magic <= max_n:
                ax.axvline(magic + 0.5, color=PALETTE.rule, linewidth=0.7, zorder=0)
                ax.text(magic + 0.5, -3.5, str(magic), fontsize=7, ha="center",
                        color=PALETTE.unknown, family="sans-serif")
            if magic <= max_z:
                ax.axhline(magic + 0.5, color=PALETTE.rule, linewidth=0.7, zorder=0)
                ax.text(-3.5, magic + 0.5, str(magic), fontsize=7, va="center",
                        ha="right", color=PALETTE.unknown, family="sans-serif")

        squares, values, edges = [], [], []
        if colour_by == "decay":
            for state in states:
                squares.append(Rectangle((state.n - 0.5, state.z - 0.5), 1, 1))
                edges.append(DECAY_COLOURS.get(_decay_key(state), PALETTE.unknown))
            collection = PatchCollection(
                squares, facecolors=edges, edgecolors="none", zorder=2
            )
            ax.add_collection(collection)
            present = {_decay_key(s) for s in states}
            handles = [
                Line2D([], [], marker="s", linestyle="none", markersize=7,
                       markerfacecolor=DECAY_COLOURS[key], markeredgecolor="none",
                       label=_DECAY_LABELS.get(key, key))
                for key in ("stable", "B-", "EC", "A", "IT", "P", "N", "SF")
                if key in present
            ]
            ax.legend(handles=handles, loc="upper left", ncol=2,
                      bbox_to_anchor=(0.01, 0.99), handletextpad=0.4)
        else:
            for state in states:
                value = getattr(state, colour_by, None)
                if value is None or (log and value <= 0):
                    continue
                squares.append(Rectangle((state.n - 0.5, state.z - 0.5), 1, 1))
                values.append(math.log10(value) if log else value)
            collection = PatchCollection(squares, cmap=_ramp(), edgecolors="none",
                                         zorder=2)
            collection.set_array(values)
            if robust and len(values) >= 5 and (vmin is None or vmax is None):
                # Drop at least one value from each end: a 1% index rounds to
                # zero for small surveys and would pick the outlier itself.
                ordered = sorted(values)
                margin = max(1, round(0.01 * len(ordered)))
                collection.set_clim(
                    ordered[margin] if vmin is None else vmin,
                    ordered[-1 - margin] if vmax is None else vmax,
                )
            elif vmin is not None or vmax is not None:
                collection.set_clim(vmin, vmax)
            ax.add_collection(collection)
            if colorbar:
                bar = fig.colorbar(collection, ax=ax, pad=0.015, fraction=0.03)
                label = AXIS_LABELS.get(colour_by, colour_by.replace("_", " "))
                if log and colour_by not in AXIS_LABELS:
                    label = rf"$\log_{{10}}$ {label}"
                bar.set_label(label, fontsize=9)
                bar.outline.set_visible(False)  # type: ignore[operator]

            # stable nuclides stay legible whatever the ramp is doing
            stable = [
                Rectangle((s.n - 0.5, s.z - 0.5), 1, 1) for s in states if s.stable
            ]
            ax.add_collection(PatchCollection(
                stable, facecolors="none", edgecolors=PALETTE.ink,
                linewidths=0.45, zorder=3,
            ))

        if mark:
            wanted = {(n.z, n.a) for n in mark}
            rings = [
                Rectangle((s.n - 0.5, s.z - 0.5), 1, 1)
                for s in states if (s.z, s.nuclide.a) in wanted
            ]
            if rings:
                ax.add_collection(PatchCollection(
                    rings, facecolors="none", edgecolors=PALETTE.isomer,
                    linewidths=1.6, zorder=5,
                ))
                ax.scatter([], [], marker="s", s=46, facecolors="none",
                           edgecolors=PALETTE.isomer, linewidths=1.6,
                           label=mark_label)
                ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))

        if label_elements:
            from .. import nuclide as _nuclide

            seen: dict[int, list[int]] = {}
            for state in states:
                seen.setdefault(state.z, []).append(state.n)
            for z, ns in sorted(seen.items()):
                if z % 10 == 0 and z > 0:
                    try:
                        symbol = _nuclide.z_to_symbol(z)
                    except ValueError:
                        continue  # beyond the named elements (HFB-14 reaches Z=120)
                    ax.text(min(ns) - 1.6, z, symbol, fontsize=7,
                            va="center", ha="right", color=PALETTE.ink)

        ax.set_xlim(-5, max_n + 3)
        ax.set_ylim(-5, max_z + 3)
        ax.set_aspect("equal")
        ax.set_xlabel("neutron number $N$")
        ax.set_ylabel("proton number $Z$")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title or f"chart of nuclides ({len(states)} nuclides)",
                     loc="left", pad=12)
    return fig, ax
