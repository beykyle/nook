"""Visual language for the figures.

Two decisions drive everything here.

**Type.** These figures land beside LaTeX-set body text in NIM, PRC or the
Nuclear Data Sheets, so they are set in Latin Modern with Computer Modern
mathtext -- the same faces as the surrounding prose. Matplotlib's DejaVu
default is the giveaway that a figure was made in a hurry.

**Colour encodes parity.** Positive and negative parity get their own hue, so
a reader sees the parity structure of a band before reading a single label.
That makes colour carry information rather than decorate; unassigned parity
stays grey precisely so that missing data looks missing.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "HALO",
    "separate_labels",
    "PALETTE",
    "Palette",
    "dodge",
    "figure_style",
    "half_life_label",
    "jpi_label",
]


@dataclass(frozen=True)
class Palette:
    """Named colours. Parity is the organising binary."""

    ink: str = "#14181F"  # level lines, text, axes
    paper: str = "#FFFFFF"
    rule: str = "#C9CDD3"  # hairlines, guides
    positive: str = "#A6321F"  # pi = +
    negative: str = "#1F5673"  # pi = -
    unknown: str = "#8A8F98"  # parity unassigned
    isomer: str = "#C98A2B"  # metastable states
    wash: str = "#F2F4F6"  # band backing tint

    def for_parity(self, parity: int | None) -> str:
        if parity == 1:
            return self.positive
        if parity == -1:
            return self.negative
        return self.unknown


PALETTE = Palette()

#: A soft background for labels that have to sit over a drawn line. Slightly
#: translucent, so it reads as the label lifting off the artwork rather than
#: as a hole punched through it.
HALO = {
    "boxstyle": "round,pad=0.22",
    "facecolor": PALETTE.paper,
    "edgecolor": "none",
    "alpha": 0.82,
}

#: Faces chosen to match LaTeX-set body text, with graceful fallbacks.
_SERIF = ["Latin Modern Roman", "TeX Gyre Termes", "DejaVu Serif"]
_SANS = ["Latin Modern Sans", "TeX Gyre Heros", "DejaVu Sans"]

_RC = {
    "font.family": "serif",
    "font.serif": _SERIF,
    "font.sans-serif": _SANS,
    "mathtext.fontset": "cm",
    "axes.edgecolor": PALETTE.ink,
    "axes.labelcolor": PALETTE.ink,
    "axes.linewidth": 0.7,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "text.color": PALETTE.ink,
    "xtick.color": PALETTE.ink,
    "ytick.color": PALETTE.ink,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": PALETTE.paper,
    "axes.facecolor": PALETTE.paper,
    "savefig.facecolor": PALETTE.paper,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}


@contextmanager
def figure_style(**overrides):
    """Apply the package's rcParams for the duration of a block."""
    import matplotlib.pyplot as plt

    # the stubs key rc dicts by a Literal of every rc name; plain str is fine
    with plt.rc_context({**_RC, **overrides}):  # pyright: ignore[reportArgumentType]
        yield


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'nook[plot]'"
        ) from exc


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def _one_jpi(two_j: int | None, parity: int | None) -> str:
    if two_j is None:
        spin = "?"
    elif two_j % 2 == 0:
        spin = str(two_j // 2)
    else:
        spin = rf"{two_j}/2"
    return spin + {1: "^{+}", -1: "^{-}"}.get(parity or 0, "")


def jpi_label(spin_parity, max_alternatives: int = 3) -> str:
    """Render a :class:`SpinParity` as mathtext.

    Alternatives are joined with commas and a shared parity is factored out,
    the way an evaluator would write it: ``1-,2-`` becomes ``(1,2)^-`` rather
    than repeating the superscript. Tentative assignments keep their
    parentheses because that distinction is physics, not styling.
    """
    if spin_parity is None:
        return ""
    if not spin_parity.candidates:
        if spin_parity.natural_parity is not None:
            return "natural" if spin_parity.natural_parity else "unnatural"
        return ""

    candidates = list(spin_parity.candidates)[:max_alternatives]
    parities = {c.parity for c in candidates}
    if len(candidates) > 1 and len(parities) == 1 and parities != {None}:
        spins = ",".join(_one_jpi(c.two_j, None) for c in candidates)
        body = f"({spins})" + _one_jpi(0, candidates[0].parity)[1:]
    elif len(candidates) > 1:
        body = ",".join(_one_jpi(c.two_j, c.parity) for c in candidates)
    else:
        body = _one_jpi(candidates[0].two_j, candidates[0].parity)

    if len(spin_parity.candidates) > max_alternatives:
        body += r"\ldots"
    if spin_parity.tentative:
        body = f"({body})"
    return f"${body}$"


_TIME_STEPS = (
    (3.15576e16, "Gy"),
    (3.15576e13, "My"),
    (3.15576e7, "y"),
    (86400.0, "d"),
    (3600.0, "h"),
    (60.0, "min"),
    (1.0, "s"),
    (1e-3, "ms"),
    (1e-6, "\u00b5s"),
    (1e-9, "ns"),
    (1e-12, "ps"),
    (1e-15, "fs"),
)


def half_life_label(half_life) -> str:
    """Render a half-life compactly, picking a sensible unit."""
    if half_life is None or not half_life.is_known:
        return ""
    if half_life.stable:
        return "stable"

    seconds = half_life.seconds.value
    if seconds is None:
        return ""
    operator = {"GT": ">", "GE": r"\geq", "LT": "<", "LE": r"\leq"}.get(
        half_life.seconds.operator, ""
    )
    for scale, unit in _TIME_STEPS:
        if seconds >= scale:
            value = seconds / scale
            text = f"{value:.3g}"
            if "e" in text:  # very large numbers read better as powers of ten
                mantissa, exponent = text.split("e")
                text = rf"{mantissa}\times10^{{{int(exponent)}}}"
            return rf"${operator}{text}$ {unit}" if operator else f"${text}$ {unit}"
    return rf"${operator}{seconds:.2g}$ s"


def dodge(
    values: Sequence[float],
    min_gap: float,
    low: float | None = None,
    high: float | None = None,
) -> list[float]:
    """Nudge label positions apart while preserving their order.

    Level schemes routinely put two states a few keV apart -- 1607.97 and
    1609.80 keV in the 180Lu decay -- and their labels then overprint. This
    spreads a run of crowded labels just enough to separate them, keeping the
    block centred on where it started so the figure does not visibly drift.
    Callers draw a leader line wherever a label has moved.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)

    for previous, current in zip(order, order[1:]):
        if out[current] - out[previous] < min_gap:
            out[current] = out[previous] + min_gap

    drift = (sum(values) - sum(out)) / len(values)
    out = [value + drift for value in out]

    if low is not None and out[order[0]] < low:
        shift = low - out[order[0]]
        out = [value + shift for value in out]
    if high is not None and out[order[-1]] > high:
        shift = out[order[-1]] - high
        out = [value - shift for value in out]
    return out


def separate_labels(fig, ax, texts: Sequence, max_passes: int = 40) -> int:
    """Nudge overlapping labels apart, measuring in rendered pixels.

    Placing labels by rule -- dodge this column, stagger that one along its
    arrow -- gets most of the way there and then fails on the awkward cases:
    an unusually wide spin assignment, a decay with only two levels, a Q-value
    that lands under the parent's half-life. Rather than keep tuning offsets
    per figure, this measures what was actually drawn and moves the offenders.

    Only vertical moves are made, so a label stays over the feature it
    annotates, and the halo covers whatever it crosses on the way. Returns the
    number of pairs still overlapping, which should be zero.
    """
    if not texts:
        return 0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

    for _ in range(max_passes):
        boxes = [
            (t, t.get_window_extent(renderer)) for t in texts if t.get_text().strip()
        ]
        clashes = [
            (a, b)
            for i, (a, box_a) in enumerate(boxes)
            for b, box_b in boxes[i + 1 :]
            if box_a.overlaps(box_b)
        ]
        if not clashes:
            return 0
        for first, second in clashes:
            box_a = first.get_window_extent(renderer)
            box_b = second.get_window_extent(renderer)
            # move the upper one up and the lower one down, half the overlap each
            shift = (min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)) / 2 + 1.5
            upper, lower = (first, second) if box_a.y0 > box_b.y0 else (second, first)
            for text, direction in ((upper, +1), (lower, -1)):
                x, y = ax.transData.transform(text.get_position())
                moved = ax.transData.inverted().transform((x, y + direction * shift))
                text.set_position((float(moved[0]), float(moved[1])))
        fig.canvas.draw()

    boxes = [(t, t.get_window_extent(renderer)) for t in texts if t.get_text().strip()]
    return sum(
        1
        for i, (_, box_a) in enumerate(boxes)
        for _, box_b in boxes[i + 1 :]
        if box_a.overlaps(box_b)
    )
