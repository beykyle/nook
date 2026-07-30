"""Publication-quality figures for level, band and decay schemes.

Needs matplotlib: ``pip install 'nook[plot]'``. Nothing here is
imported by the core package, so the zero-dependency install is unaffected.

Every function returns the matplotlib ``(figure, axes)`` so you can keep
adjusting before saving.
"""

from .chart import plot_chart
from .decay import plot_decay_scheme
from .levels import plot_band_scheme, plot_level_scheme
from .ripl import (
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
from .style import PALETTE, Palette, figure_style, half_life_label, jpi_label

__all__ = [
    "PALETTE",
    "Palette",
    "figure_style",
    "half_life_label",
    "jpi_label",
    "plot_band_scheme",
    "plot_chart",
    "plot_chart_panels",
    "plot_decay_scheme",
    "plot_deformation_chart",
    "plot_fission_barriers",
    "plot_gdr",
    "plot_gsf",
    "plot_level_comparison",
    "plot_level_density",
    "plot_level_scheme",
    "plot_mass_residuals",
    "plot_matter_density",
    "plot_resonance_charts",
]
