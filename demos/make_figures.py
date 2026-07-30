#!/usr/bin/env python3
"""Generate the figure gallery.

    python demos/make_figures.py                    # RIPL-3 figures only
    python demos/make_figures.py ~/ensdf            # everything
    python demos/make_figures.py ~/ensdf docs/figures

The RIPL-3 figures draw from the committed mirror and need nothing beyond
matplotlib.  The ENSDF figures need a local distribution (mass-chain files);
when it is absent they are skipped with a note.  The chart of nuclides walks
every chain once, which takes a minute or two; the result is cached beside
the data as ``ground-state-survey.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import nook as el  # noqa: E402
from nook.plotting import (  # noqa: E402
    plot_band_scheme,
    plot_chart,
    plot_chart_panels,
    plot_decay_scheme,
    plot_deformation_chart,
    plot_fission_barriers,
    plot_gdr,
    plot_gsf,
    plot_level_comparison,
    plot_level_density,
    plot_level_scheme,
    plot_mass_residuals,
    plot_matter_density,
    plot_resonance_charts,
)
from nook.survey import survey  # noqa: E402


def save(fig, out: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}")
    print(f"  {name}.png / .pdf")
    fig.clf()


def ensdf_figures(data: Path, out: Path) -> None:
    # 1. level scheme -- 180Ta, whose 9- isomer is the longest-lived known
    scheme = el.level_scheme("180Ta", source="file", path=data)
    save(
        plot_level_scheme(
            scheme, limit=12,
            title=r"$^{180}$Ta  adopted levels",
        )[0],
        out, "level-scheme-180Ta",
    )

    # 2. band scheme -- the Gallagher-Moszkowski structure, K=1+/8+ and 0-/9-
    save(
        plot_band_scheme(
            scheme, max_bands=4, max_energy=3000,
            title=r"$^{180}$Ta  rotational bands",
        )[0],
        out, "band-scheme-180Ta",
    )

    # 3. decay scheme -- 180Lu beta-minus, which has an N record so the
    #    intensities can be put on an absolute scale
    decay = next(
        d for d in el.decay_schemes("180Hf", path=data) if "180LU" in d.dsid
    )
    save(
        plot_decay_scheme(decay, title=r"$^{180}$Lu  $\beta^-$ decay")[0],
        out, "decay-scheme-180Lu",
    )

    # 4-6. the chart of nuclides, three ways
    print("  surveying ground states (cached after the first run) ...")
    states = survey(path=data)
    save(
        plot_chart(states, colour_by="decay",
                   title=f"chart of nuclides  ({len(states)} nuclides, ENSDF)")[0],
        out, "chart-decay-modes",
    )
    save(
        plot_chart(states, colour_by="half_life_s",
                   title="ground-state half-life")[0],
        out, "chart-half-life",
    )
    # S(n) from both backends on one scale: RIPL's value comes from its own
    # mass table, so agreement here is a cross-backend consistency check
    panels: list[tuple[str, list]] = [("ensdf", list(states))]
    if (el.default_ripl_path() / "levels").is_dir():
        from types import SimpleNamespace

        print("  reading RIPL S(n) from every levels file (cached after) ...")
        ripl_sn = [
            SimpleNamespace(nuclide=el.Nuclide(z, a), z=z, n=a - z,
                            stable=False, dominant_decay=None, s_n=sn)
            for (z, a), sn in el.Ripl3Source().sn_table().items()
        ]
        panels.append(("ripl3 levels", ripl_sn))
    save(
        plot_chart_panels(panels, colour_by="s_n",
                          title="neutron separation energy")[0],
        out, "chart-separation-energy",
    )

    # 7. the same quantity after repair, with what still looks wrong ringed
    from nook.repair import repair
    from nook.survey import inconsistencies

    fixed, changes = repair(states)
    suspect = sorted({state.nuclide for state, _ in inconsistencies(fixed)},
                     key=lambda n: (n.z, n.a))
    print(f"  {len(changes)} repairs; {len(suspect)} nuclides still flagged")
    save(
        plot_chart(
            fixed, colour_by="s_n", log=False, mark=suspect,
            mark_label="still inconsistent",
            title=(f"neutron separation energy after {len(changes)} verified "
                   "repairs"),
        )[0],
        out, "chart-separation-energy-repaired",
    )

    # 8. the comparison nook.compare exists to make -- needs both sources.
    #    24Mg keeps both completeness cutoffs on one readable figure
    #    (heuristic at level 7, RIPL's Nmax at level 28).
    comparison = el.compare.levels("24Mg", sources=("file", "ripl3"),
                                   below=10000.0, path=data)
    save(
        plot_level_comparison(
            comparison, title=r"$^{24}$Mg  ENSDF vs RIPL-3",
        )[0],
        out, "level-comparison-24Mg",
    )


def ripl_figures(out: Path) -> None:
    src = el.Ripl3Source()

    # 1. GDR -- 181Ta, the textbook deformed nucleus with a split resonance
    save(
        plot_gdr(src.gdr("181Ta"),
                 title=r"$^{181}$Ta  giant dipole resonance")[0],
        out, "gdr-181Ta",
    )

    # 2. E1 strength function, with S_n marking the capture window
    save(
        plot_gsf(src.gsf("56Fe"), sn_kev=src.levels("56Fe").sn_kev,
                 title=r"$^{56}$Fe  E1 strength function")[0],
        out, "gamma-strength-56Fe",
    )

    # 3. level density and cumulative count -- the n+56Fe compound system
    save(
        plot_level_density(
            src.hfb_density("57Fe"), scheme=src.fetch("57Fe"),
            ct=src.levels_param()[(26, 57)],
            title=r"$^{57}$Fe  level density  (n+$^{56}$Fe compound)",
        )[0],
        out, "level-density-57Fe",
    )

    # 4. matter density -- 208Pb and its neutron skin
    save(
        plot_matter_density(src.matter_density("208Pb"),
                            title=r"$^{208}$Pb  matter density")[0],
        out, "matter-density-208Pb",
    )

    # 5. fission barriers -- 238U, empirical with the HFB overlay
    save(
        plot_fission_barriers(
            src.fission_barriers("238U"),
            overlay=src.fission_barriers("238U", model="hfb"),
            title=r"$^{238}$U  fission barriers",
        )[0],
        out, "fission-barriers-238U",
    )

    # 6-7. the bulk mass table, two ways -- FRDM95 and HFB-14 side by side
    print("  reading the mass tables ...")
    table = src.mass_table()
    save(
        plot_mass_residuals(table)[0],
        out, "chart-mass-residuals",
    )
    save(
        plot_deformation_chart(
            table, title="FRDM95 vs HFB-14 ground-state deformation")[0],
        out, "chart-deformation",
    )

    # 8. resonance systematics, three panels
    save(
        plot_resonance_charts(src.resonance_table(),
                              title="s-wave resonance systematics")[0],
        out, "chart-resonances",
    )


def main(argv: list[str]) -> int:
    data = Path(argv[1]).expanduser() if len(argv) > 1 else el.default_ensdf_path()
    out = Path(argv[2]).expanduser() if len(argv) > 2 else REPO / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    mirror = el.default_ripl_path()
    if (mirror / "levels").is_dir():
        ripl_figures(out)
    else:
        print("skipping RIPL-3 figures (populate the mirror with "
              "tools/fetch_ripl3.py or set $RIPL_PATH)")

    if data.is_dir():
        ensdf_figures(data, out)
    else:
        print(f"skipping ENSDF figures (no data at {data}; pass an ENSDF "
              "directory as the first argument)")

    print(f"\nwrote {len(list(out.glob('*.png')))} figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
