"""Command line front end: ``nook levels 24Mg --below 10000``.

Subcommand-based; a bare nuclide still works (``nook 24Mg``) because the
argument list is rewritten to ``levels 24Mg`` when the first token is not a
known subcommand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from . import __version__, level_scheme
from .sources.livechart import LivechartError

_SUBCOMMANDS = ("levels", "ripl", "compare")


def _dump_json(payload) -> None:
    """``json.dump`` to stdout with non-finite floats mapped to ``None``.

    ``half_life_s`` is ``inf`` for stable levels; the default encoder would
    emit the bare token ``Infinity``, which is not JSON (RFC 8259) and breaks
    ``jq`` and strict parsers.  The ``stable`` field keeps the information.
    """

    def clean(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    json.dump(clean(payload), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _levels_parser(sub) -> None:
    p = sub.add_parser("levels", help="print a level scheme")
    p.add_argument("nuclide", help='e.g. "24Mg", "Mg-24"')
    p.add_argument(
        "--source",
        choices=("livechart", "file", "ripl3"),
        default="livechart",
        help="IAEA API (adopted levels), local NNDC flat files, or the RIPL-3 mirror",
    )
    p.add_argument("--dataset", help="DSID substring; only with --source file")
    p.add_argument("--path", help="directory holding the source's data files")
    p.add_argument("--below", type=float, metavar="KEV", help="energy cutoff")
    p.add_argument(
        "--complete",
        action="store_true",
        help="truncate at the first level without a firm E, J and parity",
    )
    p.add_argument(
        "--nmax",
        action="store_true",
        help="truncate at RIPL's completeness cutoff; only with --source ripl3",
    )
    p.add_argument("--json", action="store_true", help="emit records as JSON")


def _run_levels(args) -> int:
    try:
        scheme = level_scheme(
            args.nuclide,
            source=args.source,
            dataset=args.dataset,
            path=args.path,
        )
        if args.nmax:
            nmax = scheme.metadata.get("nmax")
            if nmax is None:
                raise ValueError(
                    "--nmax needs a source that publishes a completeness "
                    "cutoff (--source ripl3)"
                )
            scheme = scheme[:nmax]
    except (LivechartError, LookupError, FileNotFoundError, ValueError) as exc:
        print(f"nook: {exc}", file=sys.stderr)
        return 1

    if args.below is not None:
        scheme = scheme.below(args.below)
    if args.complete:
        scheme = scheme.complete_up_to()

    if args.json:
        _dump_json(scheme.to_records())
    else:
        print(f"# {scheme.nuclide}  {scheme.dataset}  [{scheme.source}]")
        for lv in scheme:
            print(f"{lv.index:>4}  {lv}")
    return 0


def _print_masses(entry) -> None:
    print(f"# {entry.nuclide}  RIPL-3 masses  [ripl3-local]")
    for name in (
        "mass_excess_exp",
        "mass_excess_frdm95",
        "mass_excess_hfb14",
        "emic_frdm95",
        "beta2_frdm95",
        "beta2_hfb14",
        "abundance",
    ):
        value = getattr(entry, name)
        if value is None:
            continue
        unit = entry.units(name) or ""
        print(f"  {name:<22} {value!s:<24} {unit}")


def _print_resonances(entry) -> None:
    print(f"# {entry.nuclide} target  {'sp'[entry.wave]}-wave  [ripl3-local]")
    print(f"  D{entry.wave:<21} {entry.spacing_kev!s:<24} keV")
    print(f"  S{entry.wave:<21} {entry.strength_1e4!s:<24} 1e-4")
    print(f"  {'gamma width':<22} {entry.gamma_width_mev_milli!s:<24} meV")
    if entry.estimated:
        print("  (D estimated from the other wave)")


def _print_gdr(fits) -> None:
    for fit in fits:
        span = f"  fit {fit.fit_range_mev} MeV" if fit.fit_range_mev else ""
        print(f"# {fit.nuclide}  GDR {fit.kind}{span}  {fit.reference}")
        for e, w, cs in fit.peaks:
            sigma = "" if cs is None else f"  sigma {cs} mb"
            print(f"  E {e} MeV  Gamma {w} MeV{sigma}")


def _print_fission(entry) -> None:
    print(f"# {entry.nuclide}  fission barriers ({entry.model})  [ripl3-local]")
    for i, barrier in enumerate(entry.barriers):
        sym = f"  {barrier.symmetry}" if barrier.symmetry else ""
        print(
            f"  barrier {i + 1}: {barrier.height_mev} MeV  hw {barrier.hw_mev} MeV{sym}"
        )


#: segment -> (help text, extra argparse options, fetch, plain-text printer).
#: One row per ``nook ripl <segment>``; the parser, dispatch and printing all
#: derive from this table, and JSON output is handled generically.
_RIPL_SEGMENTS = {
    "masses": (
        "mass excesses, deformations, abundance",
        (),
        lambda source, args: source.masses(args.nuclide),
        _print_masses,
    ),
    "resonances": (
        "average neutron resonance parameters",
        (("--wave", dict(type=int, choices=(0, 1), default=0)),),
        lambda source, args: source.resonances(args.nuclide, wave=args.wave),
        _print_resonances,
    ),
    "gdr": (
        "giant dipole resonance parameters",
        (),
        lambda source, args: source.gdr(args.nuclide),
        _print_gdr,
    ),
    "fission": (
        "fission barriers",
        (("--model", dict(choices=("empirical", "hfb"), default="empirical")),),
        lambda source, args: source.fission_barriers(args.nuclide, model=args.model),
        _print_fission,
    ),
}


def _ripl_parser(sub) -> None:
    p = sub.add_parser("ripl", help="query non-level RIPL-3 segments")
    what = p.add_subparsers(dest="what", required=True)
    for name, (help_text, extra_args, _fetch, _printer) in _RIPL_SEGMENTS.items():
        q = what.add_parser(name, help=help_text)
        q.add_argument("nuclide")
        q.add_argument("--path", help="RIPL-3 mirror directory")
        q.add_argument("--json", action="store_true")
        for flag, options in extra_args:
            q.add_argument(flag, **options)


def _run_ripl(args) -> int:
    from dataclasses import asdict

    from .sources.ripl3 import Ripl3Source

    _help, _extra, fetch, printer = _RIPL_SEGMENTS[args.what]
    try:
        entry = fetch(Ripl3Source(path=args.path), args)
    except (LookupError, FileNotFoundError, ValueError) as exc:
        print(f"nook: {exc}", file=sys.stderr)
        return 1
    if args.json:
        # gdr returns a tuple of entries; everything else a single dataclass
        _dump_json(
            [asdict(e) for e in entry] if isinstance(entry, tuple) else asdict(entry)
        )
    else:
        printer(entry)
    return 0


def _compare_parser(sub) -> None:
    p = sub.add_parser("compare", help="compare a quantity across sources")
    p.add_argument("nuclide")
    p.add_argument("--what", choices=("levels", "masses"), default="levels")
    p.add_argument(
        "--sources",
        default="file,ripl3",
        help="two comma-separated level sources (default file,ripl3)",
    )
    p.add_argument("--below", type=float, metavar="KEV")
    p.add_argument("--tolerance", type=float, default=2.0, metavar="KEV")
    p.add_argument("--path", help="ENSDF flat-file directory")
    p.add_argument("--ripl-path", help="RIPL-3 mirror directory")
    p.add_argument(
        "--no-livechart", action="store_true", help="masses: skip the network entry"
    )
    p.add_argument("--json", action="store_true")


def _run_compare(args) -> int:
    from . import compare
    from .sources.livechart import LivechartError

    try:
        if args.what == "levels":
            sources = tuple(s.strip() for s in args.sources.split(","))
            if len(sources) != 2:
                raise ValueError("--sources needs exactly two names")
            result = compare.levels(
                args.nuclide,
                sources=sources,
                below=args.below,
                tolerance_kev=args.tolerance,
                path=args.path,
                ripl_path=args.ripl_path,
            )
            if args.json:
                _dump_json(result.to_records())
            else:
                print(result)
                for m in result.matched:
                    if m.significant:
                        print(
                            f"  != {m.a.energy_kev:10.2f} vs {m.b.energy_kev:10.2f} keV"
                            f"  ({m.a.spin_parity} / {m.b.spin_parity})"
                        )
        else:
            result = compare.masses(
                args.nuclide,
                ripl_path=args.ripl_path,
                with_livechart=not args.no_livechart,
            )
            if args.json:
                _dump_json({k: str(v) for k, v in result.entries.items()})
            else:
                print(result)
    except (LivechartError, LookupError, FileNotFoundError, ValueError) as exc:
        print(f"nook: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nook", description="Nuclear structure data from ENSDF and RIPL-3."
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    _levels_parser(sub)
    _ripl_parser(sub)
    _compare_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `nook 24Mg` == `nook levels 24Mg`
    if (
        argv
        and argv[0] not in _SUBCOMMANDS
        and argv[0] not in ("-h", "--help", "--version")
    ):
        argv.insert(0, "levels")
    args = build_parser().parse_args(argv)
    if args.command == "levels":
        return _run_levels(args)
    if args.command == "ripl":
        return _run_ripl(args)
    if args.command == "compare":
        return _run_compare(args)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
