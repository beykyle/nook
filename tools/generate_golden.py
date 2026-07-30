#!/usr/bin/env python3
"""Capture an independent parser's output as a frozen test oracle.

Run by hand to refresh the oracle. This is not part of the test suite, and
``nudel`` is deliberately **not** a dependency of this package -- the point is
that the committed values came from a codebase that does not share our
assumptions, and once captured that independence needs nothing installed.

    pip install git+https://github.com/op3/nudel
    python tools/generate_golden.py ~/ensdf/ensdf.180

Writes two files, and the distinction matters: ``oracle`` holds nudel's values,
so a disagreement means one of the two parsers is wrong; ``characterization``
holds ours, for quantities nudel does not compute, and only detects change.

Levels whose ``J`` field falls in a pattern where nudel is known to be wrong
are excluded and listed under ``known_divergences``. See ``docs/testing.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import nook as el  # noqa: E402

sys.path.insert(0, str(REPO / "tests"))
from _fingerprint import (  # noqa: E402
    absent,
    digest,
    divergence_reason,
    level_fingerprint,
)
from nook.sources.ensdf_file import (  # noqa: E402
    parse_ensdf_text,
    place_gammas,
)

PARITY = {"+": 1, "-": -1, "": None, None: None}


def nudel_jpi(level) -> list[list[int | None]]:
    return sorted(
        [int(Fraction(str(a.val)) * 2), PARITY.get(a.parity)]
        for a in level.ang_mom
        if a.val is not None
    )


def nudel_version() -> str:
    import nudel

    return getattr(nudel, "__version__", "unknown")


def file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_ensdf(folder: Path):
    """A nudel ENSDF reader over ``folder``, with its index cache cleared."""
    import nudel

    cache = Path.home() / ".cache" / "nudel" / "ensdf_index.pickle.xz"
    cache.unlink(missing_ok=True)
    return nudel.core.ENSDF(nudel.provider.ENSDFFileProvider(folder))


def capture_excerpt_oracle(excerpt: Path, workdir: Path) -> dict:
    """Per-level values from nudel for the committed excerpt.

    nudel indexes whole ``ensdf.NNN`` files and needs full 80-column lines,
    so the excerpt is staged into its own directory and padded first.
    """
    excerpt_dir = workdir / "excerpt"
    excerpt_dir.mkdir(exist_ok=True)
    staged = excerpt_dir / "ensdf.180"
    staged.write_text(
        "\n".join(line.ljust(80) for line in excerpt.read_text().splitlines()) + "\n\n"
    )
    dataset = _fresh_ensdf(excerpt_dir).get_adopted_levels((180, 73))

    levels, skipped = [], []
    for lv in dataset.levels:
        raw_j = lv.record[0].ljust(80)[21:39].strip()
        # normalise nudel's nan to null so the file is valid JSON and the
        # two "no value" spellings cannot be confused with a real number
        entry = {
            "energy": None if absent(lv.energy.val) else lv.energy.val,
            "energy_unc": None if absent(lv.energy.pm) else lv.energy.pm,
            "jpi_raw": raw_j,
            "jpi": nudel_jpi(lv),
            "xref": sorted(lv.xref),
            "metastable": bool(lv.metastable),
        }
        reason = divergence_reason(raw_j)
        if reason:
            skipped.append({**entry, "reason": reason})
        else:
            levels.append(entry)
    return {"levels": levels, "known_divergences": skipped}


def capture_chain_digests(chain: Path, workdir: Path) -> dict:
    """A per-dataset digest of nudel's values for a whole mass chain."""
    staged = workdir / chain.name
    staged.write_bytes(chain.read_bytes())
    ensdf = _fresh_ensdf(workdir)

    out, divergent = {}, 0
    for (nucleus, dsid), _offset in ensdf.provider.index.items():
        try:
            dataset = ensdf.get_dataset(nucleus, dsid)
        except Exception:
            continue
        prints = []
        for lv in dataset.levels:
            raw_j = lv.record[0].ljust(80)[21:39].strip()
            if divergence_reason(raw_j):
                divergent += 1
                continue
            prints.append(
                level_fingerprint(
                    lv.energy.val,
                    lv.energy.pm,
                    nudel_jpi(lv),
                    lv.xref,
                    bool(lv.metastable),
                )
            )
        key = f"{nucleus[0]}/{nucleus[1]}/{dsid}"
        out[key] = {"levels": len(prints), "digest": digest(prints)}
    return {"datasets": out, "levels_excluded_as_divergent": divergent}


def capture_characterization(excerpt: Path, decay_excerpt: Path) -> dict:
    """Our own values, for quantities nudel does not compute."""
    scheme = place_gammas(parse_ensdf_text(excerpt.read_text())[0].to_scheme())
    decay = parse_ensdf_text(decay_excerpt.read_text())[0].to_decay_scheme()
    return {
        "_warning": (
            "Produced by nook itself. These detect change, not "
            "correctness -- nudel does not compute these quantities."
        ),
        "bands": {
            "labels": sorted({lv.band for lv in scheme.levels if lv.band}),
            "definitions": scheme.metadata["band_definitions"],
        },
        "moments": [
            {
                "energy": lv.energy_kev,
                "mu": lv.magnetic_dipole.value if lv.magnetic_dipole else None,
                "q": lv.electric_quadrupole.value if lv.electric_quadrupole else None,
            }
            for lv in scheme.levels
            if lv.magnetic_dipole or lv.electric_quadrupole
        ],
        "q_record": {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in scheme.metadata["Q"].items()
        },
        "gammas": [
            {
                "energy": g.energy_kev,
                "start": g.start_index,
                "end": g.end_index,
                "mult": g.multipolarity,
            }
            for g in scheme.gammas
        ],
        "decay": {
            "dsid": decay.dsid,
            "parent": str(decay.parent.nuclide),
            "normalization": {
                "NR": decay.normalization.photon.value,
                "BR": decay.normalization.branching.value,
                "NB": decay.normalization.feeding.value,
            },
            "feedings": [
                {
                    "kind": f.kind,
                    "level": f.level_index,
                    "intensity": f.intensity.value,
                    "log_ft": f.log_ft.value if f.log_ft else None,
                }
                for f in decay.feedings
            ],
            "total_feeding": decay.total_feeding().value,
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    chain = Path(argv[1]).expanduser()
    if not chain.is_file():
        print(f"no such chain file: {chain}", file=sys.stderr)
        return 1

    data = REPO / "tests" / "data"
    # nudel resolves its data directory at import time and raises if there is
    # nothing there, so stage the chain and point it here before importing.
    workdir = Path("/tmp/ensdf-oracle-stage")
    workdir.mkdir(exist_ok=True)
    (workdir / chain.name).write_bytes(chain.read_bytes())
    os.environ["ENSDF_PATH"] = str(workdir)

    payload = {
        "_provenance": {
            "oracle": "nudel",
            "oracle_version": nudel_version(),
            "oracle_url": "https://github.com/op3/nudel",
            "oracle_licence": "GPL-3.0-or-later (used only to generate this "
            "file; not a dependency of nook)",
            "chain_file": chain.name,
            "chain_checksum": file_checksum(chain),
            "generated": date.today().isoformat(),
            "generated_by": "tools/generate_golden.py",
            "nook_version": el.__version__,
        },
        "excerpt": capture_excerpt_oracle(data / "ensdf180_excerpt.ensdf", workdir),
        "chain": capture_chain_digests(chain, workdir),
    }
    out = data / "ensdf180_oracle.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size} bytes)")
    print(f"  excerpt levels from nudel : {len(payload['excerpt']['levels'])}")
    print(
        f"  held back as divergent    : {len(payload['excerpt']['known_divergences'])}"
    )
    print(f"  chain datasets digested   : {len(payload['chain']['datasets'])}")
    print(
        f"  chain levels excluded     : {payload['chain']['levels_excluded_as_divergent']}"
    )

    char = capture_characterization(
        data / "ensdf180_excerpt.ensdf", data / "ensdf180_decay_excerpt.ensdf"
    )
    char_out = data / "ensdf180_characterization.json"
    char_out.write_text(json.dumps(char, indent=1, sort_keys=True))
    print(f"wrote {char_out.relative_to(REPO)} ({char_out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
