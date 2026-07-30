"""Mirror the RIPL-3 data segments (no codes) into ``data/ripl3/``.

RIPL-3 is a static library (2009 release; the discrete-levels segment was last
cut in 2021), which is why the mirror is committed rather than fetched at
runtime.  This tool exists to populate the mirror once, to verify it offline
afterwards, and to refresh a segment if the IAEA ever republishes one.

    uv run python tools/fetch_ripl3.py --record     # first population
    uv run python tools/fetch_ripl3.py --verify     # offline integrity check
    uv run python tools/fetch_ripl3.py --refresh levels

Downloads resume (HTTP Range) into ``data/ripl3/_archives/*.part``; archives
are extracted through a temp directory and never committed.  Every extracted
file's size and sha256 land in ``data/ripl3/MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www-nds.iaea.org/RIPL-3/"
# The IAEA server answers 403 to blank user agents, same as the Livechart API.
USER_AGENT = (
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0"
)
RELEASE = "RIPL-3 (2009); discrete levels cut 2021"

# (segment, url path relative to BASE_URL, kind, destination relative to the
# mirror root).  kind "file" downloads one file; "tgz"/"zip" download and
# extract into dest (a shared leading directory inside the archive is
# stripped).  A "file" whose destination already exists -- typically because a
# bundle archive provided it -- is hashed and recorded but not re-downloaded.
ITEMS: list[tuple[str, str, str, str]] = [
    # masses: experimental + FRDM95 + HFB14 masses, abundances, deformations
    ("masses", "masses/masses.tgz", "tgz", "masses/"),
    ("masses", "masses/mass-frdm95.readme", "file", "masses/mass-frdm95.readme"),
    ("masses", "masses/mass-hfb14.readme", "file", "masses/mass-hfb14.readme"),
    ("masses", "masses/abundance.readme", "file", "masses/abundance.readme"),
    # gs-deformations-exp.dat is linked from the RIPL-3 index page but has
    # been removed from the server (404 as of 2026-07); theoretical
    # deformations remain available inside mass-frdm95.dat.
    ("masses", "masses/duflo-zuker96.readme", "file", "masses/duflo-zuker96.readme"),
    # theoretical matter-density distributions (bulk data)
    ("masses", "masses/densities-hfb14.tgz", "tgz", "masses/matter-densities-hfb14/"),
    ("masses", "masses/density-hfb14.readme", "file", "masses/density-hfb14.readme"),
    ("masses", "masses/density-d1s.tgz", "tgz", "masses/matter-density-d1s/"),
    ("masses", "masses/density-d1s.readme", "file", "masses/density-d1s.readme"),
    # levels: the 2021 cut only
    ("levels", "levels/levels_2021.zip", "zip", "levels/"),
    ("levels", "levels/levels-param-2021.data", "file", "levels/levels-param-2021.data"),
    ("levels", "levels/levels-param-2021.readme", "file",
     "levels/levels-param-2021.readme"),
    ("levels", "levels/levels-readme-2020.html", "file", "levels/levels-readme-2020.html"),
    ("levels", "levels/levels-readme-2015.html", "file", "levels/levels-readme-2015.html"),
    # resonances: average resonance parameters
    ("resonances", "resonances/resonances0.dat", "file", "resonances/resonances0.dat"),
    ("resonances", "resonances/resonances1.dat", "file", "resonances/resonances1.dat"),
    ("resonances", "resonances/resonances.readme", "file", "resonances/resonances.readme"),
    # optical: the OMP archive and its indexes
    ("optical", "optical/om-parameter-u.dat", "file", "optical/om-parameter-u.dat"),
    ("optical", "optical/om-parameter-u.readme", "file", "optical/om-parameter-u.readme"),
    ("optical", "optical/om-deformations.dat", "file", "optical/om-deformations.dat"),
    ("optical", "optical/om-deformations.readme", "file",
     "optical/om-deformations.readme"),
    ("optical", "optical/optical.readme", "file", "optical/optical.readme"),
    ("optical", "optical/omp-index-ordered-by-Z.txt", "file",
     "optical/omp-index-ordered-by-Z.txt"),
    ("optical", "optical/omp-index-RIPL-ID.txt", "file", "optical/omp-index-RIPL-ID.txt"),
    ("optical", "optical/references-RIPL-ID.txt", "file", "optical/references-RIPL-ID.txt"),
    # densities: analytic parameters + HFB tables (total and partial)
    ("densities", "densities/densities.tgz", "tgz", "densities/"),
    # gamma: GDR parameters, strength functions (exp, analytic, microscopic)
    ("gamma", "gamma/gamma.tgz", "tgz", "gamma/"),
    ("gamma", "gamma/gamma-strength-micro.readme", "file",
     "gamma/gamma-strength-micro.readme"),
    # fission: empirical + HFB barriers, HFB level densities at saddle points
    ("fission", "fission/fission.tgz", "tgz", "fission/"),
    ("fission", "fission/hfb-barriers.tgz", "tgz", "fission/"),
    ("fission", "fission/hfb-levden.tgz", "tgz", "fission/"),
    ("fission", "fission/empirical-barriers.readme", "file",
     "fission/empirical-barriers.readme"),
    ("fission", "fission/empirical-hfb-barriers.readme", "file",
     "fission/empirical-hfb-barriers.readme"),
    ("fission", "fission/hfb-levden.readme", "file", "fission/hfb-levden.readme"),
]

SEGMENTS = tuple(dict.fromkeys(seg for seg, *_ in ITEMS))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dest() -> Path:
    """``$RIPL_PATH`` when set, else the repository mirror.

    Matches ``nook.sources.ripl3.default_ripl_path`` so that following an
    error message's advice to run this tool populates the directory the
    backend will actually read.
    """
    env = os.environ.get("RIPL_PATH")
    if env:
        return Path(env).expanduser()
    return repo_root() / "data" / "ripl3"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url_path: str, target: Path) -> None:
    """Fetch ``BASE_URL + url_path`` to ``target``, resuming a partial file."""
    url = BASE_URL + urllib.parse.quote(url_path, safe="/")
    part = target.with_name(target.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            mode = "ab" if offset and response.status == 206 else "wb"
            with part.open(mode) as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            # Range beyond the end: the previous run already downloaded the
            # whole file; finish the rename instead of throwing the data away.
            part.replace(target)
            return
        # Keep the .part so a later run can resume; a 404's empty part is
        # harmless and a transient 5xx should not cost the progress so far.
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    part.replace(target)


def extract(archive: Path, dest: Path) -> None:
    """Extract into ``dest``, stripping a shared leading directory if present."""
    with tempfile.TemporaryDirectory(dir=archive.parent) as tmp:
        tmp_path = Path(tmp)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp_path)
        else:
            with tarfile.open(archive) as tf:
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(tmp_path, filter="data")
                else:  # pragma: no cover - 3.10/3.11
                    for member in tf.getmembers():
                        if member.name.startswith(("/", "..")) or ".." in member.name:
                            raise RuntimeError(f"unsafe path in {archive}: {member.name}")
                    tf.extractall(tmp_path)
        entries = list(tmp_path.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_path
        dest.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                    continue
                target.unlink()
            shutil.move(str(item), target)


def populate(dest: Path, segments: set[str], refresh: bool) -> list[str]:
    archives = dest / "_archives"
    failures: list[str] = []
    for segment, url_path, kind, rel in ITEMS:
        if segment not in segments:
            continue
        if kind == "file":
            target = dest / rel
            if target.exists() and not refresh:
                continue
            try:
                download(url_path, target)
                print(f"  fetched {rel}")
            except (RuntimeError, OSError) as exc:
                # OSError covers URLError/timeouts: a network blip should
                # fail this item, not abort the run before the manifest.
                failures.append(f"{url_path}: {exc}")
            continue
        archive = archives / Path(url_path).name
        marker = archives / (Path(url_path).name + ".extracted")
        if marker.exists() and not refresh:
            continue
        try:
            if not archive.exists() or refresh:
                print(f"  downloading {url_path} ...")
                download(url_path, archive)
            print(f"  extracting {archive.name} -> {rel}")
            extract(archive, dest / rel)
            marker.write_text(sha256_of(archive) + "\n")
        except (RuntimeError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            failures.append(f"{url_path}: {exc}")
    return failures


def record_manifest(dest: Path) -> dict:
    # Rehashing the whole ~1.2 GB mirror on every run is pointless when most
    # files are untouched, so reuse the previous manifest's sha256 whenever
    # size and mtime_ns both match (the same fingerprint the parse cache
    # trusts).  --verify still rehashes everything.
    previous: dict = {}
    manifest_file = dest / "MANIFEST.json"
    if manifest_file.is_file():
        try:
            previous = json.loads(manifest_file.read_text()).get("extracted", {})
        except (json.JSONDecodeError, OSError):
            previous = {}
    manifest = {
        "base_url": BASE_URL,
        "release": RELEASE,
        "items": [
            {"segment": seg, "url": url, "kind": kind, "dest": rel}
            for seg, url, kind, rel in ITEMS
        ],
        "extracted": {},
    }
    for path in sorted(dest.rglob("*")):
        if not path.is_file() or "_archives" in path.parts:
            continue
        rel = path.relative_to(dest).as_posix()
        if rel == "MANIFEST.json":
            continue
        stat = path.stat()
        known = previous.get(rel)
        if known and known["size"] == stat.st_size and known.get("mtime_ns") == stat.st_mtime_ns:
            sha256 = known["sha256"]
        else:
            sha256 = sha256_of(path)
        manifest["extracted"][rel] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256,
        }
    manifest_file.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def verify(dest: Path) -> int:
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    bad = 0
    for rel, want in manifest["extracted"].items():
        path = dest / rel
        if not path.is_file():
            print(f"MISSING  {rel}")
            bad += 1
        elif path.stat().st_size != want["size"] or sha256_of(path) != want["sha256"]:
            print(f"CHANGED  {rel}")
            bad += 1
    print(f"{len(manifest['extracted'])} files checked, {bad} problems")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mirror the RIPL-3 data segments into data/ripl3/."
    )
    parser.add_argument(
        "--dest", type=Path, default=default_dest(),
        help="mirror directory (default: $RIPL_PATH, else <repo>/data/ripl3)",
    )
    parser.add_argument("--only", choices=SEGMENTS, help="restrict to one segment")
    parser.add_argument("--record", action="store_true",
                        help="download, extract, and (re)write the manifest")
    parser.add_argument("--verify", action="store_true",
                        help="offline: check every manifest entry's size and sha256")
    parser.add_argument("--refresh", choices=SEGMENTS,
                        help="re-download one segment, then rewrite the manifest")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.dest)

    segments = {args.refresh} if args.refresh else (
        {args.only} if args.only else set(SEGMENTS)
    )
    if not (args.record or args.refresh):
        parser.error("choose one of --record, --verify, --refresh")

    args.dest.mkdir(parents=True, exist_ok=True)
    failures = populate(args.dest, segments, refresh=bool(args.refresh))
    manifest = record_manifest(args.dest)
    total = sum(entry["size"] for entry in manifest["extracted"].values())
    print(f"mirror: {len(manifest['extracted'])} files, {total / 1e6:.0f} MB")
    for failure in failures:
        print(f"FAILED  {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
