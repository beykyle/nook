"""Fixed-width parsing helpers shared by the RIPL-3 segment parsers.

RIPL files are Fortran fixed-format.  Fields are sliced by column, never
split on whitespace: several formats butt columns together (a negative
parity against a spin, a chi-squared against a flag) and whitespace
splitting silently merges or shifts them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, TypeVar

MEV_TO_KEV = 1000.0

_T = TypeVar("_T")

#: ``Z= 82 A=170`` -- the space after ``=`` vanishes at three digits, so a
#: block header can never be split on whitespace.
Z_A_HEADER = re.compile(r"Z=\s*(\d+)\s+A=\s*(\d+)")


def z_blocks(
    text: str, is_header: Callable[[str], bool]
) -> Iterator[tuple[int, int, str, list[str]]]:
    """Split a bulk per-element file into its ``Z=... A=...`` blocks.

    Yields ``(z, a, header_line, body_lines)`` per block; a header line that
    ``is_header`` accepts but that carries no Z/A raises, since silently
    skipping it would attach its rows to the previous nuclide.
    """
    header: tuple[int, int, str] | None = None
    body: list[str] = []
    for line in text.splitlines():
        if is_header(line):
            if header is not None:
                yield (*header, body)
            match = Z_A_HEADER.search(line)
            if match is None:
                raise ValueError(f"unrecognised block header: {line.strip()!r}")
            header = (int(match.group(1)), int(match.group(2)), line)
            body = []
        elif header is not None:
            body.append(line)
    if header is not None:
        yield (*header, body)


def field(line: str, start: int, stop: int) -> str:
    """The stripped text in columns ``[start, stop)``; safe past line end."""
    return line[start:stop].strip()


def is_number(token: str) -> bool:
    """True when ``float`` accepts the token (shared by ragged-row parsers)."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def data_lines(text: str):
    """The lines that carry data: blank lines and ``#`` comment rows skipped."""
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            yield line


def float_field(line: str, start: int, stop: int) -> float | None:
    text = field(line, start, stop)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def int_field(line: str, start: int, stop: int) -> int | None:
    text = field(line, start, stop)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


@lru_cache(maxsize=16)
def _read_cached(path: str, fingerprint: tuple[int, int]) -> str:
    del fingerprint  # part of the cache key only
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_file(path: Path) -> str:
    """Read a data file, reusing the text while size and mtime are unchanged.

    Same idea as the ENSDF chain cache: a per-element RIPL file is parsed on
    every nuclide lookup otherwise.
    """
    stat = path.stat()
    return _read_cached(str(path.resolve()), (stat.st_size, stat.st_mtime_ns))


@lru_cache(maxsize=32)
def _parse_cached(parser, path: str, fingerprint: tuple[int, int], args: tuple):
    return parser(_read_cached(path, fingerprint), *args)


def parse_file(parser: Callable[..., _T], path: Path, *args) -> _T:
    """``parser(text, *args)`` for a data file, cached while size and mtime hold.

    Every segment loader funnels its whole-file parse through here so repeat
    lookups are dictionary hits, and so :func:`clear_ripl_cache` covers every
    cache there is.  ``parser`` must be a module-level function and ``args``
    hashable (both are part of the cache key).
    """
    stat = path.stat()
    return _parse_cached(parser, str(path.resolve()), (stat.st_size, stat.st_mtime_ns), args)


def require_file(file: Path, what: str) -> None:
    """Raise the standard remedy-bearing error when a mirror file is absent."""
    if not file.is_file():
        raise FileNotFoundError(
            f"no {what} at {file}. Populate the mirror with "
            f"`python tools/fetch_ripl3.py --record` or point $RIPL_PATH at one."
        )


def clear_ripl_cache() -> None:
    """Forget every cached RIPL file, raw text and parsed tables alike."""
    _read_cached.cache_clear()
    _parse_cached.cache_clear()
