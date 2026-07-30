"""A tiny on-disk cache for remote responses.

Reproducibility matters more than cleverness here: a calculation that pulled
levels last March should be able to pull the identical bytes today, and a
cached tree can be committed alongside a paper's input decks.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["Cache", "default_cache_dir"]


def default_cache_dir() -> Path:
    """``$NOOK_CACHE``, else the XDG cache directory."""
    env = os.environ.get("NOOK_CACHE")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "nook"


class Cache:
    """Content-addressed key/value store over plain files."""

    def __init__(self, directory: str | os.PathLike[str] | None = None, enabled: bool = True):
        self.directory = Path(directory) if directory is not None else default_cache_dir()
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / digest[:2] / f"{digest}.txt"

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)

    def clear(self) -> int:
        """Delete every cached entry; returns the number of files removed."""
        if not self.directory.is_dir():
            return 0
        n = 0
        for path in self.directory.rglob("*.txt"):
            path.unlink()
            n += 1
        return n
