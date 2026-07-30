"""tools/fetch_ripl3.py: resume behaviour that only shows up against a server."""

from __future__ import annotations

import importlib.util
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_ripl3", Path(__file__).parent.parent / "tools" / "fetch_ripl3.py"
)
assert _SPEC is not None and _SPEC.loader is not None
fetch_ripl3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_ripl3)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", hdrs=Message(), fp=None)


def test_416_on_resume_means_the_part_is_complete(tmp_path, monkeypatch):
    # A finished .part left by a crashed run answers 416 to the Range request;
    # the data must be kept and promoted, not deleted and re-downloaded.
    target = tmp_path / "resonances0.dat"
    part = tmp_path / "resonances0.dat.part"
    part.write_bytes(b"complete payload")

    def refuse_range(request, timeout):
        assert request.headers.get("Range") == "bytes=16-"
        raise _http_error(416)

    monkeypatch.setattr(urllib.request, "urlopen", refuse_range)
    fetch_ripl3.download("resonances/resonances0.dat", target)
    assert target.read_bytes() == b"complete payload"
    assert not part.exists()


def test_transient_http_error_keeps_the_partial_download(tmp_path, monkeypatch):
    target = tmp_path / "big.tgz"
    part = tmp_path / "big.tgz.part"
    part.write_bytes(b"half of it")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(_http_error(503)),
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        fetch_ripl3.download("masses/masses.tgz", target)
    assert part.read_bytes() == b"half of it"  # resume is still possible
    assert not target.exists()


def test_default_dest_honours_ripl_path(monkeypatch, tmp_path):
    monkeypatch.setenv("RIPL_PATH", str(tmp_path / "mirror"))
    assert fetch_ripl3.default_dest() == tmp_path / "mirror"
    monkeypatch.delenv("RIPL_PATH")
    assert fetch_ripl3.default_dest() == fetch_ripl3.repo_root() / "data" / "ripl3"


def test_default_dest_matches_the_backend(monkeypatch, tmp_path):
    # The tool promises to populate the directory the backend reads; the two
    # implementations are separate (the tool must run pre-install), so pin
    # them together here.
    from nook.sources.ripl3 import default_ripl_path

    monkeypatch.setenv("RIPL_PATH", str(tmp_path / "elsewhere"))
    assert fetch_ripl3.default_dest() == default_ripl_path()
    monkeypatch.delenv("RIPL_PATH")
    assert fetch_ripl3.default_dest() == default_ripl_path()


def test_record_manifest_reuses_hashes_for_unchanged_files(tmp_path, monkeypatch):
    dest = tmp_path / "mirror"
    (dest / "levels").mkdir(parents=True)
    (dest / "levels" / "z012.dat").write_text("data\n")
    first = fetch_ripl3.record_manifest(dest)

    hashed: list[str] = []
    original = fetch_ripl3.sha256_of

    def counting_sha256(path):
        hashed.append(path.name)
        return original(path)

    monkeypatch.setattr(fetch_ripl3, "sha256_of", counting_sha256)
    second = fetch_ripl3.record_manifest(dest)
    assert second["extracted"] == first["extracted"]
    assert hashed == []  # size+mtime matched: nothing rehashed
