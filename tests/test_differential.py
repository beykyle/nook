"""Validation against a frozen independent oracle.

The values in ``tests/data/ensdf180_oracle.json`` came from `nudel`_, an
independently written ENSDF parser, and were committed. Nothing here imports
nudel and it is not a dependency: the independence was captured once by
``tools/generate_golden.py`` and now travels with the repository.

A snapshot of our *own* output would detect change but not wrongness -- it
would preserve any bug present when it was taken. These values come from a
codebase that does not share our assumptions, so a disagreement means one of
the two is actually wrong.

Two things a frozen oracle cannot do:

* where nudel is itself wrong, freezing its answer would enshrine an error, so
  those levels are held out (``_fingerprint.DIVERGENT_J``) and our behaviour is
  pinned by spec-derived tests in ``test_nook.py`` instead;
* where nudel computes nothing -- bands, moments, decay normalisation -- there
  is no oracle, so those live in ``ensdf180_characterization.json`` and are
  labelled as characterization tests.

See ``docs/testing.md``.

.. _nudel: https://github.com/op3/nudel
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import nook as el  # noqa: F401  (kept for parity with other modules)
from _fingerprint import digest, divergence_reason, level_fingerprint
from nook.sources.ensdf_file import parse_ensdf_text, place_gammas

DATA = Path(__file__).parent / "data"
ORACLE = json.loads((DATA / "ensdf180_oracle.json").read_text())
CHARACTERIZATION = json.loads((DATA / "ensdf180_characterization.json").read_text())

CHAIN_DIR = Path(os.environ.get("ENSDF_PATH", "/nonexistent"))
HAS_CHAIN = (CHAIN_DIR / "ensdf.180").is_file()

needs_chain = pytest.mark.skipif(
    not HAS_CHAIN, reason="set ENSDF_PATH to a directory holding ensdf.180"
)


@pytest.fixture(scope="module")
def adopted():
    return place_gammas(
        parse_ensdf_text((DATA / "ensdf180_excerpt.ensdf").read_text())[0].to_scheme()
    )


@pytest.fixture(scope="module")
def decay():
    text = (DATA / "ensdf180_decay_excerpt.ensdf").read_text()
    return parse_ensdf_text(text)[0].to_decay_scheme()


# --------------------------------------------------------------------------
# validation against the frozen oracle -- always runs
# --------------------------------------------------------------------------


def test_oracle_records_its_own_provenance():
    """A frozen oracle is only trustworthy if you can tell where it came from."""
    provenance = ORACLE["_provenance"]
    assert provenance["oracle"] == "nudel"
    assert provenance["oracle_version"]
    assert provenance["chain_checksum"].startswith("sha256:")
    assert "not a dependency" in provenance["oracle_licence"]


def test_level_count_matches_the_oracle(adopted):
    expected = len(ORACLE["excerpt"]["levels"]) + len(
        ORACLE["excerpt"]["known_divergences"]
    )
    assert len(adopted.levels) == expected


@pytest.mark.parametrize("i", range(len(ORACLE["excerpt"]["levels"])))
def test_level_agrees_with_the_oracle(adopted, i):
    """Energy, uncertainty, Jpi, XREF and metastable, against nudel's values."""
    want = ORACLE["excerpt"]["levels"][i]
    got = adopted.levels[i]
    assert got.energy_kev == pytest.approx(want["energy"])
    if want["energy_unc"] is not None:
        assert got.energy.symmetric == pytest.approx(want["energy_unc"], rel=1e-6)
    assert got.spin_parity.raw == want["jpi_raw"]
    assert (
        sorted([c.two_j, c.parity] for c in got.spin_parity.candidates) == want["jpi"]
    )
    assert sorted(got.xref) == want["xref"]
    assert bool(got.metastable) == want["metastable"]


def test_no_level_silently_bypassed_the_divergence_filter():
    """Every level in the oracle really is one where nudel can be trusted."""
    for want in ORACLE["excerpt"]["levels"]:
        assert divergence_reason(want["jpi_raw"]) is None


@needs_chain
def test_whole_chain_digests_match_the_oracle():
    """Re-derive fingerprints for the full mass chain and compare to nudel's."""
    from nook.sources.ensdf_file import _parse_chain

    mine = {}
    for dataset in _parse_chain(CHAIN_DIR / "ensdf.180"):
        prints = [
            level_fingerprint(
                lv.energy_kev,
                lv.energy.symmetric,
                [[c.two_j, c.parity] for c in lv.spin_parity.candidates],
                lv.xref,
                bool(lv.metastable),
            )
            for lv in dataset.levels
            if divergence_reason(lv.spin_parity.raw) is None
        ]
        key = f"{dataset.nuclide.a}/{dataset.nuclide.z}/{dataset.dsid}"
        mine[key] = {"levels": len(prints), "digest": digest(prints)}

    compared = 0
    for key, want in ORACLE["chain"]["datasets"].items():
        if key not in mine:
            continue  # nudel indexes a few datasets we filter out by nuclide
        assert mine[key]["levels"] == want["levels"], key
        assert mine[key]["digest"] == want["digest"], key
        compared += 1
    assert compared >= 50, f"only {compared} datasets compared"


# --------------------------------------------------------------------------
# where nudel is wrong: documented, deliberately not asserted against
# --------------------------------------------------------------------------


def test_divergent_patterns_are_excluded_from_the_oracle():
    """The filter must actually fire, or the exclusions are decorative.

    ENSDF A=180 happens to contain none of these fields, so the mechanism is
    exercised directly rather than through the chain.
    """
    assert divergence_reason("3+ TO 6-") is not None
    assert divergence_reason("LE 5+") is not None
    assert divergence_reason("(GE 4)") is not None
    for ordinary in ("9-", "(1,2)+", "3 TO 6-", "1-,2-", "", "NOT 3-"):
        assert divergence_reason(ordinary) is None


def test_known_divergences_are_recorded_not_silently_dropped():
    """Anything held back must carry its reason with it."""
    for held in ORACLE["excerpt"]["known_divergences"]:
        assert held["reason"]
        assert divergence_reason(held["jpi_raw"]) is not None


# --------------------------------------------------------------------------
# characterization: our own output, for quantities nudel does not compute
# --------------------------------------------------------------------------


def test_characterization_file_says_what_it_is():
    assert "not correctness" in CHARACTERIZATION["_warning"]


def test_band_labels_and_definitions_unchanged(adopted):
    want = CHARACTERIZATION["bands"]
    assert sorted({lv.band for lv in adopted.levels if lv.band}) == want["labels"]
    assert adopted.metadata["band_definitions"] == want["definitions"]


def test_moments_unchanged(adopted):
    got = [
        {
            "energy": lv.energy_kev,
            "mu": lv.magnetic_dipole.value if lv.magnetic_dipole else None,
            "q": lv.electric_quadrupole.value if lv.electric_quadrupole else None,
        }
        for lv in adopted.levels
        if lv.magnetic_dipole or lv.electric_quadrupole
    ]
    assert got == CHARACTERIZATION["moments"]


def test_gamma_placement_unchanged(adopted):
    got = [
        {
            "energy": g.energy_kev,
            "start": g.start_index,
            "end": g.end_index,
            "mult": g.multipolarity,
        }
        for g in adopted.gammas
    ]
    assert got == CHARACTERIZATION["gammas"]


def test_q_record_unchanged(adopted):
    for key, expected in CHARACTERIZATION["q_record"].items():
        value = adopted.metadata["Q"][key]
        actual = value.value if hasattr(value, "value") else value
        if isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected


def test_decay_scheme_unchanged(decay):
    want = CHARACTERIZATION["decay"]
    assert decay.dsid == want["dsid"]
    assert str(decay.parent.nuclide) == want["parent"]
    assert decay.normalization.photon.value == pytest.approx(
        want["normalization"]["NR"]
    )
    got = [
        {
            "kind": f.kind,
            "level": f.level_index,
            "intensity": f.intensity.value,
            "log_ft": f.log_ft.value if f.log_ft else None,
        }
        for f in decay.feedings
    ]
    assert got == want["feedings"]
    assert decay.total_feeding().value == pytest.approx(want["total_feeding"])
