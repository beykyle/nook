"""Canonicalisation shared by the oracle generator and the tests.

Two independent parsers can only be compared byte-for-byte once their outputs
are reduced to a common form, so this is the single definition of that form:
``tools/generate_golden.py`` uses it to capture nudel's output,
``test_differential.py`` to check ours.

It also holds the ``J`` fields where nudel is known to be wrong, so the
generator can hold those levels back and the tests can say why. See
``docs/testing.md``.
"""

from __future__ import annotations

import hashlib
import math
import re

__all__ = [
    "DIVERGENT_J",
    "absent",
    "canonical",
    "digest",
    "divergence_reason",
    "level_fingerprint",
]

#: ``J`` fields nudel decodes incorrectly, each checked against ENSDF manual
#: V.20. Levels matching these are excluded from the oracle: freezing a value
#: we believe to be wrong is worse than freezing none. Our own behaviour for
#: these is pinned by spec-derived unit tests in ``test_nook.py``.
DIVERGENT_J = (
    (
        re.compile(r"\d\s*[+-]\s*(TO|:)"),
        "range with a parity on the lower bound: V.20 case (b)/(c) leaves the "
        "interior parities undetermined, nudel propagates one endpoint's",
    ),
    (
        re.compile(r"^\s*\(?\s*(LE|GE|LT|GT|AP)\b"),
        "spin bounded by an operator: V.20 says 'LE 5+' means pi=+ with J<=5, "
        "nudel yields no assignment at all",
    ),
)


def divergence_reason(raw_j: str) -> str | None:
    """Why nudel's reading of this ``J`` field cannot be trusted, if it can't."""
    for pattern, reason in DIVERGENT_J:
        if pattern.search((raw_j or "").upper()):
            return reason
    return None


def absent(value) -> bool:
    """True for every way the two libraries spell "no value".

    nudel uses ``float('nan')`` where this package uses ``None``. Reconciling
    that here, in the shared canonicalisation, keeps it out of the tests --
    otherwise every comparison grows a special case and a genuine ``nan``
    could slip through as agreement.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def canonical(value: float | None, digits: int = 6) -> str | None:
    """Round to fixed precision so two implementations can agree exactly."""
    return None if absent(value) else f"{value:.{digits}g}"


def level_fingerprint(
    energy: float | None,
    unc: float | None,
    jpi: list,
    xref,
    metastable: bool,
) -> str:
    """The fields both parsers compute, in a float-noise-robust form.

    Half-lives are deliberately excluded: the two implementations use
    different definitions of the year (Julian vs tropical, 21 ppm apart), so
    they are compared with a tolerance rather than hashed.
    """
    return "|".join(
        [
            canonical(energy) or "-",
            canonical(unc) or "-",
            ";".join(f"{j}/{p}" for j, p in sorted(jpi)),
            "".join(sorted(xref)),
            "M" if metastable else "-",
        ]
    )


def digest(fingerprints: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
