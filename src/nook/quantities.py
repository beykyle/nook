"""Parsing of ENSDF-flavoured physical quantities.

This is the part of the package that actually earns its keep: ENSDF encodes
uncertainties in a significant-digit convention, spin-parity as a small
grammar of alternatives and limits, and half-lives either as times or as
level widths.  Everything here is pure, stdlib-only and unit-testable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import ClassVar, Iterator

__all__ = [
    "HBAR_LN2_MEV_S",
    "HalfLife",
    "JPi",
    "SpinParity",
    "Uncertain",
    "add",
    "divide",
    "multiply",
    "parse_half_life",
    "parse_spin_parity",
    "parse_value_with_uncertainty",
]

#: hbar * ln(2) in MeV.s -- converts a level width Gamma to a half-life.
HBAR_LN2_MEV_S = 4.5623797e-22

# ENSDF limit / approximation operators, as they appear in the DE, DT fields.
_OPERATORS = ("AP", "GE", "LE", "GT", "LT", "SY", "CA", "?")


# --------------------------------------------------------------------------
# values with uncertainties
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertain:
    """A value with an (optionally asymmetric) uncertainty.

    ``operator`` carries ENSDF limit flags such as ``AP`` (approximately),
    ``GT``/``LT`` (greater/less than).  When an operator is present the
    uncertainty is usually absent and ``value`` should be read as a bound.
    """

    value: float | None
    plus: float | None = None
    minus: float | None = None
    operator: str | None = None
    raw: str = ""

    @property
    def symmetric(self) -> float | None:
        """A single symmetrised uncertainty, or ``None`` if unknown."""
        if self.plus is None and self.minus is None:
            return None
        if self.plus is None:
            return self.minus
        if self.minus is None:
            return self.plus
        return 0.5 * (self.plus + self.minus)

    @property
    def uncertainty_known(self) -> bool:
        """False when the evaluator quoted no uncertainty at all."""
        return self.plus is not None or self.minus is not None

    @property
    def is_limit(self) -> bool:
        return self.operator in ("GE", "LE", "GT", "LT")

    def __float__(self) -> float:
        if self.value is None:
            raise ValueError(f"no numeric value in {self.raw!r}")
        return self.value

    def to_ufloat(self):
        """Convert to an ``uncertainties.ufloat`` (optional dependency).

        Use this when you want that package's correlation tracking.  It is
        lossy in both directions ENSDF cares about: ``ufloat`` carries a
        single scalar standard deviation, so asymmetric errors are
        symmetrised and limit operators are dropped.  Both are reported.
        """
        from uncertainties import ufloat  # optional; not a hard dependency

        if self.value is None:
            raise ValueError(f"no numeric value in {self.raw!r}")
        if self.is_limit:
            raise ValueError(
                f"{self.operator} is a bound, not a distribution; "
                "ufloat cannot represent it"
            )
        return ufloat(self.value, self.symmetric or 0.0)

    #: Rendered forms of the limit operators, for readability.
    _SYMBOLS: ClassVar[dict[str, str]] = {
        "LT": "<",
        "LE": "\u2264",
        "GT": ">",
        "GE": "\u2265",
    }

    def __str__(self) -> str:
        if self.value is None:
            return "?"
        head = ""
        if self.operator:
            head = f"{self._SYMBOLS.get(self.operator, self.operator)} "
        sym = self.symmetric
        if sym is None:
            return f"{head}{self.value:g}"
        if self.plus is not None and self.minus is not None and self.plus != self.minus:
            return f"{head}{self.value:g} +{self.plus:g}/-{self.minus:g}"
        return f"{head}{self.value:g} +/- {sym:g}"


def _decimal_scale(value_text: str) -> tuple[float, float]:
    """Return ``(mantissa, scale)`` such that ``value = mantissa * scale``.

    ``scale`` is the weight of the last significant digit of the mantissa,
    which is the unit in which ENSDF expresses the uncertainty.  For
    ``"12.34"`` this is ``0.01``; for ``"1.2E+3"`` it is ``100.0``.
    """
    text = value_text.strip()
    exponent = 0
    m = re.search(r"[EeDd]([+-]?\d+)$", text)
    if m:
        exponent = int(m.group(1))
        text = text[: m.start()]
    if "." in text:
        decimals = len(text.split(".", 1)[1])
    else:
        decimals = 0
    return float(text) * 10.0**exponent, 10.0 ** (exponent - decimals)


def parse_value_with_uncertainty(value_text: str, unc_text: str = "") -> Uncertain:
    """Parse an ENSDF value/uncertainty pair.

    The uncertainty is expressed in units of the last significant digit of
    the value, so ``("12.1", "11")`` means ``12.1 +/- 1.1``.  Asymmetric
    uncertainties are written ``"+5-3"``.  Leading operators may sit in
    either field.
    """
    value_text = (value_text or "").strip()
    unc_text = (unc_text or "").strip()

    operator: str | None = None
    for op in _OPERATORS:
        if value_text.upper().startswith(op):
            operator, value_text = op, value_text[len(op) :].strip()
            break
    if operator is None:
        for op in _OPERATORS:
            if unc_text.upper() == op or unc_text.upper().startswith(op):
                operator = op
                unc_text = unc_text[len(op) :].strip()
                break

    raw = f"{value_text} {unc_text}".strip()
    if not value_text:
        return Uncertain(None, operator=operator, raw=raw)

    try:
        value, scale = _decimal_scale(value_text)
    except ValueError:
        return Uncertain(None, operator=operator, raw=raw)

    if not unc_text:
        return Uncertain(value, operator=operator, raw=raw)

    asym = re.fullmatch(r"\+(\d+)\s*-(\d+)", unc_text)
    if asym:
        return Uncertain(
            value,
            plus=int(asym.group(1)) * scale,
            minus=int(asym.group(2)) * scale,
            operator=operator,
            raw=raw,
        )
    if re.fullmatch(r"\d+", unc_text):
        sigma = int(unc_text) * scale
        return Uncertain(value, plus=sigma, minus=sigma, operator=operator, raw=raw)

    # Anything else (e.g. "LT", stray text) -- keep the value, drop the sigma.
    return Uncertain(value, operator=operator, raw=raw)


#: Operators that assert an upper bound, a lower bound, or neither.
_UPPER_LIMITS = frozenset({"LT", "LE"})
_LOWER_LIMITS = frozenset({"GT", "GE"})
_QUALITY_FLAGS = frozenset({"AP", "SY", "CA"})
_FLIP = {"LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE"}


def _combine_operators(
    a: str | None, b: str | None, flip_b: bool = False
) -> tuple[str | None, bool]:
    """Combine two ENSDF operators across a monotonic binary operation.

    Returns ``(operator, bounded)``.  ``bounded`` is False when the inputs
    bound the result in opposite directions -- an upper limit times a lower
    limit constrains nothing, and inventing a number there would be worse
    than admitting ignorance.  ``flip_b`` reverses b's direction, which is
    what division by a bounded quantity does.
    """
    if flip_b and b in _FLIP:
        b = _FLIP[b]
    limits = [op for op in (a, b) if op in _UPPER_LIMITS | _LOWER_LIMITS]
    if not limits:
        flags = {op for op in (a, b) if op in _QUALITY_FLAGS}
        for weakest in ("SY", "CA", "AP"):  # least defensible claim wins
            if weakest in flags:
                return weakest, True
        return None, True
    if len(limits) == 2:
        upper = all(op in _UPPER_LIMITS for op in limits)
        lower = all(op in _LOWER_LIMITS for op in limits)
        if not (upper or lower):
            return None, False
        strict = any(op in ("LT", "GT") for op in limits)
        if upper:
            return ("LT" if strict else "LE"), True
        return ("GT" if strict else "GE"), True
    return limits[0], True


def _relative(term: Uncertain) -> tuple[float, float]:
    """Fractional (plus, minus) uncertainties; zero when unknown."""
    if not term.value:
        return 0.0, 0.0
    magnitude = abs(term.value)
    plus = (term.plus / magnitude) if term.plus is not None else 0.0
    minus = (term.minus / magnitude) if term.minus is not None else 0.0
    return plus, minus


def _combine(
    a: Uncertain, b: Uncertain, value: float, swap: bool, raw: str, strict: bool = False
) -> Uncertain:
    """Assemble a product/quotient, keeping the two error branches separate.

    A blank uncertainty is treated as exact -- the ENSDF convention for a
    normalisation of exactly 1.0. It can also mean "not quoted", in which case
    contributing zero flatters the result; ``strict=True`` refuses instead.
    """
    if strict and not (a.uncertainty_known and b.uncertainty_known):
        return Uncertain(value, None, None, a.operator or b.operator, raw=raw)
    ap, am = _relative(a)
    bp, bm = _relative(b)
    if swap:  # a denominator's upward fluctuation pushes the result down
        bp, bm = bm, bp
    plus = abs(value) * math.sqrt(ap * ap + bp * bp)
    minus = abs(value) * math.sqrt(am * am + bm * bm)
    operator, bounded = _combine_operators(a.operator, b.operator, flip_b=swap)
    if not bounded:
        return Uncertain(None, raw=f"{raw} (opposing limits, unbounded)")
    known = a.uncertainty_known or b.uncertainty_known
    return Uncertain(
        value,
        plus if known else None,
        minus if known else None,
        operator,
        raw=raw,
    )


def multiply(a: Uncertain, b: Uncertain, strict: bool = False) -> Uncertain:
    """Product of two values, each error branch propagated independently.

    Relative errors add in quadrature per side, which assumes independence --
    see :meth:`DecayScheme.total_feeding` for the correlated case.
    """
    if a.value is None or b.value is None:
        return Uncertain(None, raw=f"{a.raw} * {b.raw}")
    return _combine(
        a, b, a.value * b.value, swap=False, raw=f"{a.raw} * {b.raw}", strict=strict
    )


def divide(a: Uncertain, b: Uncertain, strict: bool = False) -> Uncertain:
    """Quotient of two values, with the denominator's error branches swapped."""
    if a.value is None or b.value is None or not b.value:
        return Uncertain(None, raw=f"{a.raw} / {b.raw}")
    return _combine(
        a, b, a.value / b.value, swap=True, raw=f"{a.raw} / {b.raw}", strict=strict
    )


def add(*terms: Uncertain, strict: bool = False) -> Uncertain:
    """Sum of independent values, each error branch in quadrature.

    Independence is assumed.  Summing quantities that share a common scale
    factor -- every intensity in one decay dataset shares ``NR`` and ``BR``
    -- violates that; sum the unscaled values and apply the factor once.
    """
    usable = [t for t in terms if t.value is not None]
    if not usable:
        return Uncertain(None, raw="empty sum")
    total = math.fsum(t.value for t in usable if t.value is not None)
    if strict and not all(t.uncertainty_known for t in usable):
        return Uncertain(total, None, None, raw="sum with an unquoted uncertainty")
    plus = math.sqrt(sum((t.plus or 0.0) ** 2 for t in usable))
    minus = math.sqrt(sum((t.minus or 0.0) ** 2 for t in usable))
    operator: str | None = None
    for term in usable:
        operator, bounded = _combine_operators(operator, term.operator)
        if not bounded:
            operator = None
            break
    known = any(t.uncertainty_known for t in usable)
    return Uncertain(
        total,
        plus if known else None,
        minus if known else None,
        operator,
        raw=f"sum of {len(usable)} values",
    )


# --------------------------------------------------------------------------
# spin and parity
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class JPi:
    """A single spin-parity assignment.

    Spin is stored as ``two_j`` so that half-integer values stay exact.
    ``parity`` is ``+1``, ``-1`` or ``None`` when unassigned.
    """

    two_j: int | None
    parity: int | None = None

    @property
    def j(self) -> Fraction | None:
        return None if self.two_j is None else Fraction(self.two_j, 2)

    @property
    def j_float(self) -> float | None:
        return None if self.two_j is None else self.two_j / 2.0

    def __str__(self) -> str:
        if self.two_j is None:
            j = "?"
        elif self.two_j % 2 == 0:
            j = str(self.two_j // 2)
        else:
            j = f"{self.two_j}/2"
        return j + {1: "+", -1: "-", None: ""}[self.parity]


@dataclass(frozen=True)
class SpinParity:
    """An ENSDF ``J`` field (manual V.20) -- a small language, not a number.

    Two traps, both silent when got wrong:

    * a parity attaches only where written, *except* after a closing
      parenthesis, which covers the group: ``(3,4)-`` is 3- or 4-, but
      ``1,2+`` is "J=1 with unknown parity, or 2+";
    * ``NOT`` inverts the statement, so those values go to :attr:`excluded`
      and must never be reported as the assignment.
    """

    candidates: tuple[JPi, ...] = ()
    #: Values the evaluator ruled out with ``NOT``.
    excluded: tuple[JPi, ...] = ()
    tentative: bool = False
    assumed: bool = False
    constraint: str | None = None
    #: True for ``NATURAL`` parity (pi = (-1)**J), False for ``UNNATURAL``.
    natural_parity: bool | None = None
    raw: str = ""

    @property
    def unique(self) -> JPi | None:
        """The assignment when it is unambiguous, else ``None``."""
        return self.candidates[0] if len(self.candidates) == 1 else None

    @property
    def is_known(self) -> bool:
        return bool(self.candidates)

    def allows(self, two_j: int, parity: int | None = None) -> bool:
        """Whether the field permits this spin-parity.

        Honours ``NOT`` exclusions and ``NATURAL``/``UNNATURAL`` parity,
        neither of which appear in :attr:`candidates`.
        """
        for bad in self.excluded:
            if bad.two_j == two_j and (parity is None or bad.parity in (parity, None)):
                return False
        if self.natural_parity is not None and parity is not None:
            natural = 1 if (two_j // 2) % 2 == 0 else -1
            if two_j % 2 == 0 and parity != (
                natural if self.natural_parity else -natural
            ):
                return False
        if not self.candidates:
            return not self.excluded or True
        return self.matches(two_j, parity)

    def matches(self, two_j: int | None = None, parity: int | None = None) -> bool:
        """True if any candidate is compatible with the given quantum numbers."""
        for cand in self.candidates:
            if two_j is not None and cand.two_j != two_j:
                continue
            if parity is not None and cand.parity not in (parity, None):
                continue
            return True
        return False

    def __iter__(self) -> Iterator[JPi]:
        return iter(self.candidates)

    def __str__(self) -> str:
        return self.raw or "?"


_J_SEPARATORS = re.compile(r"[,&]|\bAND\b|\bOR\b")
_PARITY = {"+": 1, "-": -1, "": None}
_GROUP_PARITY = re.compile(r"\(([^()]*)\)\s*([+-])")
_RANGE = re.compile(r"(\d+(?:/2)?)\s*([+-]?)\s*(?:TO|:)\s*(\d+(?:/2)?)\s*([+-]?)")


def _token_to_two_j(token: str) -> int | None:
    token = token.strip()
    if token.endswith("/2"):
        head = token[:-2].strip()
        return int(head) if head.isdigit() else None
    return int(token) * 2 if token.isdigit() else None


def _parse_jpi_token(token: str) -> JPi | None:
    token = token.strip()
    parity = None
    if token.endswith(("+", "-")):
        parity, token = _PARITY[token[-1]], token[:-1]
    two_j = _token_to_two_j(token)
    return None if two_j is None else JPi(two_j, parity)


def _expand_range(lo_text: str, lo_par: str, hi_text: str, hi_par: str) -> list[JPi]:
    """Expand ``J TO J'`` per the manual's three parity cases.

    ``3 TO 6-`` gives every value pi=-; ``3+ TO 6-`` keeps the endpoints and
    leaves the interior parities undetermined; ``3+ TO 6`` leaves everything
    above the first value undetermined.
    """
    lo, hi = _token_to_two_j(lo_text), _token_to_two_j(hi_text)
    if lo is None or hi is None or hi < lo:
        return []
    low_parity, high_parity = _PARITY[lo_par], _PARITY[hi_par]
    out = []
    for value in range(lo, hi + 1, 2):
        if value == lo:
            parity = low_parity if low_parity is not None else high_parity
        elif value == hi:
            parity = high_parity
        else:
            parity = high_parity if low_parity is None else None
        out.append(JPi(value, parity))
    return out


def _distribute_group_parity(text: str) -> str:
    """``(3,4)-`` -> ``3-,4-``; a parity after a group applies to all of it."""

    def replace(match: re.Match[str]) -> str:
        inner, parity = match.group(1), match.group(2)
        parts = [p.strip() for p in _J_SEPARATORS.split(inner) if p.strip()]
        return ",".join(p if p.endswith(("+", "-")) else p + parity for p in parts)

    return _GROUP_PARITY.sub(replace, text)


def parse_spin_parity(text: str) -> SpinParity:
    """Parse an ENSDF ``J`` field into a :class:`SpinParity`.

    A parity is attached to a value only where the file attaches it -- there
    is no inheritance across a comma.  ``1,2+`` is "J=1 with unknown parity,
    or 2+", not "1+ or 2+".  The one exception is a parity written after a
    closing parenthesis, which the manual defines as applying to the group.
    """
    raw = (text or "").strip()
    if not raw or raw in ("?", "*"):
        return SpinParity(raw=raw)

    body = raw.upper()
    assumed = "[" in body
    tentative = "(" in body

    if "NATURAL" in body:
        return SpinParity(
            natural_parity="UNNATURAL" not in body,
            tentative=tentative,
            assumed=assumed,
            raw=raw,
        )

    negated = bool(re.search(r"\bNOT\b", body))
    body = re.sub(r"\bNOT\b", " ", body)

    body = _distribute_group_parity(body)
    for bracket in "()[]":
        body = body.replace(bracket, " ")

    constraint = None
    for operator in ("GE", "LE", "GT", "LT", "AP"):
        if re.search(rf"\b{operator}\b", body):
            constraint = operator
            body = re.sub(rf"\b{operator}\b", " ", body)

    found: list[JPi] = []
    for chunk in _J_SEPARATORS.split(body):
        chunk = chunk.strip()
        if not chunk:
            continue
        span = _RANGE.fullmatch(chunk)
        if span:
            found.extend(_expand_range(*span.groups()))
            continue
        parsed = _parse_jpi_token(chunk)
        if parsed is not None:
            found.append(parsed)

    return SpinParity(
        candidates=() if negated else tuple(found),
        excluded=tuple(found) if negated else (),
        tentative=tentative,
        assumed=assumed,
        constraint=constraint,
        raw=raw,
    )


# --------------------------------------------------------------------------
# half-lives and widths
# --------------------------------------------------------------------------

_TIME_UNITS = {
    "Y": 3.15576e7,
    "YR": 3.15576e7,
    "D": 86400.0,
    "H": 3600.0,
    "M": 60.0,
    "MIN": 60.0,
    "S": 1.0,
    "MS": 1e-3,
    "US": 1e-6,
    "NS": 1e-9,
    "PS": 1e-12,
    "FS": 1e-15,
    "AS": 1e-18,
}
_WIDTH_UNITS = {"EV": 1e-6, "KEV": 1e-3, "MEV": 1.0}


@dataclass(frozen=True)
class HalfLife:
    """A level half-life, possibly given as a width or as ``STABLE``."""

    seconds: Uncertain = field(default_factory=lambda: Uncertain(None))
    stable: bool = False
    from_width: bool = False
    width_mev: Uncertain | None = None
    raw: str = ""

    @property
    def is_known(self) -> bool:
        return self.stable or self.seconds.value is not None

    def __str__(self) -> str:
        if self.stable:
            return "STABLE"
        return self.raw or str(self.seconds)


def parse_half_life(text: str, unc_text: str = "") -> HalfLife:
    """Parse an ENSDF ``T1/2`` field, e.g. ``"12.1 H"``, ``"3.2 KEV"``.

    Widths are converted to half-lives through ``T = hbar ln2 / Gamma``.
    """
    raw = (text or "").strip()
    if not raw:
        return HalfLife(raw=raw)
    if raw.upper().startswith("STABLE"):
        return HalfLife(seconds=Uncertain(math.inf, raw=raw), stable=True, raw=raw)

    m = re.match(
        r"^\s*((?:AP|GE|LE|GT|LT)\s*)?([\d.]+(?:[EeDd][+-]?\d+)?)\s*([A-Za-z]+)\s*(.*)$",
        raw,
    )
    if not m:
        return HalfLife(raw=raw)

    operator = (m.group(1) or "").strip().upper() or None
    number, unit, trailing = m.group(2), m.group(3).upper(), m.group(4).strip()
    sigma_text = unc_text.strip() or trailing

    base = parse_value_with_uncertainty(number, sigma_text)
    if operator and base.operator is None:
        base = Uncertain(base.value, base.plus, base.minus, operator, base.raw)

    if unit in _TIME_UNITS:
        f = _TIME_UNITS[unit]
        return HalfLife(
            seconds=Uncertain(
                None if base.value is None else base.value * f,
                None if base.plus is None else base.plus * f,
                None if base.minus is None else base.minus * f,
                base.operator,
                raw,
            ),
            raw=raw,
        )

    if unit in _WIDTH_UNITS:
        f = _WIDTH_UNITS[unit]
        width = Uncertain(
            None if base.value is None else base.value * f,
            None if base.plus is None else base.plus * f,
            None if base.minus is None else base.minus * f,
            base.operator,
            raw,
        )
        seconds = None if not width.value else HBAR_LN2_MEV_S / width.value
        # dT/T = dGamma/Gamma, with the branches swapped by the reciprocal.
        rel_p = (width.plus / width.value) if width.value and width.plus else None
        rel_m = (width.minus / width.value) if width.value and width.minus else None
        return HalfLife(
            seconds=Uncertain(
                seconds,
                plus=None if (seconds is None or rel_m is None) else seconds * rel_m,
                minus=None if (seconds is None or rel_p is None) else seconds * rel_p,
                operator=width.operator,
                raw=raw,
            ),
            from_width=True,
            width_mev=width,
            raw=raw,
        )

    return HalfLife(raw=raw)
