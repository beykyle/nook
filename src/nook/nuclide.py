"""Nuclide identifiers: parsing, normalisation and ENSDF NUCID handling."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ELEMENTS", "Nuclide", "symbol_to_z", "z_to_symbol"]

ELEMENTS = (
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

_BY_SYMBOL = {sym.upper(): z for z, sym in enumerate(ELEMENTS)}


def symbol_to_z(symbol: str) -> int:
    """``"Fe"`` -> ``26``.  Case-insensitive."""
    try:
        return _BY_SYMBOL[symbol.strip().upper()]
    except KeyError:
        raise ValueError(f"unknown element symbol {symbol!r}") from None


def z_to_symbol(z: int) -> str:
    """``26`` -> ``"Fe"``."""
    if not 0 <= z < len(ELEMENTS):
        raise ValueError(f"Z={z} outside the known element table")
    return ELEMENTS[z]


# "56Fe", "Fe56", "Fe-56", "fe_56", and NUCID-style " 56FE"
_PATTERNS = (
    re.compile(r"^\s*(?P<a>\d{1,3})\s*[-_ ]?\s*(?P<sym>[A-Za-z]{1,2})\s*$"),
    re.compile(r"^\s*(?P<sym>[A-Za-z]{1,2})\s*[-_ ]?\s*(?P<a>\d{1,3})\s*$"),
)


@dataclass(frozen=True, order=True)
class Nuclide:
    """An immutable ``(Z, A)`` nuclide identifier."""

    z: int
    a: int

    def __post_init__(self) -> None:
        if self.z < 0 or self.a < 1 or self.a < self.z:
            raise ValueError(f"implausible nuclide Z={self.z}, A={self.a}")

    @property
    def n(self) -> int:
        return self.a - self.z

    @property
    def symbol(self) -> str:
        return z_to_symbol(self.z)

    @classmethod
    def parse(cls, value: "str | Nuclide | tuple[int, int]") -> "Nuclide":
        """Accept ``"56Fe"``, ``"Fe-56"``, ``(26, 56)`` or a ``Nuclide``."""
        if isinstance(value, Nuclide):
            return value
        if isinstance(value, tuple):
            z, a = value
            return cls(int(z), int(a))
        for pattern in _PATTERNS:
            m = pattern.match(str(value))
            if m:
                return cls(symbol_to_z(m.group("sym")), int(m.group("a")))
        raise ValueError(f"cannot parse nuclide from {value!r}")

    def nucid(self) -> str:
        """The 5-character ENSDF NUCID, e.g. ``" 56FE"``."""
        return f"{self.a:>3d}{self.symbol.upper():<2s}"

    @classmethod
    def from_nucid(cls, nucid: str) -> "Nuclide":
        """Parse a 5-character ENSDF NUCID card field."""
        a_text, sym = nucid[:3].strip(), nucid[3:5].strip()
        if not a_text.isdigit() or not sym:
            raise ValueError(f"not a NUCID: {nucid!r}")
        return cls(symbol_to_z(sym), int(a_text))

    def livechart_id(self) -> str:
        """The identifier the IAEA Livechart API expects, e.g. ``"56fe"``."""
        return f"{self.a}{self.symbol.lower()}"

    def __str__(self) -> str:
        return f"{self.a}{self.symbol}"
