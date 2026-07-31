"""Ground-state properties: masses, abundances, radii and moments.

Livechart aggregates several distinct compilations behind one endpoint. They
are not ENSDF and do not share a provenance, so :data:`EVALUATIONS` records the
attribution per quantity and :data:`UNITS` the units -- ``atomic_mass`` is in
micro-u, the kind of mismatch that survives review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .nuclide import Nuclide
from .quantities import HalfLife, SpinParity, Uncertain

__all__ = ["EVALUATIONS", "UNITS", "GroundState"]


#: Which compilation each quantity comes from.  Livechart does not stamp a
#: version per field, so treat these as the evaluation *families*; check the
#: Livechart guide for the vintage currently being served.
EVALUATIONS = {
    "mass_excess": "AME (atomic mass evaluation)",
    "atomic_mass": "AME (atomic mass evaluation)",
    "binding_per_nucleon": "AME (atomic mass evaluation)",
    "q_beta_minus": "AME (atomic mass evaluation)",
    "q_beta_minus_n": "AME (atomic mass evaluation)",
    "q_alpha": "AME (atomic mass evaluation)",
    "q_ec": "AME (atomic mass evaluation)",
    "s_n": "AME (atomic mass evaluation)",
    "s_p": "AME (atomic mass evaluation)",
    "abundance": "NUBASE",
    "discovery_year": "NUBASE",
    "charge_radius": "nuclear ground-state charge radii compilation",
    "magnetic_dipole": "nuclear magnetic dipole moment tables",
    "electric_quadrupole": "nuclear electric quadrupole moment tables",
    "spin_parity": "ENSDF",
    "half_life": "ENSDF",
    "isospin": "ENSDF",
}

#: The API's units.  Note that atomic_mass is in micro-u, not u.
UNITS = {
    "mass_excess": "keV",
    "atomic_mass": "micro-u",
    "binding_per_nucleon": "keV",
    "q_beta_minus": "keV",
    "q_beta_minus_n": "keV",
    "q_alpha": "keV",
    "q_ec": "keV",
    "s_n": "keV",
    "s_p": "keV",
    "abundance": "% (mole fraction)",
    "charge_radius": "fm",
    "magnetic_dipole": "nuclear magnetons",
    "electric_quadrupole": "barn",
}


def _empty() -> Uncertain:
    return Uncertain(None)


@dataclass(frozen=True)
class GroundState:
    """Aggregated ground-state properties for one nuclide.

    Every numeric field is an :class:`~nook.quantities.Uncertain`; a
    quantity the compilations do not carry is left with ``value is None``
    rather than defaulted, so a missing mass excess cannot be mistaken for
    zero.
    """

    nuclide: Nuclide
    spin_parity: SpinParity = field(default_factory=SpinParity)
    half_life: HalfLife = field(default_factory=HalfLife)

    # atomic mass evaluation
    mass_excess: Uncertain = field(default_factory=_empty)
    atomic_mass: Uncertain = field(default_factory=_empty)
    binding_per_nucleon: Uncertain = field(default_factory=_empty)
    #: True when the mass comes from systematics rather than measurement.
    mass_from_systematics: bool = False
    q_beta_minus: Uncertain = field(default_factory=_empty)
    q_beta_minus_n: Uncertain = field(default_factory=_empty)
    q_alpha: Uncertain = field(default_factory=_empty)
    q_ec: Uncertain = field(default_factory=_empty)
    s_n: Uncertain = field(default_factory=_empty)
    s_p: Uncertain = field(default_factory=_empty)

    # nubase
    abundance: Uncertain = field(default_factory=_empty)
    discovery_year: int | None = None

    # radii and electromagnetic moments
    charge_radius: Uncertain = field(default_factory=_empty)
    magnetic_dipole: Uncertain = field(default_factory=_empty)
    electric_quadrupole: Uncertain = field(default_factory=_empty)

    isospin: str | None = None
    decay_modes: tuple[tuple[str, Uncertain], ...] = ()
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def is_stable(self) -> bool:
        return self.half_life.stable

    @property
    def total_binding_energy(self) -> Uncertain:
        """Binding energy summed over all nucleons, in keV."""
        from .quantities import multiply

        return multiply(self.binding_per_nucleon, Uncertain(float(self.nuclide.a)))

    def provenance(self, quantity: str) -> str | None:
        """Which compilation a given attribute came from."""
        return EVALUATIONS.get(quantity)

    def units(self, quantity: str) -> str | None:
        return UNITS.get(quantity)

    def __str__(self) -> str:
        rows = [
            f"{self.nuclide} ground state  {self.spin_parity}  T1/2 = {self.half_life}"
        ]
        for name in (
            "mass_excess",
            "binding_per_nucleon",
            "s_n",
            "s_p",
            "abundance",
            "charge_radius",
            "magnetic_dipole",
            "electric_quadrupole",
        ):
            value = getattr(self, name)
            if value.value is not None:
                rows.append(f"  {name:<22} {value!s:<24} {UNITS.get(name, '')}")
        if self.discovery_year:
            rows.append(f"  {'discovered':<22} {self.discovery_year}")
        return "\n".join(rows)
