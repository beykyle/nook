# Suspicious entries

A record of what `inconsistencies()` still flags after `repair()` has done what
it can, on a full ENSDF distribution (299 mass chains, 3168 nuclides).

These are not bugs in this package. They are places where the evaluated file
contradicts itself, and each is listed with what the contradiction is, what the
value would have to be to resolve it, and why the repair machinery declined to
act. Nothing here is corrected in the data.

Regenerate with:

```python
from nook.survey import survey, inconsistencies
from nook.repair import repair

fixed, changes = repair(survey())
for state, why in inconsistencies(fixed):
    print(state.nuclide, why)
```

## Diagnosed but not repaired

### ²¹⁰Ac — S(n) is 5000 keV too low

The strongest case in the set. ²¹⁰Ac appears in **three** failing loops, always
as the same term:

| loop | anchor | residual |
|---|---|---|
| neutron surface | ²¹⁰Ra | 5000 keV (38 σ) |
| neutron surface | ²¹⁰Ac | 5000 keV (30 σ) |
| proton surface | ²¹⁰Ac | 4992 keV (47 σ) |

Three independent identities implicating one value is a strong localisation.
The file reads `S(n) = 3130 ± 80 keV`; closure requires **8130 keV**. Two
further arguments agree: ²⁰⁹Ac has S(n) = 9990 keV at N = 120, and the odd-N
neighbour should sit roughly 2 MeV lower — 8130 does, 3130 does not.

`3130 → 8130` is a single-digit substitution. That is deliberately outside the
repair vocabulary: unlike a sign or a power of ten, a digit error has no unique
inverse, so accepting one would mean trusting the closure arithmetic to invent
a value rather than to confirm a hypothesis.

### ¹¹²Nb — almost certainly a sign flip, but unprovable

`S(n) = −3500 ± 400 keV`, with a half-life of 33 ms. Neutron emission from an
unbound state is prompt, so the two cannot both be right, and the rest of
`ensdf.112` has exactly this error twenty times over.

It is not repaired because its loops cannot be evaluated: they need ¹¹²Mo and
¹¹¹Nb, and every Q-value for both is flagged `SY`. With no measured
constraint there is no evidence, and a repair without evidence is a guess —
even a very good one.

## Flagged, with no identity to constrain them

### ¹¹⁷Ru — Q(α) = −91800 keV

An order of magnitude outside any physical range; the value is likely −9180 keV.
Neither identity involves Q(α), so nothing here can confirm it. It stays as the
file has it.

### ¹¹²Cd, ¹¹²Sn — S(p) after repair

Both had S(n) and S(p) negated and both were repaired. They no longer appear.
Listed here only because earlier versions of this document, written before the
proton-surface identity existed, reported them as unfixable.

## Extrapolations, not errors

²⁴N, ³³Ne and ³⁹Mg each have a slightly negative S(n) alongside a half-life of
tens to hundreds of nanoseconds:

| nuclide | S(n) | T½ |
|---|---|---|
| ²⁴N | −500 keV | 52 ns |
| ³³Ne | −900 keV | 180 ns |
| ³⁹Mg | −130 keV | 180 ns |

Every one of those S(n) values carries the `SY` operator: extrapolated from
systematics, not measured, with an uncertainty that straddles zero. The
flag is telling you the number is unreliable, not that it is wrong — the true
values are presumably small and positive. These are never repaired, because
rewriting an extrapolation would be fabrication rather than correction.

## Small residuals: the evaluation disagreeing with itself

⁹⁹Sr and ¹⁰⁰Sr fail the proton-surface identity by 55 and 98 keV. This is a
different character from everything above — no transcription error produces a
55 keV offset.

The identity holds to 0–2 keV for every other Rb → Sr pair from A = 85 to 104:

```
A=85:0  A=86:0  A=87:0  A=88:0  A=89:0  A=90:0  A=91:0  A=92:0  A=93:1
A=94:0  A=95:1  A=96:0  A=97:0  A=98:0  A=99:55 A=100:98 A=101:2 A=102:2
A=103:58 A=104:2
```

So it is isolated to three neutron-rich nuclides where the masses are least
well determined, and it is consistent with the A=98/99/100 chains having
adopted marginally different mass inputs for the same nuclide. (A=103 shows the
same 58 keV effect but sits under the 5 σ threshold, because the uncertainties
there are larger.)

Worth knowing rather than worth fixing: if you are differencing separation
energies in this region at the 50 keV level, the inputs are not internally
consistent to that precision.

## Summary

| nuclide | quantity | issue | why not repaired |
|---|---|---|---|
| ²¹⁰Ac | S(n) | 3130 keV, should be 8130 | digit substitution, no unique inverse |
| ¹¹²Nb | S(n) | −3500 keV with a 33 ms half-life | neighbours are all `SY`; no loop to test against |
| ¹¹⁷Ru | Q(α) | −91800 keV | no identity constrains Q(α) |
| ²⁴N, ³³Ne, ³⁹Mg | S(n) | slightly negative, ns half-lives | `SY` extrapolations straddling zero |
| ⁹⁹Sr, ¹⁰⁰Sr | S(p)/S(n)/Q(β⁻) | 55–98 keV non-closure | too small for any transcription error |
