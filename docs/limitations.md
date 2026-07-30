# Limitations

## Not parsed

- **Decay-scheme normalisation beyond the basics.** `R` (reference) records are
  decoded only via `parse_references`; keynumbers are not expanded inline.
- **Delayed-particle (`D`) records** are decoded but lightly modelled — the
  intermediate-level energy and width are typed, the rest stays in `raw`.
- **ENSDF/2 records** (`I` statistics, indexed levels) are not handled.
- Most adopted datasets carry **two Q records**: NNDC prepends one from the
  newest mass evaluation, then a comment reading *"current evaluation has used
  the following Q record"*, then the older set the evaluator worked from. This
  package takes the first (newer masses) and keeps the rest in
  `metadata["Q_records"]`. If you need internal consistency with the
  evaluation's own derived quantities, use the later one.

## Approximations

- Asymmetric errors propagate branch-by-branch in quadrature. That is the usual
  convention, not a rigorous treatment of skewed distributions; if you need that
  for log *ft* limits, you want sampling.
- `multiply()` symmetrises only in `to_ufloat()`, never internally.
- `NP` for delayed particles is applied without `BR`, inferred from a single
  dataset (see `Normalization.factor_for`). Worth checking against the manual
  before relying on it.
- Gamma placement without an `FL=` continuation is a heuristic energy match.

## Data quality in the files themselves

Evaluated files carry occasional transcription errors, and a survey that
absorbs them silently puts them straight into a figure.
`survey.inconsistencies()` flags records that contradict themselves — nothing
is corrected, because guessing which of two numbers is wrong is not the
package's job:

```python
from nook.survey import survey, inconsistencies

for state, why in inconsistencies(survey()):
    print(state.nuclide, why)
```

Most checks are statements that cannot be true of any nuclide: a stable nuclide
with a negative separation energy; a stable nuclide with a positive Q(β⁻); a
neutron-unbound nuclide with a measurable half-life, since neutron emission
from an unbound state is prompt; a separation energy of tens of MeV.

### The mass surface

The strongest check needs no threshold at all. Separation energies and beta
Q-values are differences of the same mass excesses, so they close a loop:

    S(n)[Z, A] − S(n)[Z+1, A]  ==  Q(β⁻)[Z, A−1] − Q(β⁻)[Z, A]

Both sides reduce to the same four masses, so this holds *exactly*, whatever
the masses are. A loop that fails to close means at least one of the four
numbers is wrong — and it catches errors no single-record check can, because
each value looks entirely reasonable on its own.

Across a full distribution the measured loops are sharply bimodal: they close
to a median of **0.02 σ**, or they fail by more than **20 σ**, with nothing in
between. So the threshold is not delicate — 5 σ and 20 σ select the same set.
Loops containing a value flagged `SY` or `CA` are skipped, since extrapolations
are not expected to close; those are looser by two orders of magnitude
(median 10 keV against 0.4 keV).

Two identities are used. The second is what makes S(p) checkable:

    S(p)[Z, A] − S(n)[Z, A]  ==  Q(β⁻)[Z−1, A−1] + Δ(¹H) − Δ(n)

It closes to a median of 0.35 keV over 1739 testable instances.

`closure_failures()` reports the residual and how many sigma it represents.

## Repairing transcription errors

**Off by default.** Correcting evaluated data is dangerous, and `repair()` is
built so that a correction is never a guess about what a number *ought* to be:

```python
from nook.repair import repair

fixed, changes = repair(survey())
for change in changes:
    print(change)
# 112Sn s_n: -10788 -> 10788 keV (negate); loop residual 21575 -> 1 keV
```

Four constraints make it defensible:

1. **A fixed vocabulary.** Only `negate` and powers of ten — the two ways a
   transcription actually fails. A value wrong by a factor of three has no
   transcription story behind it and is left alone.
2. **Verified, not assumed.** A candidate is accepted only if it closes a
   mass-surface loop that was open, which is evidence independent of any
   opinion about the value. It must also not break a loop that was closing.
3. **Only what an identity constrains.** `Q(α)` enters neither identity, so
   ¹¹⁷Ru's `−91800 keV` stays exactly as the file has it, however obviously
   wrong. `SY`/`CA` values are never touched: an extrapolation is uncertain,
   not wrong, and rewriting one would be a fabrication.
4. **Recorded and visible.** Every change carries the original value, the
   transform, and the residual before and after. Repaired quantities are
   listed in `NuclideSummary.repaired`, so downstream code can tell which
   numbers are no longer the file's.

On a full distribution this makes 22 repairs — 21 sign flips and one dropped
exponent — taking closure failures from 29 to 5. **Twenty of the 22 land within
1 keV of closure**, from residuals of 10–27 MeV.

An independent check the repair machinery knows nothing about: afterwards, the
Cd, Sn and Te isotopic chains show the correct odd-even staggering, with the
even-N A=112 members sitting ~2.5 MeV above their A=111 and A=113 neighbours.
That is the neutron pairing gap, and nothing in the repair logic is aware
pairing exists.

What survives is the honest residue: nine nuclides, each written up with its
diagnosis in [suspicious entries](suspicious-entries.md). The check can
localise an error to a loop, and sometimes — as with ²¹⁰Ac, implicated by three
independent loops — to a single value; it still cannot always say what the
right number is.

Four things this found in one ENSDF snapshot:

- **A=112.** Eleven of the fifteen Q records in `ensdf.112` have S(n) *and*
  S(p) negated. ¹¹²Sn reads `S(n) = −10788 ± 5` when the true value is
  +10.788 MeV — magnitude right, sign wrong. ¹¹¹Sn and ¹¹³Sn are correct with
  the same AME2012 reference, so it is localised to that file. On a chart of
  nuclides it appears as a diagonal streak, because A = constant is a diagonal
  in N–Z.
- **¹¹⁷Ru** has Q(α) = −91800 keV, an order of magnitude out.
- **A=173.** The closure test alone catches this one. ¹⁷³Ir's S(n) field reads
  `1.0960` where the value should be ≈10960 keV — the exponent is missing, and
  the loop residual is 10960.9 keV, exactly the lost factor. The same
  corruption appears in ⁶³Ga's superseded record.
- **A=210/211.** ²¹⁰Ac, ²¹⁰Ra and ²¹¹Ac sit in loops missing closure by 5–10 MeV
  at 30–70 σ. Each value is individually plausible; only the relationship
  between them is broken.

A flag is not always an error. ²⁴N, ³³Ne and ³⁹Mg are flagged because their
S(n) is slightly negative while they live tens of nanoseconds — but those
values carry the `SY` operator, meaning they are extrapolated from systematics
with an uncertainty that straddles zero. The message says
`(from systematics)` so the two cases are distinguishable: an extrapolation to
treat with care, versus a measured value quoted to ±5 keV with the wrong sign.

`plot_chart(robust=True)` is the default for the same reason — one sign-flipped
record is enough to stretch a colour scale until the real structure washes out.

## Livechart specifics

- The `ground_states` and `levels` endpoints spell shared columns differently
  (`ENSDFauthors` vs `ENSDF_authors`, `unc_md` vs `unc_mn`). Both spellings are
  accepted, but a third would land in `.raw` with the typed field left `None`.
- The `levels` endpoint emits `unc_hl` twice — once for the native unit, once
  for seconds. Repeats are suffixed `.1` in header order rather than silently
  collapsed.
- Livechart applies a RIPL offset (`ripl_shift`) to some floating levels. This
  package does not apply it; the shift is available in `.raw`.
- The neutron's `radius` entry is a mean-square charge radius in fm², not an rms
  radius in fm like every other nuclide. Passed through as given.
- The Livechart paths are tested against captured responses, not live calls.
