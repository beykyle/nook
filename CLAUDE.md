# Notes for working on this package

Context a fresh session would otherwise have to rediscover. Most of it is
constraints that look like they could be simplified away and should not be.

## Getting set up

```
pip install -e '.[dev,plot]'
pytest                              # ~5 s, no external data needed
```

Most tests run against fixtures committed under `tests/data`. Two things need a
local ENSDF distribution:

```
export ENSDF_PATH=~/ensdf            # directory of ensdf.001 ... ensdf.299
pytest                               # now also runs the chain-level tests
```

Mass-chain files come from <https://www.nndc.bnl.gov/ensdfarchivals/>. They are
**not** bundled — 220 MB, and they are somebody else's data.

RIPL-3, by contrast, **is** bundled: a full mirror under `data/ripl3/`
(~1.2 GB extracted, excluded from build artifacts), populated and verified by
`tools/fetch_ripl3.py`. RIPL tests gate on the mirror being present the same
way chain tests gate on `ENSDF_PATH`; `$RIPL_PATH` overrides the location.

`test_every_decay_scheme_in_several_chains_draws_cleanly` takes about two
minutes and is the only slow test. `pytest -k "not several_chains"` skips it.

## Layout

| path | what |
|---|---|
| `src/nook/sources/` | the three backends: flat files, the IAEA API, RIPL-3 |
| `src/nook/sources/ripl3/` | one module per RIPL segment behind the `Ripl3Source` facade |
| `src/nook/quantities.py` | the ENSDF value grammar; the fiddliest module |
| `src/nook/compare.py` | cross-source matching for levels and masses |
| `src/nook/survey.py` | ground states across many chains, plus consistency checks |
| `src/nook/repair.py` | verified correction of transcription errors, off by default |
| `src/nook/plotting/` | figures; optional, needs matplotlib |
| `tools/generate_golden.py` | run by hand to refresh the test oracle |
| `tools/fetch_ripl3.py` | populate/verify the RIPL-3 mirror; `data/ripl3/MANIFEST.json` |

## Things that look wrong but are not

**Parity attaches only where written.** `1,2+` is "J=1 with unknown parity, or
2+", not "1+ or 2+". The exception is a parity after a closing parenthesis,
which covers the group. This is ENSDF manual V.20 and it is easy to "fix" back
into a bug. `NOT 3-` goes to `SpinParity.excluded` and must never be reported
as an assignment.

**Two Q records per dataset is normal.** NNDC prepends one from the newest mass
evaluation, then a comment, then the older set the evaluator used. We take the
first. Taking the last silently substitutes AME2003 for AME2012 across ~1600
nuclides.

**Energy offsets can be a suffix.** Both `X+1440` and `4172.3+X` are valid
(V.18). Reading only the prefix form drops 498 energies in A=180 alone.

**A blank uncertainty is treated as exact**, which is the convention for a
normalisation of 1.0 but not always right. `strict=True` refuses instead.

**Totals scale once, not per term.** Every intensity in a decay dataset shares
`NR * BR`, so those errors are fully correlated; per-term quadrature
understates the total by ~40%.

**RIPL level-density files disagree on units by design.** EGSM ships D0 in
keV, bfm/ctm/hfm in eV; the parser normalises all four to `spacing_kev` and a
test asserts they agree. Do not "fix" one file's scale factor to match another.

**RIPL densities are keyed by the compound nucleus.** The row for the n+56Fe
system lives under 57Fe. Resonances are the exception: filed under the target.

**RIPL spins can be inventions.** Levels flagged `g`/`c`/`n` carry a spin the
RIPL evaluators *estimated* (from gamma cascades, a listed set, or the pure
spin distribution); the parser marks these `assumed` and keeps ENSDF's
original J-field string in `Level.raw`. Never report an assumed spin as
ENSDF's. The `u` flag is different: it marks ENSDF's own single candidate
(every firm unique assignment below Emax is u-flagged, including 24Mg's 0+
ground state), so `u` is *not* marked `assumed` -- only its tentativeness
survives, in `tentative`. `g` appears on 5000+ levels but in no readme.

## Testing philosophy

Three kinds of check, and they are not interchangeable:

1. **A frozen independent oracle.** `tests/data/ensdf180_oracle.json` holds
   *nudel's* values, not ours. Do not regenerate it from our own output — that
   would turn validation into a snapshot that defends whatever bug is present.
   nudel is not a dependency; only `tools/generate_golden.py` touches it.
2. **Spec-derived tests** for the four `J`-field patterns where nudel is wrong.
   Those levels are held out of the oracle deliberately.
3. **Physics invariants** where no oracle exists — Gallagher-Moszkowski
   coupling for bands, mass-surface closure for Q-values. These validate code
   that nothing external covers.

The RIPL levels parser gets its independent oracle a different way: RIPL
levels derive from ENSDF, and the ENSDF flat-file chain shares no code with
the RIPL chain, so `test_compare_24mg_ripl_vs_ensdf_adopted_matches_low_lying`
checks both end to end. `tests/data/ripl3_characterization.json` freezes
counts and first excited energies for 20 nuclides to catch parser drift — it
*is* our own output, so it guards against regression, not wrongness.

Figure tests measure rather than eyeball: render, then check that no two labels
overlap and that nothing bare sits on a stroke. Rule-based placement passed
while 8 of 233 real decay schemes still clashed, which is why
`separate_labels()` exists. If you change figure layout, run the chain sweep.

## Repair is deliberately constrained

`repair()` accepts only `negate` and powers of ten, only when a mass-surface
loop closes that did not before, only for quantities an identity constrains,
never for `SY`/`CA` values, and records every change. Loosening any of those
turns it from correction into invention. `docs/suspicious-entries.md` documents
what it declines to touch and why.

## Known gaps

- Livechart paths are tested against captured responses, not live calls.
- `repair()` is greedy, so the repair set is not provably minimal.
- Closure localises an error to a loop, not always to a value.
- Decay records `N`/`P`/`B`/`E`/`A`/`D` are parsed; `R` only via
  `parse_references`.
- Optical potentials are parsed as records; nothing evaluates V(r).
- `masses/gs-deformations-exp.dat` is a dead link on the IAEA index (404 as of
  2026-07) and is deliberately absent from the mirror.
