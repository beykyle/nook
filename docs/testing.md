# Testing

```
pip install -e ".[dev]"
pytest
```

No network needed: the Livechart transport is injected and the flat-file tests
run against embedded fixed-column fixtures.

## Validation against an independent parser

The values in `tests/data/ensdf180_oracle.json` were produced by
[nudel](https://github.com/op3/nudel), an independently written ENSDF parser.
**nudel is not a dependency** — not required, not optional. Its output was
captured once and committed, so the independence travels with the repository
and `pytest` alone runs the comparison.

This is deliberately not a snapshot of our own output. A self-snapshot detects
*change*; if the parser were wrong when it was taken, the file would preserve
the bug and the test would defend it. These numbers came from a codebase that
does not share our assumptions, so a disagreement means one of the two is
actually wrong.

| file | whose values | what a failure means |
|---|---|---|
| `ensdf180_oracle.json` | nudel's | one of the two parsers is wrong |
| `ensdf180_characterization.json` | ours | our output changed; correctness unaddressed |

The oracle holds per-level fields for a committed excerpt plus a per-dataset
digest over the full A=180 chain — 59 datasets, 3179 levels — in ~11 KB. Set
`ENSDF_PATH` to a directory holding `ensdf.180` and the chain digests are
checked too; otherwise that one test skips.

To refresh it — the only time nudel is ever installed:

```
pip install git+https://github.com/op3/nudel
python tools/generate_golden.py ~/ensdf/ensdf.180
```

The file records nudel's version, the chain file's SHA-256 and the generation
date, so a future disagreement is attributable.

### Bugs this has caught

Neither was reachable by a self-snapshot.

- **Spin-parity inheritance.** ENSDF attaches a parity only where it is
  written; this parser used to infer one across commas, so `1,2+` read as
  `1+,2+`.
- **Suffix energy offsets.** Manual V.18 allows `NUM+A` as well as `A+NUM`, and
  `4172.3+X` was parsing as *no energy at all* — 498 levels in the A=180 chain
  alone.

## Where nudel is wrong

nudel is the oracle, not an authority. Where the two disagree the manual
decides, and in four `J`-field patterns it decides against nudel. Levels
matching these are **held out of the oracle** rather than frozen — enshrining a
value we believe to be wrong would be worse than having none — and our
behaviour is pinned instead by spec-derived tests in `test_nook.py`.
The patterns live in `tests/_fingerprint.py:DIVERGENT_J`.

| `J` field | manual V.20 | nudel | this package |
|---|---|---|---|
| `3+ TO 6-` | case (b): `3+, 4±, 5±, 6-` | `3-,4-,5-,6-` | `3+, 4±, 5±, 6-` |
| `3+ TO 6` | case (c): `3+, 4±, 5±, 6±` | `3+,4+,5+,6+` | `3+, 4±, 5±, 6±` |
| `LE 5+` | "π=+ and J≤5" | no assignment | `5+`, `constraint="LE"` |
| `GE 4` | as above | no assignment | `4`, `constraint="GE"` |

In both range cases nudel propagates one endpoint's parity across the whole
range, where the manual leaves the interior undetermined. For the operator forms
it yields nothing.

Two further cases are losses rather than disagreements: nudel discards `NOT 3-`
and `NATURAL`/`UNNATURAL` entirely, where this package records them in
`SpinParity.excluded` and `SpinParity.natural_parity`. Discarding is safe, so
these stay in the oracle — the fields simply have no counterpart.

## Physics invariants

Some code has no oracle available. nudel does not parse bands at all, so
column 77, the `BAND()` documentation records and the ASCII markup rendering
have nothing external to check them against — and a self-snapshot would only
freeze whatever they currently produce.

Nuclear structure supplies one. Two quasiparticles couple to
K = |Ω_p − Ω_n| or Ω_p + Ω_n — the Gallagher–Moszkowski doublet — with parity
the product of the two orbital parities, which for a Nilsson label [N n_z Λ]
is (−1)^N. Those three pieces of the parser are decoded independently, so if
any one were misread the bandheads would stop satisfying the relations.

All four doublets in 180Ta hold, on both K and parity:

| configuration | bands | K^π | \|Ω_p−Ω_n\| | Ω_p+Ω_n | π_p·π_n |
|---|---|---|---|---|---|
| π7/2[404] ν9/2[624] | A, F | 1+, 8+ | 1 | 8 | + |
| π9/2[514] ν9/2[624] | K, Q | 0−, 9− | 0 | 9 | − |
| π7/2[404] ν5/2[512] | L, J | 1−, 6− | 1 | 6 | − |
| π7/2[404] ν1/2[510] | f, h | 3−, 4− | 3 | 4 | − |

`test_band_assignments_obey_the_coupling_rules` asserts this against the
committed excerpt, and refuses to pass if fewer than four doublets are found —
otherwise a parsing failure that produced no bands at all would look like
success. A companion test feeds it a deliberately wrong bandhead to confirm
the invariant is sensitive.

A second, weaker check: fitting E(J) = E₀ + A[J(J+1) − K(K+1)] over the lowest
members gives A = 10.3–11.1 keV across bands A, K, F and Q. The same deformed
core should give the same A; a few percent of spread is Coriolis mixing, not a
parsing error. This one is left as a note rather than a test, since the
tolerance would have to be loose enough to be uninformative.

## A third kind of oracle: the data checking itself

The mass-surface closure in [limitations](limitations.md#the-mass-surface) is
neither a frozen oracle nor a spec-derived test — it is an identity the data
must satisfy internally, so it validates numbers that no external reference is
available for. Its tests build a synthetic survey from chosen mass excesses, so
closure holds by construction and any failure comes from the code rather than
from the numbers.

## Reconciliations

Two representation differences are normalised in `tests/_fingerprint.py` rather
than in each test:

- nudel spells "no value" as `float('nan')` where this package uses `None`.
  (Normalising also makes the JSON valid — `NaN` is not legal JSON.)
- nudel uses the tropical year (31 556 926 s), this package the Julian year
  (31 557 600 s). That is 21 ppm, far below any quoted half-life precision, so
  half-lives are kept out of the digest and compared with a tolerance.
