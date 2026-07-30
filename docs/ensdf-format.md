# ENSDF format notes

What the parsers decode, and the quirks that cost time to discover. Field and
section numbers refer to the ENSDF manual (BNL-NCS-51655).

## Values and uncertainties

Uncertainties follow the significant-digit convention (V.11), including
exponents and asymmetric errors:

```python
parse_value_with_uncertainty("1368.672", "5").symmetric   # 0.005
parse_value_with_uncertainty("1.2E+3", "3").symmetric     # 300.0
parse_value_with_uncertainty("5.62", "+13-9")             # +0.13 / -0.09
parse_value_with_uncertainty("LT 3.0", "").is_limit       # True
```

## Spin and parity (V.20)

The `J` field is a small language, not a number. Every alternative is kept, and
*why* an assignment is soft is recorded:

```python
parse_spin_parity("3/2-").unique          # JPi(two_j=3, parity=-1)
parse_spin_parity("(2+)").tentative       # weak arguments
parse_spin_parity("[0+]").assumed         # from theory
parse_spin_parity("1,2+").candidates      # 1 (no parity), 2+
parse_spin_parity("(3,4)-").candidates    # 3-, 4-
parse_spin_parity("NOT 3-").excluded      # 3- is ruled out
parse_spin_parity("NATURAL").natural_parity
parse_spin_parity("GE 4").constraint      # 'GE'
```

Two rules are easy to get wrong, and getting them wrong is silent:

- **A parity attaches only where it is written.** `1,2+` is "J=1 with unknown
  parity, or 2+" — not "1+ or 2+". The exception is a parity *after a closing
  parenthesis*, which covers the group: `(3,4)-` is 3− or 4−.
- **`NOT` inverts the statement.** Treating `NOT 3-` as an assignment of 3−
  reports the opposite of the data. Those values go to `excluded`, and
  `allows()` honours them.

Ranges follow the manual's three cases: `3 TO 6-` gives every value π=−, while
`3+ TO 6-` keeps the endpoints and leaves the interior undetermined.

Spin is stored as `two_j` so half-integers stay exact; `.j` gives a `Fraction`,
`.j_float` a float.

## Half-lives (V.14)

Time units convert to seconds. Unbound levels quoted as widths convert through
`T = ħ ln2 / Γ`, with the error branches correctly swapped by the reciprocal.
`half_life.from_width` says which case you got; `half_life.width_mev` keeps Γ.

## Energies and floating levels (V.18)

A level whose position relative to the ground state is unknown carries an offset
symbol, and it can appear on **either side** of the number: both `X+1440` and
`4172.3+X` are valid. `level.energy_offset` and `level.is_floating` record it,
and energy filters drop such levels rather than pretending the number is an
excitation energy.

Reading only the prefix form silently discarded 498 energies in the A=180 chain
alone — see [testing](testing.md).

## Continuation records (IV)

Anything that doesn't fit the 80-column primary record lands on a continuation
as `KEY=value$KEY=value`, decoded into `level.properties` / `gamma.properties`:
moments (`MOMM1`, `MOME2`), branchings (`%B-`, `%EC`, `%IT`), shell conversion
coefficients (`KC`, `LC`, …), reduced transition probabilities (`BE2W`), and
`XREF`.

Column 6 marks a continuation; column 7 splits data (blank) from comment
(`c`/`C`). Identification records are not data records — anything scanning raw
lines rather than dataset blocks will misread `180    REFERENCES` as an `R`
record, because `R` lands in column 8.

## Provenance: XREF and X records (V.23, III.B.3)

`XREF` records which datasets reported a level. The symbols are defined by
explicit **`X` records**, not by dataset ordering — the ordering convention
usually holds but is not guaranteed:

```python
scheme.metadata["xref_key"]   # {'I': '181TA(P,D)', ...}
level.observed_in             # resolved dataset names
scheme.seen_in("(P,D)")
```

`XREF=ALL`, `ALL EXCEPT xy` and parenthesised uncertain symbols are handled.

## Bands

Each level carries a one-character band flag in **column 77**, with the meaning
given by a `BAND(x)` documentation record. The `MS` field spans columns
**78–79**, not 78 alone — the 180Ta isomer reads `M1`.

ASCII markup is rendered on the way out: `|p7/2[404]` becomes π7/2[404].

## Gamma placement

`G` records name the level a transition depopulates but not the one it feeds.
Where a continuation states `FL=`, that is used directly; otherwise placement
falls back to matching `E_start − E_gamma` within a tolerance. Ambiguous cases
keep `end_index=None` rather than a guess.

## Q record

Separation energies and decay Q-values in keV, on `scheme.metadata["Q"]`:
`s_n`, `s_p`, `q_beta_minus`, `q_alpha`.
