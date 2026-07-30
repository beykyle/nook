# nook

A nook for nuclear structure data: level schemes from ENSDF and RIPL-3, with
the awkward parts — significant-digit uncertainties, spin-parity alternatives,
widths masquerading as half-lives, Fortran fixed-format files — parsed into
real objects instead of strings.

No required dependencies. `pandas` is optional, for `.to_dataframe()`.

Built in claude code using Claude Opus 5 with pretty minimal prompting. 

Tested against the IAEA Livechart API, the RIPL-3 mirror, and the NNDC flat-file archive.

```python
import nook

scheme = nook.level_scheme("24Mg")          # adopted levels, via the IAEA API
for lv in scheme.below(10_000):
    print(lv.energy_kev, lv.spin_parity, lv.half_life.seconds.value)
```

## Three backends, one data model

| | `livechart` (default) | `file` | `ripl3` |
|---|---|---|---|
| Source | IAEA Livechart API | NNDC archival flat files | RIPL-3 mirror on disk |
| Coverage | **adopted levels only** | every evaluated dataset | levels + reaction inputs |
| Needs network | yes (responses cached) | no | no |
| Setup | none | download mass chains, set `$ENSDF_PATH` | committed mirror, or `$RIPL_PATH` |

All three return the same `LevelScheme`, so downstream code doesn't branch.
For a specific evaluation rather than the adopted set:

```python
scheme = nook.level_scheme("24Mg", source="file", dataset="(P,T)")
```

Flat files come from <https://www.nndc.bnl.gov/ensdfarchivals/> — one per mass
number, `ensdf.024`. Extract them somewhere and export `ENSDF_PATH`.

RIPL-3 goes beyond levels: masses (experimental and theory), average resonance
parameters, giant dipole resonances and gamma strength, optical-model
potentials, level densities and fission barriers, all parsed from the mirror
under `data/ripl3/`. Where sources overlap, `nook.compare` matches them:

```python
result = nook.compare.levels("24Mg", sources=("file", "ripl3"))
result.rms_delta_kev            # 1.5 keV over 72 matched levels
nook.compare.masses("180Ta")    # AME vs RIPL vs FRDM95 vs HFB-14
```

## Documentation

- **[Figures](docs/figures.md)** — publication-quality level, band, decay and
  chart-of-nuclides drawings, plus the RIPL-3 suite: strength functions, level
  densities, fission barriers, mass residuals, resonance systematics
- **[Usage](docs/usage.md)** — querying schemes, ground-state properties, decay
  data, caching, the CLI
- **[RIPL-3](docs/ripl3.md)** — the mirror, the seven segments, units, and
  cross-source comparison
- **[ENSDF format](docs/ensdf-format.md)** — what the parsers decode, and the
  file quirks worth knowing about
- **[Uncertainties](docs/uncertainties.md)** — asymmetric errors, limit algebra,
  correlations, and why the `uncertainties` package isn't used
- **[Testing](docs/testing.md)** — validation against an independently written
  parser, and where the two disagree
- **[Suspicious entries](docs/suspicious-entries.md)** — where the evaluated
  files contradict themselves, and what was done about it
- **[Limitations](docs/limitations.md)** — what isn't parsed, and known
  approximations

## Related work

[nudel](https://github.com/op3/nudel) is a more complete flat-file parser
(GPL, local files only) and is used here as a test oracle.
[PyNE](https://pyne.io) covers decay data inside a much larger toolkit.
[TkN](https://tkn.in2p3.fr) is a C++ interface over the same IAEA API. This
package is deliberately smaller, aimed at getting a clean, queryable level
scheme out of either source.

## Working on it

See [CLAUDE.md](CLAUDE.md) for setup, layout, and the constraints that look
optional but are not.

## Install

With [uv](https://docs.astral.sh/uv/):

```
uv sync --extra dev        # creates .venv with the package and dev extras
uv run pytest              # ~5 s against committed fixtures
uv run nook 24Mg           # the CLI
```

Extras: `plot` (matplotlib figures), `pandas` (`.to_dataframe()`),
`uncertainties`; `dev` pulls in all of them plus pytest. To use the package
elsewhere, `uv pip install -e '/path/to/nook[plot]'` into that
environment.

With pip:

```
pip install -e '.[dev]'
pytest
```

Tests needing local flat files skip unless `$ENSDF_PATH` is set — see
[CLAUDE.md](CLAUDE.md).
