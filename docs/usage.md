# Usage

## Querying a level scheme

`LevelScheme` is immutable; every query returns a new one. Gammas follow the
level they depopulate, so filtering never leaves an orphan.

```python
scheme.below(8000)                    # energy cutoff, drops floating levels
scheme.with_known_jpi(allow_tentative=False)
scheme.isomers(min_half_life_s=1e-9)
scheme.complete_up_to()               # RIPL-style discrete-level cutoff:
                                      # truncate at the first level lacking a
                                      # firm E, J and parity
scheme.filter(lambda lv: lv.spin_parity.matches(two_j=4, parity=+1))
scheme.seen_in("(P,D)")               # only levels a transfer reaction observed
scheme.decays_from(2)
scheme.to_records()                   # list of flat dicts
scheme.to_dataframe()                 # needs pandas
```

Level indices are stable, so a kept gamma can point at a level a filter
removed. `level_by_index` returns `None` there rather than inventing one.

### Rotational bands

```python
for label, band in scheme.bands().items():      # ordered by bandhead
    print(label, scheme.band_definition(label))
    # 'A  Kπ=1+ band. Configuration=π7/2[404]ν9/2[624]'
```

Not every band flag has a `BAND()` definition — 180Ta uses 27 and documents 24
— so `band_definition()` can return `None` for a group `bands()` gave you.
Labels are case-sensitive: `b` and `B` are different bands.

## Ground-state properties

Livechart aggregates several compilations besides ENSDF:

```python
gs = nook.ground_state("180Ta")
gs.mass_excess          # AME, keV
gs.atomic_mass          # AME, micro-u  (not u)
gs.abundance            # NUBASE, % mole fraction
gs.discovery_year       # NUBASE
gs.charge_radius        # radii compilation, fm
gs.magnetic_dipole      # moment tables, nuclear magnetons
gs.electric_quadrupole  # moment tables, barn
gs.s_n, gs.s_p, gs.q_alpha, gs.q_ec, gs.q_beta_minus   # AME, keV
```

These don't share a provenance — a mass excess and a charge radius for the same
nuclide come from different groups with different vintages — so attribution
stays explicit:

```python
gs.provenance("mass_excess")    # 'AME (atomic mass evaluation)'
gs.units("atomic_mass")         # 'micro-u'
nook.EVALUATIONS, nook.UNITS        # the full mappings
```

A quantity the compilations don't carry stays `None`, never `0`.
`mass_from_systematics` flags extrapolated masses.

`ground_state(..., source="ripl3")` serves the sparser RIPL-3 tables instead:
mass excess and abundance, with FRDM95/HFB-14 theory values in `metadata`.
The flat files carry none of this; the electromagnetic moments are the
exception — they're on `Level` from both ENSDF backends, read from the
`MOMM1`/`MOME2` continuations for flat files.

## RIPL-3

`source="ripl3"` reads the committed mirror under `data/ripl3/` (or
`$RIPL_PATH`), and `Ripl3Source` exposes the non-level segments — masses,
resonances, gamma strength, optical potentials, level densities, fission
barriers. `nook.compare` matches levels and masses across sources. See
[ripl3.md](ripl3.md).

## Decay data

Decay records live in `DecayScheme` rather than on `Level`, because a feeding
intensity belongs to a *decay*: the same level carries different feedings
depending on which parent populated it.

```python
for scheme in nook.decay_schemes("180Hf"):
    print(scheme.dsid, scheme.parent)
    for feeding, absolute in scheme.absolute_feedings():
        print(feeding.label, feeding.intensity, absolute, feeding.log_ft)
```

`P` (parent), `N` (normalisation), `B` (β⁻), `E` (EC/β⁺), `A` (α) and `D`
(delayed particle) records are decoded. Feeding records attach to the level they
populate — in ENSDF they *follow* that level's own records.

**The `N` record is the point.** ENSDF intensities are relative to whatever the
evaluator called 100 within one dataset, so they aren't comparable across
datasets until scaled:

```python
scheme.absolute_photon_intensity(gamma)      # RI * NR * BR, per 100 decays
scheme.total_photon_intensity()              # correlations handled
scheme.total_feeding(kind="B")
```

If a dataset has no `N` record the result is an explicit unknown — assuming
`NR = 1` would fabricate absolute intensities.

## Caching

Livechart responses cache under `$NOOK_CACHE` (default
`~/.cache/nook`). A cached tree can be committed alongside a paper's
input decks, which is the point: a calculation should be able to re-pull
identical bytes a year later.

```python
nook.level_scheme("56Fe", cache=nook.Cache("./ensdf-cache"))
nook.Cache().clear()
```

Parsed mass chains are cached in memory on `(path, size, mtime)` — six nuclides
from one chain went from ~1.9 s to 22 ms. `clear_chain_cache()` forces a re-read.

## CLI

`nook <nuclide>` is shorthand for `nook levels <nuclide>`.

```
nook 24Mg --below 10000
nook 24Mg --source file --path ~/ensdf --dataset "(P,T)"
nook 208Pb --complete --json
nook levels 24Mg --source ripl3 --nmax
nook ripl masses 24Mg
nook ripl resonances 27Al
nook compare 24Mg --what levels --below 8000
nook compare 180Ta --what masses
```
