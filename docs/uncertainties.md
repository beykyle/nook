# Uncertainties

`Uncertain` carries an asymmetric error and an ENSDF operator, because the files
are full of both: `5.62 +13-9`, `LT 3.2`, `GE 5.7`, `SY`.

## Propagation

The two error branches stay separate rather than being symmetrised:

```python
multiply(Uncertain(5.62, 0.13, 0.09), x)   # stays +0.26/-0.18
divide(a, b)                               # denominator branches swap over
add(*terms)                                # assumes independence
```

Limits combine by direction. An upper bound times an upper bound is an upper
bound; strictness is contagious (`LT` × `LE` → `LT`); an upper times a *lower*
bounds nothing and returns an explicit unknown rather than a plausible-looking
number. Quality flags degrade to the weakest claim (`AP` × `SY` → `SY`).

## Blank uncertainties

A blank ENSDF uncertainty propagates as exact, which is the convention for
fields like a normalisation of exactly 1.0. It can also mean "not quoted", in
which case the result looks better determined than it is:

```python
multiply(Uncertain(10.0), Uncertain(2.0, 0.2, 0.2))              # 20 +/- 2
multiply(Uncertain(10.0), Uncertain(2.0, 0.2, 0.2), strict=True) # 20, unknown
Uncertain(10.0).uncertainty_known                                # False
```

## Correlations

Every intensity in a decay dataset is scaled by the same `NR * BR`, so those
errors are fully correlated and per-term quadrature understates the total — by
39% on the 180Lu β⁻ dataset's 33 gammas.

Rather than track correlations, the totals sum first and scale once, which is
exact:

```python
scheme.total_photon_intensity()   # (sum RI) * NR * BR
scheme.total_feeding(kind="B")    # (sum IB) * NB * BR
```

## Why not the `uncertainties` package

It isn't a drop-in: `ufloat` carries a single scalar `std_dev`, so it cannot
represent asymmetric errors or limit operators, and both are ubiquitous in
ENSDF. `ufloat(5.62, (0.13, 0.09))` is a `TypeError`.

Its correlation tracking is better than anything here, so there is an opt-in
bridge (`pip install nook[uncertainties]`):

```python
level.energy.to_ufloat()   # symmetrises; raises on a limit rather than lying
```
