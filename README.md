# sampling-rate-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24153259.svg)](https://doi.org/10.5281/zenodo.24153259) [![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/Xbotsd/badge.svg)](https://hub.xerj.org/contributors/Xbotsd)

How much of a feature is measuring the signal, and how much is measuring the
sampling rate?

Given a signal and a sampling rate, this package computes, for eighteen
standard feature definitions, the **paired relative deviation** and descriptive
separation/effect-size summaries produced by a change of digital grid. Its
signature labels are heuristic structural warnings, not universal invariance
orders or formal proofs.

It is the reference implementation accompanying

> Xu, Y., Han, L., and Song, C. *Sampling Rate as an Influence Quantity: A
> Paired Audit of Sampled-Signal Feature Definitions* (substantially revised).

## Status and provenance

- Version 1.0.0. Archived on Zenodo as doi:10.5281/zenodo.24153259 (concept doi:10.5281/zenodo.24123019).
  The accompanying manuscript is under review.
- **Archived on Zenodo: [10.5281/zenodo.24153259](https://doi.org/10.5281/zenodo.24153259)**
  (software, MIT). That record carries two files: this package, and the per-record
  feature values, fit outputs and figures behind every table and figure of the
  manuscript. Cite the DOI for the code; cite the article for the science.
- `10.5281/zenodo.23679114` is an **earlier preprint**, not this code: one
  file, the old title, and claims (asymptotic invariance orders, `r >= 10`,
  "strict lower bound") that the manuscript has since withdrawn. It is
  superseded by the version under review.
- The per-record results are in the archival deposit, not in this git tree.

### Versioning

From 1.0.0 the public surface — `Audit`, `AuditResult`, `REGISTRY`, `ORDER_OF`,
`resample_fft`, `guess_signature` and the `leakage-audit` / `invariance-agent`
entry points — follows semantic versioning: breaking changes bump the major,
new behaviour bumps the minor, fixes bump the patch. The signature labels
returned by the engine are heuristic structural warnings, not invariance
orders; an order is measured, not read off a formula.

---

## Invariance agent (new)

`invariance-agent` wraps the audit into a guided assistant for first-time users
and for programmatic use: it profiles your acquisition settings, plans the
structural class of a feature from its formula, measures the paired deviation on
your own record, and tells you what grid your tolerance needs — or that no grid
will help.

```bash
invariance-agent wizard                                            # guided first run
invariance-agent plan   --expr "zero crossings divided by len(x)"   # class from the formula
invariance-agent full   --npy rec.npy --fs 500 --band 40 --json     # plan + audit + advice
```

It deliberately refuses to print a universal minimum sampling rate, and every
response states that only the digital-grid component is isolated. See
**[AGENT.md](AGENT.md)** for the contract, the verdict table and the integration
API.

---

## Why

Within a declared numerical measurement model, sampling rate can be treated as
an **influence quantity** when changing the digital grid changes the computed
feature while the source record is held fixed. Sample-count normalisers,
difference/crossing estimators and bin-count normalisers show different
finite-range responses. Their magnitude and slope depend on the signal,
implementation, anchor, bandwidth convention and fitted range.

This package tells you which family a feature is in, and how large the error is
on your data.

| signature | expected mechanism | required interpretation |
|---|---|---|
| count or division by sample count | physical scale moves with the grid | redefine in physical units or audit explicitly |
| differences/crossings | often near-quadratic error under smoothness assumptions | fit the actual paired-error curve; no universal safe ratio |
| division by $\log$ FFT-bin count | slowly varying bin normaliser | do not assign a fixed asymptotic power |
| value-only control | no explicit sample-count scale | near-invariance is empirical, not guaranteed |

## Install

```bash
pip install git+https://github.com/Xbotsd/sampling-rate-audit
# or, from a clone:
pip install -e .
```

or, without installing, copy `src/leakage_audit/` next to your script. The only
dependency is `numpy`.

## Thirty-second example

```python
import numpy as np
from leakage_audit import Audit

rng = np.random.default_rng(0)
signals = []
for _ in range(40):
    t = np.arange(5000) / 500.0
    x = np.zeros_like(t)
    for _ in range(25):
        f = rng.uniform(0.5, 20)
        x += rng.rayleigh(1.0) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    signals.append(x)

res = Audit().run_many(signals, fs=500, band=20)
print(res.markdown())
```

```
| feature                  | order | r=3.125        | r=6.25         | r=12.5         | fitted p |
|--------------------------|-------|----------------|----------------|----------------|----------|
| `hjorth_mobility_Hz`     | 2     | 0.320 / 0.26   | 0.360 / 0.07   | 0.400 / 0.02   | 2.01     |
| `hjorth_mobility_sample` | 1     | 1.000 / 86.44  | 1.000 / 44.76  | 1.000 / 21.25  | 1.06     |
| `spec_entropy`           | 0.45  | 1.000 / 14.18  | 1.000 / 10.12  | 0.920 / 6.85   | 0.44     |
| `std`                    | inf   | 0.500 / 0.00   | 0.500 / 0.00   | 0.500 / 0.00   | -        |
```

Each cell is `AUC / d`: the probability that a classifier separates the
down-sampled from the reference representation, and the standardised shift of
the feature value.

`order` is retained as legacy output metadata for broad structural classes;
it must not be interpreted as a universal theorem. `fitted p` is a finite-range
slope measured from the paired relative deviation. **When labels and fitted
slopes disagree, inspect the implementation, anchor, signal and fit range**.

## Command line

```bash
leakage-audit --selftest                                  # verify the engine
leakage-audit --npy signal.npy --fs 500 --band 20
leakage-audit --dir records/ --fs 500 --band 20 --out report/
```

`--band` is the content bandwidth $B$ in Hz. It is omitted only reluctantly:
leave it out and the package estimates the 99 % power bandwidth, which is a
reasonable default but not a substitute for a declaration.

## What `--selftest` shows

Synthetic signals with known ground truth, twenty realisations, reference at
8 kHz:

```
feature                     bias exponent   expected
turning_per_second                   2.48   2
zcr_per_second                       2.58   2
hjorth_mobility_Hz                   2.01   2
spec_entropy                         0.44   0.45
turning_per_sample                   1.02   1
zcr_per_sample                       1.02   1
std                                   inf    inf
```

## Definitions audited

Hjorth mobility and complexity, Katz, Higuchi and Petrosian fractal dimensions,
DFA $\alpha_1$, sample entropy, normalised spectral entropy, zero-crossing and
turning-point rates in both per-sample and per-second forms, spectral centroid,
band-power ratio, standard deviation, skewness and kurtosis.

That is **18 registered definitions**. The accompanying manuscript audits
**16** of them, because the per-sample and per-second forms of a rate are one
definition counted twice here. The implementation distinguishes five
signatures (`PER_SAMPLE`, `N_IN_FORMULA`, `DIFF_COUNT`, `LOGBIN_NORM`,
`VALUE_ONLY`); the manuscript groups the first two into a single
**sample-indexed** class, which is why it reports four structural response
classes.

## Things to know before trusting the numbers

- **It measures resampling, not hardware.** Analog front ends, anti-alias
  filters, quantisation and clock jitter are absent. They can reinforce or
  partially offset the grid component, so the audit isolates one component and
  is **neither an upper nor a lower bound** on the device-level discrepancy.
- **One signal gives bias, not AUC.** AUC needs several records; with a single
  record use `run()` and read the relative bias.
- **`--band` is a declaration, not a measurement.** A smaller $B$ flatters the
  result. Declare it, and prefer a conservative value.
- **Cost.** `sample_entropy` is $O(N^2)$; above about 2 kHz it dominates the
  runtime.

## Citation

See `CITATION.cff`. If you use the package, please cite the article above.

## License

MIT.
