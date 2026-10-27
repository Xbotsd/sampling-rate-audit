# sampling-rate-audit

How much of a feature is measuring the signal, and how much is measuring the
sampling rate?

Given a signal and a sampling rate, this package computes, for eighteen
standard feature definitions, the **relative bias** that a change of sampling
rate produces, the **effect size** of that change, and the **invariance order**
predicted from the definition alone.

It is the reference implementation accompanying

> Xu, Y. *Sampling Rate as an Influence Quantity: Invariance Orders and a
> Definition-Level Audit of Physiological Feature Sets* (under review).

---

## Why

A feature computed from a sampled signal is a measurement result, and the
sampling rate is an **influence quantity**: it is not the measurand, but it
changes the value. For one whole family of features — anything normalised by
the number of samples — the change is a constant factor that no amount of
oversampling removes. For another family it decays as $r^{-2}$, where
$r = f_s / B$ is the sampling rate over the content bandwidth.

This package tells you which family a feature is in, and how large the error is
on your data.

| order | signature | what happens when $f_s$ changes |
|---|---|---|
| 1 | count or division by the sample count | value scales by the rate ratio; **no $f_s$ repairs it** |
| 2 | first differences, crossings, path length | bias decays as $r^{-2}$; safe above $r \approx 10$ |
| ~0.45 | divided by $\log$ of the number of FFT bins | decays only logarithmically; definitional |
| ∞ | algebraic function of sample values | invariant to machine precision |

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

`order` is read off the definition; `fitted p` is measured from the bias. **When
the two disagree, the implementation and the definition are not the same
thing** — which is usually what you want to know first.

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

## Things to know before trusting the numbers

- **It measures resampling, not hardware.** Analog front ends, anti-alias
  filters and clock jitter are absent, and they can only add error. Every
  number is a lower bound.
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
