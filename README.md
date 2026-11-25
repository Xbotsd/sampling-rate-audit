# sampling-rate-audit

> [!WARNING]
> **Version 3.1 adds an optional search layer. It does not turn a prediction
> into a measurement.**
>
> - **2.1.2 added a fitted decay curve and an optional strict mode. Neither
>   turns an extrapolation into a measurement.**
>
> - **The fitted exponent is descriptive.** `d = C·r^(−p)` is fitted to the rate
>   cells you actually audited. Outside that range it is an extrapolation, and
>   extrapolating a power law past its data is exactly the mistake this package
>   exists to warn about.
> - **Strict mode raises; it does not protect.** `--strict` aborts the run and
>   appends an event to `.sra_deadlock` when a fitted curve predicts a deficit
>   above your tolerance. It cannot make the prediction true, and it cannot stop
>   a script — the protocol reading-time check is a process requirement, not a
>   security control.
> - **Rollback means the sandbox.** A deadlock discards the temporary directory
>   this run created. It does not and cannot undo anything you did elsewhere;
>   nothing outside the sandbox is ever written.
> - **The archived release is still `v1.0.1`**
>   ([doi:10.5281/zenodo.24154075](https://doi.org/10.5281/zenodo.24154075)).
>   Version 2.1.2 is not archived and has no DOI.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24123019.svg)](https://doi.org/10.5281/zenodo.24123019) [![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/Xbotsd/badge.svg)](https://hub.xerj.org/contributors/Xbotsd)

How much of a feature is measuring the signal, and how much is measuring the
sampling rate?

Given a signal and a sampling rate, this package computes, for eighteen
standard feature definitions, the **paired relative deviation** and descriptive
separation/effect-size summaries produced by a change of digital grid. Its
signature labels are heuristic structural warnings, not universal invariance
orders or formal proofs.

It is the reference implementation accompanying

> Xu, Y., Han, L., and Song, C. *An audit protocol and reference implementation
> for sampling-grid sensitivity in physiological signal features* (under review,
> Computer Methods and Programs in Biomedicine).

## Status and provenance

- **The audited release is version 1.0.1**, archived as doi:10.5281/zenodo.24154075.
  The concept doi `10.5281/zenodo.24123019` always resolves to the latest archived
  version. The accompanying manuscript is under review at Computer Methods and
  Programs in Biomedicine.
- **`main` reports version 2.1.2 (Formal Guard Edition), `DEMO_BUILD = False`.**
  2.1.2 adds the order fit and strict mode on top of the unchanged 1.0.1 engine.
  It is tagged `v2.1.2` but **not archived**; 1.0.1 remains the citable artefact.
  See [`RELEASE_NOTES_V2.1.X.md`](RELEASE_NOTES_V2.1.X.md).
- **Zenodo holds two archived versions of this package**: 1.0.0
  ([10.5281/zenodo.24153259](https://doi.org/10.5281/zenodo.24153259)) and 1.0.1
  ([10.5281/zenodo.24154075](https://doi.org/10.5281/zenodo.24154075)); the
  concept DOI `10.5281/zenodo.24123019` resolves to the latest of them. Each
  record carries two files: this package, and the per-record feature values, fit
  outputs and figures behind every table and figure of the manuscript. Cite the
  DOI for the code; cite the article for the science. **2.1.2 is not there and
  will not be** — it stays a GitHub release only.
- `10.5281/zenodo.23679114` is an **earlier preprint**, not this code: one
  file, the old title, and claims (asymptotic invariance orders, `r >= 10`,
  "strict lower bound") that the manuscript has since withdrawn. It is
  superseded by the version under review.
- The per-record results are in the archival deposit, not in this git tree.

## What 2.1.2 adds (and what it refuses to claim)

| module | what it does |
|---|---|
| `leakage_audit/estimator.py` | fits `d = C·r^(−p)` to the paired deficits: `nls` (linear space), `mle` (log-normal, profiled σ), `huber` (robust, MAD-scaled) |
| `leakage_audit/demoguard.py` | compares a predicted deficit with your tolerance and **prints**; returns `"SAFE"` / `"WARNING"`; never raises, never blocks, never writes |
| `leakage_audit/formalguard.py` | strict mode: a tolerance violation appends an event to `.sra_deadlock` and raises `DeadlockError`; `require_agreement()` gates the run; `Sandbox` discards this run's temporary output on failure |
| `leakage_audit/visualizer.py` | scatter + fitted curve + green/amber/red bands, with a marker where the audited range ends |
| `Audit(fit_order=..., estimator_method=..., formal_guard=...)` | attaches the fits to `AuditResult.fits`; with a guard, the run is sandboxed and every fitted curve is checked at the lowest audited rate |

```bash
pip install -e ".[demo]"                 # numpy + scipy + matplotlib
python examples/v2_demo_tutorial.py      # end-to-end, writes v2_demo_decay.png
pytest -q                                # 100 tests

# strict mode
leakage-audit --npy rec.npy --fs 500 --band 20 --rates 250 125 62.5 \
              --strict --tolerance 0.01
#   ...  protocol prompt  ...
#   or, for scripted use, --i-understand-risks (records the acknowledgement,
#   keeps the sandbox and the guard)
```

### Exit codes

| code | meaning |
|---|---|
| `0` | completed |
| `1` | data / parameter error: missing file, malformed array, no arm left after the Nyquist rule |
| `2` | protocol refused: no terminal, confirmation under 3 s, or the phrase did not match |
| `3` | deadlock triggered: the fitted curve predicted a deficit above the tolerance; the sandbox was rolled back |

The constants live in `cli.py` (`EXIT_OK`, `EXIT_DATA`, `EXIT_PROTOCOL`,
`EXIT_DEADLOCK`) and are covered end to end.

### What strict mode is not

| claim | reality |
|---|---|
| "it stops a script" | it does not. The 3-second reading check is a process requirement; a script can sleep 4 seconds. |
| "it locks the machine" | it does not. The event file is an append-only log, not an interlock; nothing reads it back. |
| "it rolls back your data" | it discards *this run's* sandbox. Files you wrote elsewhere are untouched, and cannot be restored by this package. |
| "it proves the extrapolation" | it does not. A fit between r=2.5 and r=12.5 says nothing about r=1. Re-measure. |


The exponent is a **finite-range descriptive fit**. `PowerLawFit.note` says so
when a fit rests on two points, and `DemoGuard`'s message says so every time it
warns. Neither of them is a measurement of anything outside the rate cells you
actually audited.

### Versioning

From 1.0.0 the public surface — `Audit`, `AuditResult`, `REGISTRY`, `ORDER_OF`,
`resample_fft`, `guess_signature` and the `leakage-audit` / `invariance-agent`
entry points, and from 3.1.1 the `sra_opt` package and the `sra-opt` / `sra-opt-dp`
scripts — follow semantic versioning: breaking changes bump the major,
new behaviour bumps the minor, fixes bump the patch. The signature labels
returned by the engine are heuristic structural warnings, not invariance
orders; an order is measured, not read off a formula.

---

## Optimiser layer (3.1)

The audit answers *how much a definition moves when the grid changes*. The optimiser answers
the question a designer actually has: *which grid should I build*, given a tolerance and a
price per sample. It is an optional extra and it never replaces the audit's own verdicts.

```python
from sra_opt import Oracle, CostModel
from sra_opt.pareto import search

oracle = Oracle(["hjorth_mobility_Hz"], n_records=6, seed=7)
report = search(oracle, "hjorth_mobility_Hz", tau=0.02,
                cost=CostModel(fixed=1.0, per_unit_rate=1.0), budget_units=400)
print(report.verdict())
# a=25, r*=7.0 (device cost 8.0) certifies tau=0.02 from 7 measured ratios;
# 39% of the defensive sweep's compute
```

```bash
sra-opt search --feature hjorth_mobility_Hz --tau 0.02 --budget 400
sra-opt-dp --feature hjorth_mobility_Hz --tau 0.02 --export   # exit 0
sra-opt-dp --tau -1                                           # exit 3
```

| module | what it does | the part that keeps it honest |
|---|---|---|
| `sra_opt.surrogate` | fits the deficit curve on a five-point pilot | refuses any extrapolation beyond 2x whatever the fit quality; lowering the acceptance threshold writes a signed risk agreement instead of passing silently |
| `sra_opt.pareto` | predicts the crossing, measures it, returns a certificate | every ratio is labelled *measured*, *interpolated* or *extrapolated*; when the cheapest passing grid is not measured it returns a `ValueChoice` instead of a verdict |
| `sra_opt.deadlock` | the export gate, built on `formalguard` | dirty inputs and unbacked exports exit 3; exploration inside a search rolls back and is not a violation |
| `sra_opt.arbitrator` | reconciles the cheap and expensive tracks, and reads a foreign result | nine training cases, so it exposes disagreement and does not settle it, and says so |
| `sra_opt.visualizer_v3` | `.sra_bug_report`, defect heatmap, leave-one-out sensitivity | figures need the `opt` extra; the report replays byte for byte |

Two results from the demo, because they are not what the design expected:

* **the expensive track is not automatically the better one.** A fourteen-point sweep spent
  half its budget where the deficit had already died into noise, so its "first ratio that meets
  the tolerance" was biased low; the cheap five-point prediction was closer in 7 of 9 cases.
* **the surrogate gate earned its place.** A five-point PCHIP fit scored a leave-one-out R2 of
  0.99996 and was still wrong by 4x four octaves out. Fit quality alone would have passed it;
  the extrapolation rule refused it.

The optimiser inherits every limitation of the audit: it measures the digital-grid component
only, its anchor is an interpolation rather than continuous-time truth, and a certificate says
`measured` only when the ratio behind it was actually paid for.

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
