# Invariance agent

`invariance-agent` turns the audit procedure in the paper into something you can
run on your own instrument without reading the paper first. It answers three
questions, in this order:

1. **plan** — Given a feature definition, which structural response class does its
   algebraic form belong to? What should you check before measuring?
2. **audit** — Given one record and its sampling rate, how much does each feature
   move when only the digital sampling grid changes?
3. **advise** — Given the measured curve, what grid does this feature need for a
   tolerance you state — or does no grid help at all?

```bash
pip install -e .

invariance-agent plan    --expr "zero crossings divided by len(x)"
invariance-agent profile --fs 500 --band 40 --window 30
invariance-agent audit   --npy record.npy --fs 500 --band 40
invariance-agent full    --npy record.npy --fs 500 --band 40 --tolerance 0.01
invariance-agent schema                       # machine-readable contract
invariance-agent wizard                       # guided first run
```

> **Version note.** This file is current for **2.1.2**. The archived, citable
> release is still **1.0.1** (concept DOI `10.5281/zenodo.24123019`); 2.1.2 adds
> the order fit and strict mode but is not archived. See
> `RELEASE_NOTES_V2.1.X.md`.

---

## Strict mode (2.1.2)

`Audit(formal_guard=...)` turns the advisory behaviour into a gate. In this mode
the run happens inside a sandbox and every fitted curve is checked at the lowest
audited rate.

```python
from leakage_audit import Audit
from leakage_audit.formalguard import FormalGuard, DeadlockError, Sandbox

guard = FormalGuard(tolerance=0.01, state_path=".sra_deadlock")
engine = Audit(formal_guard=guard, tolerance=0.01)

try:
    result = engine.run(record, fs=500, band=20, rates=[250, 125, 62.5])
except DeadlockError:
    # The sandbox is already gone; result was never returned.
    # Nothing outside the sandbox was written.
    raise
```

### What a violation does

1. Appends one JSON line to `.sra_deadlock`: timestamp, reason, tolerance,
   context, version, and a host label.
2. Raises `DeadlockError` — a `RuntimeError` whose message is exactly
   `SRA_DEADLOCK_PROTECTION_TRIGGERED`.
3. The sandbox is rolled back before the exception leaves `Audit.run`.

Negative, `NaN`, `Inf` and non-numeric inputs count as the **highest** risk.
A failed write to the event file does **not** turn the violation into a pass.

### Protocol gate

```python
from leakage_audit.formalguard import require_agreement

require_agreement()          # prints the protocol, requires the phrase
```

A confirmation completed in under **3 seconds** raises
`RuntimeError("Scripting detected. Protocol reading time too short.")`.
A successful confirmation is cached in `.sra_agreement`; later runs skip the
prompt. With no terminal, the function raises a `RuntimeError` that names the
way out (`--i-understand-risks`) instead of an `EOFError`.

### CLI exit codes

| code | meaning |
|---|---|
| 0 | completed |
| 1 | data / parameter error (missing file, bad array, no arm left) |
| 2 | protocol refused (no terminal, too fast, wrong phrase) |
| 3 | deadlock triggered; the sandbox was rolled back |

```bash
leakage-audit --npy rec.npy --fs 500 --band 20 --rates 250 125 62.5 \
              --strict --tolerance 0.01          # prompts, then may exit 3
leakage-audit ... --strict --i-understand-risks  # skips the prompt, keeps the gate
```

### What strict mode is not

It does not stop a script, identify a machine, restore files outside the
sandbox, or make an extrapolation true. The host id is `uuid.getnode()`, a
MAC-derived label (`host_id=False` disables it). The sandbox is a temporary
directory on disk, not memory.

---

## What it will not tell you

The tool is built to refuse three claims that the underlying experiment cannot
support. If you see it decline to give a number, that is the intended behaviour:

| It will not claim | Because |
|---|---|
| a universal invariance order | The order is a finite-range empirical exponent, not a theorem. Bin-normalised entropy is not a power law at all. |
| a minimum safe sampling rate | A single ratio cannot hold for every feature. The rate you need is a property of your definition and your tolerance. |
| that your instrument is off by a given amount | The paired design isolates the **digital-grid component**. Analog front ends, filters, quantisation and clock jitter are not modelled. |

Every response carries the scope statement, and `advise` marks extrapolated
ratios as extrapolated.

---

## The four structural classes

The class is a **prior** read from the formula. It tells you what to check first;
only the paired measurement decides.

| Signature | Class | Typical response | Does a higher rate help? |
|---|---|---|---|
| `PER_SAMPLE` | sample-indexed | constant scale factor, bias independent of ratio | **no** |
| `N_IN_FORMULA` | sample-count scale | estimator's physical scale moves with the grid | **no** |
| `DIFF_COUNT` | difference or crossing | approximately quadratic decay where the approximation holds | yes, follow the curve |
| `LOGBIN_NORM` | bin-normalised entropy | slowly varying, no fixed exponent | yes, but weakly |
| `VALUE_ONLY` | grid-stable (prior) | near zero, unless preprocessing introduces dependence | yes, verify |

A worked contrast from a synthetic 2 kHz record (`--fs 2000 --band 40`):

```
hjorth_mobility_Hz   measured exponent 2.005   met tolerance 1% from r = 12.5
zcr_per_sample       measured exponent 1.073   deviation stays at 7.0  -> no ratio helps
```

Both features "depend on the sampling rate". Only one of them can be fixed by
sampling faster.

---

## Wiring your own acquisition device

`profile` checks the two numbers the rest of the analysis depends on:

```bash
invariance-agent profile --fs 500 --band 40 --window 30 --lpf 150
```

It reports the ratio `r = fs/B`, whether `fs/2 > B`, and warns when the analog
low-pass sits above Nyquist or below the band you declared. Both warnings mean
the declared band is not the band present in your data — fix that before
auditing anything.

Notes on the inputs:

- **`fs`** is a property of your device. It should come from the acquisition
  settings, not from the file metadata alone.
- **`B` is a claim, not a measurement.** Declaring a small band inflates `r` and
  makes your design look safer than it is. State the physical rationale.
- **`window_s`** matters for estimators whose parameters are expressed in
  samples: the same nominal window is a different physical duration at a
  different rate.

---

## Reading the advice

`advise` returns one of four verdicts per feature:

| Verdict | Meaning | What to do |
|---|---|---|
| `met within audited range` | The tolerance was reached at a ratio you actually measured | Report that ratio |
| `no ratio helps` | The bias is a scale factor | Redefine in physical units, or do not pool across devices |
| `extrapolated beyond audited range` | A fitted curve predicts tolerance at a higher ratio | Verify before relying on it |
| `unresolved` | Above tolerance, and the curve has no usable slope (often fewer than three audited ratios) | Extend the audited range |

`unresolved` is common when your device already runs near the ratio you care
about: the audit can only step *down* from your native rate. To fit an exponent
you need at least three ratios, which usually means a record captured at a
higher rate than the one you plan to deploy.

---

## Calling it from another program

Two integration surfaces.

**Python**

```python
from leakage_audit.agent import run_task, dumps

out = run_task({
    "action": "full",
    "device": {"fs_hz": 500, "band_hz": 40, "window_s": 30},
    "data": {"path": "record.npy"},
    "tolerance": 0.01,
})
print(dumps(out))          # NaN/Inf become null, so any JSON parser accepts it
```

**Shell**

```bash
invariance-agent full --npy record.npy --fs 500 --band 40 --json
invariance-agent schema            # the contract, for tool/function calling
```

The contract is stable and small:

```jsonc
{
  "action": "plan | profile | audit | advise | full",
  "feature": "zcr_per_sample",              // or "feature_expr": "free text"
  "device": {"fs_hz": 500, "band_hz": 40, "window_s": 30, "analog_lpf_hz": 150},
  "data":   {"path": "record.npy"},
  "rates":  [3.125, 6.25, 12.5],            // optional
  "features": ["zcr_per_second"],           // optional subset
  "tolerance": 0.01
}
```

`describe()` (or `invariance-agent schema`) returns the full field list, the
class table and the guarantees, in one JSON object.

---

## Data formats

- `.npy` — one-dimensional float array of samples
- `.csv` / `.txt` — single column of samples

Multi-channel records must be split into one file per channel; channel 0 alone
is a common source of false "invariance".

---

## Limitations you should carry into your report

1. The audit is a paired resampling of one digital record. It is not a comparison
   of two physical instruments.
2. A single record gives a deviation estimate, not a population margin. Repeat on
   several records before quoting a number to a co-author.
3. An interpolated high-rate anchor adds no physical information; it only
   provides a common dense grid.
4. Count-based estimators plateau, so a fitted exponent can stop predicting before
   the ratio you care about.
