"""Invariance agent: plan -> audit -> advise, in one callable interface.

This module turns the paper's audit procedure into something a first-time user —
or another program — can drive without reading the paper. It answers three
questions in order:

1. **plan**   Given a feature definition, which structural response class does
   its algebraic form belong to, and what should we expect before measuring?
2. **audit**  Given one record and its sampling rate, how much does each feature
   move when only the digital grid changes?
3. **advise** Given the measured curve, what grid does this feature need for a
   stated tolerance - or does no grid help?

Honesty constraints carried over from the paper, and enforced in the output:

* The audit isolates the **digital-grid component** only. Analog filtering,
  quantisation, clock jitter and vendor processing are not modelled. The
  numbers are neither a lower nor an upper bound on the difference between two
  physical instruments.
* The structural class is a **prior**, not a theorem. It tells you what to check
  first; only the paired measurement decides.
* There is no universal over-sampling ratio. Any ratio this module reports is
  extrapolated from the user's own measured curve over the range actually
  audited, and is flagged as an extrapolation.

Typical use::

    from leakage_audit.agent import run_task

    out = run_task({
        "action": "full",
        "feature_expr": "count of zero crossings divided by len(x)",
        "device": {"fs_hz": 500, "band_hz": 40, "window_s": 30},
        "data": {"path": "record.npy"},
        "tolerance": 0.01,
    })

The same task can be run from the shell::

    invariance-agent full --expr "count of zero crossings / len(x)" \\
        --npy record.npy --fs 500 --band 40 --tolerance 0.01
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import ORDER_OF, REGISTRY, Audit, estimate_band, guess_signature

__all__ = [
    "STRUCTURAL_CLASSES",
    "plan",
    "device_profile",
    "audit_array",
    "required_ratio",
    "advise",
    "run_task",
    "describe",
]

# --------------------------------------------------------------------------
# Structural classes
# --------------------------------------------------------------------------
# Each entry maps one signature bit to the class the paper describes, the
# expected behaviour, and the action a user should take *before* measuring.
STRUCTURAL_CLASSES = {
    "PER_SAMPLE": {
        "class": "sample-indexed",
        "expected": (
            "Defining quantity is normalised by a sample count, so the value carries a "
            "factor 1/N. Changing the grid changes it by a constant factor that no higher "
            "rate removes; the bias is independent of the over-sampling ratio."
        ),
        "prior_exponent": 1.0,
        "before_measuring": (
            "Check whether the denominator is a count of samples or a duration. If it is a "
            "count, redefine in physical units or do not compare this feature across devices."
        ),
        "ratio_helps": False,
    },
    "N_IN_FORMULA": {
        "class": "sample-count scale",
        "expected": (
            "The sample count or a sample-scale lag enters the formula directly (fractal "
            "dimensions, DFA, entropy estimators). The physical scale of the estimator moves "
            "with the grid, and the response depends on the definition's own parameters."
        ),
        "prior_exponent": 1.0,
        "before_measuring": (
            "Express every lag, box size and threshold in physical units, and audit the "
            "feature across the ratios you actually intend to pool."
        ),
        "ratio_helps": False,
    },
    "DIFF_COUNT": {
        "class": "difference or crossing",
        "expected": (
            "Built from finite differences or from counts per unit time. For a smooth, "
            "band-limited record the leading error term is quadratic, so the relative "
            "deviation falls roughly as r^-2 over the range where the approximation holds."
        ),
        "prior_exponent": 2.0,
        "before_measuring": (
            "Nothing needs redefining, but the usable ratio depends on your tolerance; "
            "measure the curve rather than assuming an exponent."
        ),
        "ratio_helps": True,
    },
    "LOGBIN_NORM": {
        "class": "bin-normalised entropy",
        "expected": (
            "Normalised by the logarithm of the number of frequency bins. The bin count "
            "changes with the grid, so the response is slowly varying rather than a fixed "
            "power law; a single exponent describes it only loosely."
        ),
        "prior_exponent": 0.45,
        "before_measuring": (
            "Report the entropy unnormalised, or state the bin count explicitly, so the "
            "value is reproducible from the metadata."
        ),
        "ratio_helps": True,
    },
    "VALUE_ONLY": {
        "class": "grid-stable (prior)",
        "expected": (
            "No explicit sample count, fixed sample-scale lag or bin count appears in the "
            "formula. Such features are expected to be approximately grid-stable, but "
            "interpolation, threshold ties and preprocessing can still introduce dependence."
        ),
        "prior_exponent": float("inf"),
        "before_measuring": (
            "Still audit the pipeline: a stable definition can become grid-dependent through "
            "the preprocessing attached to it."
        ),
        "ratio_helps": True,
    },
}

_META = (
    "Scope: this audit isolates the digital-grid component. Analog front ends, "
    "anti-alias filters, quantisation and clock jitter are not modelled, so the "
    "values are neither a lower nor an upper bound on the difference between two "
    "physical instruments."
)


# --------------------------------------------------------------------------
# 1. plan
# --------------------------------------------------------------------------
def plan(feature=None, expr=None):
    """Predict the structural class of a feature from its name or definition.

    Parameters
    ----------
    feature : str, optional
        A registry key such as ``"zcr_per_sample"``.
    expr : str, optional
        A free-text or formula-like description such as
        ``"count of zero crossings divided by len(x)"``.

    Returns
    -------
    dict
        ``signature``, ``class``, ``expected``, ``prior_exponent``,
        ``before_measuring``, ``ratio_helps``, ``confidence`` and ``note``.
    """
    if feature:
        if feature not in REGISTRY:
            raise KeyError(
                "unknown feature %r; known: %s" % (feature, ", ".join(sorted(REGISTRY)))
            )
        sig = REGISTRY[feature][1]
        reason = "registry entry"
        confidence = "high"
    elif expr:
        sig, reason = guess_signature(expr)
        confidence = "heuristic"
    else:
        raise ValueError("pass feature= or expr=")

    info = dict(STRUCTURAL_CLASSES[sig])
    return {
        "signature": sig,
        "class": info["class"],
        "expected": info["expected"],
        "prior_exponent": info["prior_exponent"],
        "before_measuring": info["before_measuring"],
        "ratio_helps": info["ratio_helps"],
        "confidence": confidence,
        "reason": reason,
        "note": (
            "The class is a prior derived from the formula, not a proof. Measure the "
            "feature before relying on it. " + _META
        ),
    }


# --------------------------------------------------------------------------
# 2. device profile
# --------------------------------------------------------------------------
def device_profile(fs_hz, band_hz, window_s=None, analog_lpf_hz=None):
    """Turn a user's acquisition settings into the quantities the audit needs."""
    fs_hz = float(fs_hz)
    band_hz = float(band_hz)
    if fs_hz <= 0 or band_hz <= 0:
        raise ValueError("fs_hz and band_hz must be positive")

    r = fs_hz / band_hz
    nyquist_ok = fs_hz / 2.0 > band_hz
    notes = []

    if not nyquist_ok:
        notes.append(
            "fs/2 <= declared band: the band cannot be represented at this rate. "
            "Either the band or the rate is wrong."
        )
    if analog_lpf_hz is not None:
        if analog_lpf_hz < band_hz:
            notes.append(
                "The analog low-pass (%.3g Hz) is below the declared band (%.3g Hz), so "
                "the declared band is not actually present in the signal."
                % (analog_lpf_hz, band_hz)
            )
        if analog_lpf_hz > fs_hz / 2.0:
            notes.append(
                "The analog low-pass (%.3g Hz) is above Nyquist (%.3g Hz): the converter "
                "will alias content the filter was meant to remove."
                % (analog_lpf_hz, fs_hz / 2.0)
            )
    if window_s is not None:
        n = fs_hz * float(window_s)
        if n < 64:
            notes.append("Only %.0f samples per window; estimates will be noisy." % n)

    return {
        "fs_hz": fs_hz,
        "band_hz": band_hz,
        "oversampling_ratio": round(r, 4),
        "nyquist_ok": bool(nyquist_ok),
        "window_s": window_s,
        "samples_per_window": int(fs_hz * float(window_s)) if window_s else None,
        "notes": notes,
        "declared_band_is_the_users_claim": (
            "B is a modelling declaration, not a property you can read off the metadata. "
            "A smaller B inflates r and makes the design look safer than it is."
        ),
        "scope": _META,
    }


# --------------------------------------------------------------------------
# 3. audit
# --------------------------------------------------------------------------
def audit_array(x, fs_hz, band_hz=None, rates=None, features=None):
    """Run the paired grid intervention on one record."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 256:
        raise ValueError("need at least 256 samples, got %d" % x.size)
    fs_hz = float(fs_hz)
    band = float(band_hz) if band_hz else float(estimate_band(x, fs_hz))

    eng = Audit(features=list(features) if features else None)
    res = eng.run(x, fs_hz, band=band, rates=rates)

    rows = [{k: row[k] for k in row} for row in res.rows]

    ratios = sorted({float(r["r"]) for r in res.rows})
    warnings = []
    if len(ratios) < 3:
        warnings.append(
            "Only %d over-sampling ratios were auditable at fs=%.4g Hz and B=%.4g Hz "
            "(%s). A finite-range exponent needs at least three. Supply a higher-rate "
            "record, raise --rates explicitly, or accept a deviation-only report."
            % (len(ratios), fs_hz, band, ", ".join("%g" % x for x in ratios))
        )
    if ratios and max(ratios) < 3:
        warnings.append(
            "The highest audited ratio is %.3g, which is below 3. Any conclusion about "
            "convergence is therefore very weak." % max(ratios)
        )

    return {
        "band_hz": band,
        "fs_hz": fs_hz,
        "mode": res.mode,
        "ratios": ratios,
        "warnings": warnings,
        "rows": rows,
        "exponents": res.exponents(),
        "markdown": res.markdown(),
        "scope": _META,
    }


# --------------------------------------------------------------------------
# 4. advise
# --------------------------------------------------------------------------
def required_ratio(deviation, ratio, exponent, tolerance):
    """Ratio needed so that the fitted curve falls below ``tolerance``.

    Extrapolates the user's own measured curve; it is not a universal rule.
    Returns ``math.inf`` when no ratio helps.
    """
    if exponent is None or not math.isfinite(exponent) or exponent <= 0:
        return math.inf
    if deviation <= tolerance:
        return float(ratio)
    if deviation <= 0:
        return float(ratio)
    return float(ratio * (deviation / tolerance) ** (1.0 / exponent))


def advise(audit, tolerance=0.01, target_ratio=None, band_hz=None):
    """Turn an audit result into a per-feature recommendation.

    The decision is driven by the measured deviation, not by whether an exponent
    could be fitted: a feature with a large constant bias has no usable slope but
    is emphatically not "stable".
    """
    out = []
    exps = audit.get("exponents", {})
    rows = audit.get("rows", [])

    for feat in sorted({r["feature"] for r in rows}):
        mine = [r for r in rows if r["feature"] == feat]
        # deviation as a function of the over-sampling ratio
        curve = {}
        for r in mine:
            d = abs(float(r.get("rel_bias", 0.0)))
            if math.isfinite(d):
                curve[float(r["r"])] = max(curve.get(float(r["r"]), 0.0), d)

        ratios = sorted(curve)
        top = ratios[-1] if ratios else float("nan")
        dev = curve.get(top, float("nan")) if ratios else float("nan")
        p = exps.get(feat, float("nan"))
        info = plan(feature=feat)

        meeting = [x for x in ratios if curve[x] <= tolerance]

        if meeting and curve[meeting[0]] <= tolerance:
            need = float(meeting[0])
            verdict = "met within audited range"
            action = (
                "tolerance %.3g is met from r = %.3g in the range you actually measured"
                % (tolerance, need)
            )
        elif not info["ratio_helps"]:
            need = math.inf
            verdict = "no ratio helps"
            action = (
                "deviation is %.3g at the highest audited ratio and does not shrink with the "
                "grid; this is a scale factor. Redefine the feature in physical units or do "
                "not pool it across devices" % dev
            )
        elif math.isfinite(p) and p > 0:
            need = required_ratio(dev, top, p, tolerance)
            verdict = "extrapolated beyond audited range"
            action = (
                "reaches tolerance near r = %.1f (extrapolated from your own curve; verify "
                "before relying on it)" % need
            )
        else:
            need = float("nan")
            verdict = "unresolved"
            if len(ratios) < 3:
                action = (
                    "deviation is %.3g at r = %.3g, but only %d ratios were auditable, so no "
                    "exponent can be fitted; supply a higher-rate record or a wider rate grid"
                    % (dev, top, len(ratios))
                )
            else:
                action = (
                    "deviation is %.3g at r = %.3g and the curve has no usable slope; extend "
                    "the audited range instead of extrapolating" % (dev, top)
                )

        if target_ratio is not None and verdict == "met within audited range":
            ok = target_ratio >= need
            action += "; your declared r = %.2f is %s" % (
                target_ratio,
                "sufficient" if ok else "below the first ratio that met tolerance",
            )
        elif target_ratio is not None and math.isfinite(need):
            ok = target_ratio >= need
            action += "; your declared r = %.2f is %s" % (
                target_ratio,
                "sufficient" if ok else "below the extrapolated requirement",
            )

        out.append(
            {
                "feature": feat,
                "structural_class": info["class"],
                "measured_exponent": None if not math.isfinite(p) else round(p, 3),
                "deviation_at_top_ratio": None if not math.isfinite(dev) else float(dev),
                "top_ratio": None if not math.isfinite(top) else float(top),
                "curve": {("%g" % k): v for k, v in curve.items()},
                "verdict": verdict,
                "required_ratio_for_tolerance": (
                    None if not math.isfinite(need) else round(float(need), 3)
                ),
                "no_ratio_helps": need == math.inf,
                "action": action,
            }
        )

    return {
        "tolerance": tolerance,
        "target_ratio": target_ratio,
        "band_hz": band_hz if band_hz is not None else audit.get("band_hz"),
        "audited_ratios": audit.get("ratios", []),
        "warnings": audit.get("warnings", []),
        "features": out,
        "caveat": (
            "Required ratios are extrapolations of the measured finite-range exponent, not "
            "acceptance limits. Count-based estimators plateau, so a feature can stop "
            "improving well before the extrapolated ratio. " + _META
        ),
    }


# --------------------------------------------------------------------------
# 5. task dispatcher (the interface an agent or another program calls)
# --------------------------------------------------------------------------
def run_task(task):
    """Run one task described by a dict. Returns JSON-serialisable output."""
    if not isinstance(task, dict):
        raise TypeError("task must be a dict")
    action = task.get("action", "full")

    device = task.get("device") or {}
    profile = None
    if device.get("fs_hz") and device.get("band_hz"):
        profile = device_profile(
            device["fs_hz"],
            device["band_hz"],
            window_s=device.get("window_s"),
            analog_lpf_hz=device.get("analog_lpf_hz"),
        )

    if action == "plan":
        return {"action": "plan", "plan": plan(
            feature=task.get("feature"), expr=task.get("feature_expr")
        ), "device": profile}

    if action == "profile":
        if profile is None:
            raise ValueError("action='profile' needs device.fs_hz and device.band_hz")
        return {"action": "profile", "device": profile, "scope": _META}

    if action in ("audit", "advise", "full"):
        data = task.get("data") or {}
        path = data.get("path")
        if not path:
            raise ValueError("action=%r needs data.path" % action)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            x = np.load(path)
        elif ext in (".csv", ".txt"):
            x = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        else:
            raise ValueError("unsupported data file %r; use .npy or .csv" % path)

        result = audit_array(
            x,
            fs_hz=device.get("fs_hz", data.get("fs_hz")),
            band_hz=device.get("band_hz", data.get("band_hz")),
            rates=task.get("rates"),
            features=task.get("features"),
        )
        if action == "audit":
            return {"action": "audit", "audit": result, "device": profile}

        advice = advise(
            result,
            tolerance=float(task.get("tolerance", 0.01)),
            target_ratio=(profile or {}).get("oversampling_ratio"),
            band_hz=result["band_hz"],
        )
        if action == "advise":
            return {"action": "advise", "advice": advice, "device": profile}

        priors = [
            plan(feature=f) for f in sorted({r["feature"] for r in result["rows"]})
        ]
        return {
            "action": "full",
            "device": profile,
            "plan": priors,
            "audit": result,
            "advice": advice,
            "scope": _META,
        }

    raise ValueError("unknown action %r; use plan|profile|audit|advise|full" % action)


# --------------------------------------------------------------------------
# 6. machine-readable description
# --------------------------------------------------------------------------
def describe():
    """Return the task/response contract, for tool-using agents."""
    return {
        "name": "invariance-agent",
        "purpose": (
            "Decide whether a signal feature computed from a sampled record is comparable "
            "across sampling grids, and what grid a stated tolerance requires."
        ),
        "actions": {
            "plan": {"requires": ["feature | feature_expr"], "returns": "structural class"},
            "profile": {"requires": ["device.fs_hz", "device.band_hz"], "returns": "ratio, admissibility"},
            "audit": {"requires": ["data.path", "fs_hz", "band_hz"], "returns": "per-feature deviations"},
            "advise": {"requires": ["data.path", "tolerance"], "returns": "per-feature recommendation"},
            "full": {"requires": ["data.path", "device"], "returns": "plan + audit + advice"},
        },
        "task_fields": {
            "action": "plan|profile|audit|advise|full",
            "feature": "registry feature name, e.g. zcr_per_sample",
            "feature_expr": "free-text definition, e.g. 'crossings divided by len(x)'",
            "device": {
                "fs_hz": "sampling rate of the acquisition device, Hz",
                "band_hz": "declared content band, Hz (a claim, not a measurement)",
                "window_s": "analysis window, s (optional)",
                "analog_lpf_hz": "analog anti-alias cut-off, Hz (optional)",
            },
            "data": {"path": "path to a 1-D .npy or single-column .csv"},
            "rates": "over-sampling ratios to test, default 2.5..50",
            "features": "subset of registry names (optional)",
            "tolerance": "relative tolerance, default 0.01",
        },
        "guarantees": [
            "isolates the digital-grid component only",
            "never prints a universal over-sampling rule",
            "reports no-ratio-helps when the bias is a scale factor",
        ],
        "structural_classes": {
            k: {
                "class": v["class"],
                "prior_exponent": None if not math.isfinite(v["prior_exponent"]) else v["prior_exponent"],
                "ratio_helps": v["ratio_helps"],
            }
            for k, v in STRUCTURAL_CLASSES.items()
        },
    }


def dumps(obj):
    """JSON with NaN/Infinity turned into null, so other tools can parse it."""
    def clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    return json.dumps(clean(obj), indent=2, ensure_ascii=False)
