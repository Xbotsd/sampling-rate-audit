"""leakage_audit — sampling-rate leakage audit for feature definitions.

Minimal, numpy-only engine that answers one question:

    given a signal (or a set of signals) and a feature definition, how much
    does the feature depend on the sampling rate?

Quick start
-----------
    from leakage_audit import Audit
    res = Audit().run(x, fs=500, band=20, rates=[250, 125, 62.5])
    print(res.markdown())

    res = Audit().run_many([x1, x2, ...], fs=500)   # AUC + effect size
    res.to_csv("audit.csv")

Command line
------------
    python leakage_audit.py --npy sig.npy --fs 500 --out report/
    python leakage_audit.py --selftest

Design
------
One reference representation and several arm representations of the *same*
signal, band-limited to the same B and resampled through one code path, so
that only the grid differs.  Single-signal mode reports the relative bias and
the fitted exponent; multi-signal mode adds AUC and the effect size d.

The invariance order is assigned from a five-bit signature:
    PER_SAMPLE | N_IN_FORMULA | LOGBIN_NORM | DIFF_COUNT | VALUE_ONLY
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12

# --------------------------------------------------------------------------
# feature bank
# --------------------------------------------------------------------------


def _psd(x, fs):
    X = np.fft.rfft(x - np.mean(x))
    return np.fft.rfftfreq(len(x), 1.0 / fs), np.abs(X) ** 2


def f_std(x, fs):
    return float(np.std(x))


def f_skew(x, fs):
    s = np.std(x) + EPS
    return float(np.mean(((x - x.mean()) / s) ** 3))


def f_kurt(x, fs):
    s = np.std(x) + EPS
    return float(np.mean(((x - x.mean()) / s) ** 4) - 3.0)


def f_spec_centroid(x, fs):
    f, p = _psd(x, fs)
    return float((f * p).sum() / (p.sum() + EPS))


def f_spec_entropy(x, fs):
    f, p = _psd(x, fs)
    q = p / (p.sum() + EPS)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(len(p)))


def f_band_ratio(x, fs, lo=10.0, hi=20.0):
    f, p = _psd(x, fs)
    num = p[(f >= lo) & (f < hi)].sum()
    den = p[f < lo].sum()
    return float(num / (den + EPS))


def f_zcr_sample(x, fs):
    return float(np.count_nonzero(np.diff(np.signbit(x))) / len(x))


def f_zcr_second(x, fs):
    return float(np.count_nonzero(np.diff(np.signbit(x))) / (len(x) / fs))


def f_turn_sample(x, fs):
    d = np.diff(x)
    return float(np.count_nonzero(np.diff(np.signbit(d))) / len(x))


def f_turn_second(x, fs):
    d = np.diff(x)
    return float(np.count_nonzero(np.diff(np.signbit(d))) / (len(x) / fs))


def f_hjorth_mob_sample(x, fs):
    return float(np.sqrt(np.var(np.diff(x)) / (np.var(x) + EPS)))


def f_hjorth_mob_hz(x, fs):
    return f_hjorth_mob_sample(x, fs) * fs


def f_hjorth_complexity(x, fs):
    m = f_hjorth_mob_sample(x, fs)
    d = np.diff(x)
    md = float(np.sqrt(np.var(np.diff(d)) / (np.var(d) + EPS)))
    return md / (m + EPS)


def f_petrosian(x, fs):
    n = len(x)
    nd = np.count_nonzero(np.diff(np.signbit(np.diff(x))))
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * max(nd, 1)))))


def f_katz(x, fs):
    n = len(x)
    d = float(np.max(np.abs(x - x[0])))
    L = float(np.sum(np.abs(np.diff(x))))
    if d <= 0 or L <= 0:
        return float("nan")
    return float(np.log10(n) / (np.log10(n) + np.log10(d / L)))


def f_higuchi(x, fs, kmax=8):
    n = len(x)
    ks, Ls = [], []
    for k in range(1, kmax + 1):
        tot, m = 0.0, 0
        for off in range(k):
            idx = np.arange(off, n, k)
            if len(idx) < 2:
                continue
            tot += np.sum(np.abs(np.diff(x[idx])))
            m += 1
        if m:
            ks.append(k)
            Ls.append((tot / m) * (n - 1) / (k * max(1, (n - 1) // k) * k) * k)
    if len(ks) < 2:
        return float("nan")
    return float(-np.polyfit(np.log(ks), np.log(np.array(Ls) + EPS), 1)[0])


def f_dfa(x, fs, lo=4, hi=16):
    y = np.cumsum(x - np.mean(x))
    F, used = [], []
    for n in np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), 8)).astype(int)):
        if n < 4 or n > len(y) // 4:
            continue
        k = len(y) // n
        if k < 2:
            continue
        seg = y[:k * n].reshape(k, n)
        t = np.arange(n)
        r = 0.0
        for s in seg:
            r += np.sum((s - np.polyval(np.polyfit(t, s, 1), t)) ** 2)
        F.append(np.sqrt(r / (k * n)))
        used.append(n)
    if len(F) < 3:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(np.array(F) + EPS), 1)[0])


def f_sample_entropy(x, fs, m=2, r_frac=0.2, duration_s=4.0):
    n = int(round(duration_s * fs))
    x = np.asarray(x, float)[:n]
    r = r_frac * (np.std(x) + EPS)
    N = len(x)

    def cnt(mm):
        k = N - mm
        if k <= 1:
            return 0.0, 0.0
        T = np.lib.stride_tricks.sliding_window_view(x, mm)[:k]
        c = 0
        for i in range(k - 1):
            c += np.count_nonzero(np.max(np.abs(T[i + 1:] - T[i]), axis=1) <= r)
        return float(c), float(k * (k - 1) / 2)

    c2, t2 = cnt(m)
    c3, t3 = cnt(m + 1)
    if c2 == 0 or c3 == 0:
        return float("nan")
    return float(-np.log((c3 / t3) / (c2 / t2)))


REGISTRY = {
    "std": (f_std, "VALUE_ONLY", "-"),
    "skew": (f_skew, "VALUE_ONLY", "-"),
    "kurt": (f_kurt, "VALUE_ONLY", "-"),
    "spec_centroid_Hz": (f_spec_centroid, "VALUE_ONLY", "-"),
    "band_ratio_10_20": (f_band_ratio, "VALUE_ONLY", "-"),
    "spec_entropy": (f_spec_entropy, "LOGBIN_NORM", "Inouye 1991"),
    "zcr_per_sample": (f_zcr_sample, "PER_SAMPLE", "-"),
    "turning_per_sample": (f_turn_sample, "PER_SAMPLE", "-"),
    "hjorth_mobility_sample": (f_hjorth_mob_sample, "PER_SAMPLE", "Hjorth 1970"),
    "zcr_per_second": (f_zcr_second, "DIFF_COUNT", "-"),
    "turning_per_second": (f_turn_second, "DIFF_COUNT", "-"),
    "hjorth_mobility_Hz": (f_hjorth_mob_hz, "DIFF_COUNT", "Hjorth 1970"),
    "hjorth_complexity": (f_hjorth_complexity, "DIFF_COUNT", "Hjorth 1970"),
    "petrosian_fd": (f_petrosian, "N_IN_FORMULA", "Petrosian 1995"),
    "katz_fd": (f_katz, "N_IN_FORMULA", "Katz 1988"),
    "higuchi_fd": (f_higuchi, "PER_SAMPLE", "Higuchi 1988"),
    "dfa_alpha1": (f_dfa, "N_IN_FORMULA", "Peng 1995"),
    "sample_entropy": (f_sample_entropy, "N_IN_FORMULA", "Richman 2000"),
}

ORDER_OF = {
    "PER_SAMPLE": 1.0,
    "N_IN_FORMULA": 1.0,
    "LOGBIN_NORM": 0.45,
    "DIFF_COUNT": 2.0,
    "VALUE_ONLY": float("inf"),
}


def guess_signature(expr):
    """Heuristic reading of a feature definition written as text.

    Returns (signature, reason). Conservative by design: when the form is not
    recognised it returns N_IN_FORMULA, the most damaging order.
    """
    e = expr.lower()
    if "log(" in e and ("bin" in e or "len(" in e):
        return "LOGBIN_NORM", "normalised by log of the number of bins"
    if any(t in e for t in ("len(", "/n", "/fs", "count", "/len")):
        return "PER_SAMPLE", "normalised by the sample count"
    if any(t in e for t in ("diff", "gradient", "sign", "cross", "turn", "extrema")):
        return "DIFF_COUNT", "built from differences or crossings"
    if any(t in e for t in ("mean", "std", "var", "skew", "kurt", "sum(", "max(", "min(")):
        return "VALUE_ONLY", "algebraic function of sample values"
    return "N_IN_FORMULA", "unrecognised; conservatively treated as explicit N"


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def resample_fft(x, fs_in, fs_out, lp_hz=None):
    """Band-limit to lp_hz and resample through the frequency domain."""
    X = np.fft.rfft(x)
    if lp_hz is not None:
        f = np.fft.rfftfreq(len(x), 1.0 / fs_in)
        X = X.copy()
        X[f > lp_hz] = 0.0
    n_out = int(round(len(x) * fs_out / fs_in))
    if n_out <= len(X):
        Y = X[:n_out]
    else:
        Y = np.concatenate([X, np.zeros(n_out - len(X), dtype=complex)])
    return np.fft.irfft(Y, n=n_out) * (n_out / len(x))


def estimate_band(x, fs, frac=0.99, floor=0.3):
    """Fraction-of-power bandwidth, used when the caller does not supply B."""
    f, p = _psd(x, fs)
    keep = f >= floor
    f, p = f[keep], p[keep]
    if p.sum() <= 0:
        return fs / 4.0
    c = np.cumsum(p) / p.sum()
    return float(f[min(np.searchsorted(c, frac), len(f) - 1)])


def auc(pos, neg):
    pos = np.asarray([v for v in pos if np.isfinite(v)], float)
    neg = np.asarray([v for v in neg if np.isfinite(v)], float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


@dataclass
class AuditResult:
    rows: list = field(default_factory=list)
    band: float = 0.0
    fs: float = 0.0
    mode: str = "single"

    def exponents(self):
        out = {}
        for f in sorted({r["feature"] for r in self.rows}):
            pts = []
            for r in self.rows:
                if r["feature"] != f:
                    continue
                y = r.get("rel_bias", abs(r["auc"] - 0.5))
                if np.isfinite(y) and y > 1e-5:
                    pts.append((r["r"], y))
            if len(pts) < 3:
                out[f] = float("nan")
                continue
            x = np.log([p[0] for p in pts])
            y = np.log([p[1] for p in pts])
            out[f] = float(-np.polyfit(x, y, 1)[0])
        return out

    def markdown(self):
        if not self.rows:
            return "_(no results)_"
        rates = sorted({r["r"] for r in self.rows})
        head = "| feature | order | " + " | ".join(f"r={x:g}" for x in rates) + " | fitted p |"
        sep = "|---|---|" + "---|" * (len(rates) + 1)
        out = [head, sep]
        ex = self.exponents()
        for f in sorted({r["feature"] for r in self.rows}):
            sig = [r for r in self.rows if r["feature"] == f][0]["signature"]
            od = ORDER_OF.get(sig, 0.0)
            od_s = "inf" if od == float("inf") else f"{od:g}"
            cells = []
            for x in rates:
                v = [r for r in self.rows if r["feature"] == f and r["r"] == x]
                if not v:
                    cells.append("-")
                elif self.mode == "multi":
                    cells.append("%.3f / %.2f" % (v[0]["auc"], v[0]["d"]))
                else:
                    cells.append("%.2e" % v[0]["rel_bias"])
            p = ex.get(f, float("nan"))
            ps = "-" if not np.isfinite(p) else "%.2f" % p
            out.append("| `%s` | %s | %s | %s |" % (f, od_s, " | ".join(cells), ps))
        return "\n".join(out)

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)

    def to_json(self, path):
        json.dump(dict(band=self.band, fs=self.fs, mode=self.mode,
                       rows=self.rows, exponents=self.exponents()),
                  open(path, "w"), indent=1)


class Audit:
    """Paired resampling audit.

    run(x)             single signal  -> relative bias and fitted exponent
    run_many([x, ...]) many signals   -> AUC, effect size and exponent
    """

    def __init__(self, features=None, demean=True):
        self.features = features or list(REGISTRY)
        self.demean = demean

    def _prep(self, x, fs, band, rates, anchor):
        x = np.asarray(x, float)
        if self.demean:
            x = x - x.mean()
        if band is None:
            band = estimate_band(x, fs)
        if anchor is None:
            # the reference must be an UP-sampling (or identity) of the native
            # signal; a fixed cap would silently down-sample high-rate inputs
            anchor = 4.0 * fs
            if len(x) * 4 > 4_000_000:
                anchor = 2.0 * fs
        if rates is None:
            rates = [fs / 2.0, fs / 4.0, fs / 8.0]
        # never allow an arm whose Nyquist falls below the declared content band
        rates = [float(r) for r in rates if r / 2.0 > band * 1.02]
        if not rates:
            raise ValueError(
                "no arm satisfies rate/2 > B; lower B or use a higher native rate")
        return x, band, anchor, rates

    def _one(self, x, fs, band, anchor, rates):
        ref = {}
        y = resample_fft(x, fs, anchor, lp_hz=band)
        for f in self.features:
            fn = REGISTRY[f][0]
            try:
                ref[f] = float(fn(y, anchor))
            except Exception:
                ref[f] = float("nan")
        rows = []
        for rate in rates:
            z = resample_fft(x, fs, rate, lp_hz=band)
            for f in self.features:
                fn, sig, src = REGISTRY[f]
                try:
                    v = float(fn(z, rate))
                except Exception:
                    v = float("nan")
                if not (np.isfinite(v) and np.isfinite(ref[f])):
                    continue
                rows.append(dict(feature=f, signature=sig, source=src, fs=rate,
                                 r=rate / band, value=v, reference=ref[f],
                                 rel_bias=abs(v - ref[f]) / (abs(ref[f]) + EPS)))
        return rows

    def run(self, x, fs, band=None, rates=None, anchor=None):
        x, band, anchor, rates = self._prep(x, fs, band, rates, anchor)
        return AuditResult(rows=self._one(x, fs, band, anchor, rates),
                           band=band, fs=fs, mode="single")

    def run_many(self, signals, fs, band=None, rates=None, anchor=None):
        sigs = [np.asarray(s, float) for s in signals]
        _, band, anchor, rates = self._prep(sigs[0], fs, band, rates, anchor)
        acc = {}
        for s in sigs:
            s = s - (s.mean() if self.demean else 0.0)
            for r in self._one(s, fs, band, anchor, rates):
                k = (r["feature"], r["r"])
                acc.setdefault(k, ([], []))
                acc[k][0].append(r["value"])
                acc[k][1].append(r["reference"])
        out = []
        for (f, rr), (pos, neg) in acc.items():
            if len(pos) < 3:
                continue
            mad = 1.4826 * np.median(np.abs(np.array(neg) - np.median(neg))) + EPS
            rel = float(np.median([abs(a - b) / (abs(b) + EPS)
                                   for a, b in zip(pos, neg)]))
            out.append(dict(feature=f, signature=REGISTRY[f][1], source=REGISTRY[f][2],
                            fs=rr * band, r=rr, auc=auc(pos, neg),
                            d=float(abs(np.median(pos) - np.median(neg)) / mad),
                            rel_bias=rel, n=len(pos)))
        return AuditResult(rows=out, band=band, fs=fs, mode="multi")


__all__ = [
    "Audit", "AuditResult", "REGISTRY", "ORDER_OF",
    "resample_fft", "estimate_band", "auc", "guess_signature",
]
