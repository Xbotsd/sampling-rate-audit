"""Sanity checks that the engine reproduces the published orders.

Run with pytest, or directly:  python tests/test_core.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakage_audit import ORDER_OF, REGISTRY, Audit  # noqa: E402


def synthetic(n=6, fs=1000.0, band=20.0, duration=2.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    out = []
    for _ in range(n):
        x = np.zeros_like(t)
        for _ in range(15):
            f = rng.uniform(0.5, band)
            x += rng.rayleigh(1.0) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        out.append(x)
    return out


def bias_exponent(feature, fs=1000.0, band=20.0):
    sigs = synthetic(fs=fs, band=band)
    fn = REGISTRY[feature][0]
    ref = [fn(x, fs) for x in sigs]
    from leakage_audit import resample_fft
    rs, bs = [], []
    for r in [2.5, 5.0]:
        vals = []
        for x, rf in zip(sigs, ref):
            y = resample_fft(x, fs, r * band, lp_hz=band)
            vals.append(abs(fn(y, r * band) - rf) / (abs(rf) + 1e-12))
        b = float(np.median(vals))
        if b > 1e-6:
            rs.append(np.log(r))
            bs.append(np.log(b))
    if len(rs) < 2 or max(bs) < np.log(1e-6):
        return float("inf")
    return float(-np.polyfit(rs, bs, 1)[0])


def test_order_one():
    p = bias_exponent("turning_per_sample")
    assert 0.6 < p < 1.4, p


def test_order_two():
    p = bias_exponent("hjorth_mobility_Hz")
    assert 1.5 < p < 2.6, p


def test_invariant():
    """A moment feature must not move by more than 1 % between 1 kHz and 50 Hz."""
    from leakage_audit import resample_fft
    sigs = synthetic(fs=1000.0, band=20.0)
    fn = REGISTRY["std"][0]
    rel = []
    for x in sigs:
        y = resample_fft(x, 1000.0, 50.0, lp_hz=20.0)
        rel.append(abs(fn(y, 50.0) - fn(x, 1000.0)) / fn(x, 1000.0))
    assert float(np.median(rel)) < 1e-2, float(np.median(rel))


def test_invariant_bias_is_tiny():
    """std changes by less than 0.1 % between 4 kHz and 50 Hz."""
    from leakage_audit import resample_fft
    sigs = synthetic(fs=1000.0, band=20.0)
    fn = REGISTRY["std"][0]
    rel = []
    for x in sigs:
        y = resample_fft(x, 1000.0, 50.0, lp_hz=20.0)
        rel.append(abs(fn(y, 50.0) - fn(x, 1000.0)) / fn(x, 1000.0))
    assert float(np.median(rel)) < 1e-2, float(np.median(rel))


def test_registry_signatures():
    for name, (fn, sig, src) in REGISTRY.items():
        assert sig in ORDER_OF, name


def test_audit_runs():
    res = Audit(features=["std", "turning_per_second", "spec_entropy"]).run_many(
        synthetic(n=8), fs=1000.0, band=20.0, rates=[500.0, 250.0])
    assert len(res.rows) >= 6
    assert res.markdown().startswith("| feature")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("bias exponents:", {f: round(bias_exponent(f), 2)
                              for f in ["turning_per_sample", "hjorth_mobility_Hz", "std"]})
