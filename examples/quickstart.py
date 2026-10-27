"""Minimal example: does each feature encode the sampling rate?

Runs on synthetic data, so it needs nothing but numpy and this package.

    python examples/quickstart.py
"""
import numpy as np

from leakage_audit import Audit


def synthetic(n=40, fs=500.0, band=20.0, duration=10.0, seed=0):
    """Band-limited noise, one realisation per 'record'."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    out = []
    for _ in range(n):
        x = np.zeros_like(t)
        for _ in range(25):
            f = rng.uniform(0.5, band)
            x += rng.rayleigh(1.0) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        out.append(x)
    return out


def main():
    fs, band = 500.0, 20.0
    signals = synthetic(fs=fs, band=band)
    res = Audit().run_many(signals, fs=fs, band=band, rates=[fs / 2, fs / 4, fs / 8])
    print(f"B = {res.band:g} Hz, fs = {res.fs:g} Hz, "
          f"r = {sorted({round(x['r'], 2) for x in res.rows})}\n")
    print(res.markdown())
    print("\nEach cell is AUC / d.  order = predicted from the definition; "
          "fitted p = measured from the bias.")


if __name__ == "__main__":
    main()
