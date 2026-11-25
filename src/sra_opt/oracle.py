"""Measurement oracle: the expensive ground truth behind every optimisation here.

One *cost unit* is one (record, feature, ratio) evaluation, because that is the real
recurring expense of an audit: resample, recompute the feature, compare. The oracle
caches per-record feature values, so a ratio already paid for is free for every later
question — the deficit against the continuous limit, the anchor-relative deviation at any
anchor, or the fitted order.

Everything downstream is derived from these values, in the same way the paper derives it,
so the optimiser cannot quietly use a different definition of the quantity it reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leakage_audit import REGISTRY, resample_fft

TOP_RATIO = 400.0  # fs / B at the reference arm; the best available stand-in for F_inf


def band_limited(n: int, fs: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Random-phase band-limited record, identical in construction to the paper's sweeps."""
    t = np.arange(n) / fs
    kmax = max(1, int(b * n / fs))
    x = np.zeros(n)
    for k in range(1, kmax + 1):
        x += rng.rayleigh(1.0) * np.cos(2 * np.pi * k * b / kmax * t + rng.uniform(0, 2 * np.pi))
    return x / (np.std(x) + 1e-12)


@dataclass
class Oracle:
    """Per-record feature values on synthetic records, billed by the point."""

    features: list[str]
    n_records: int = 8
    band: float = 20.0
    duration: float = 10.0
    fs: float = 8000.0
    seed: int = 0

    signals: list[np.ndarray] = field(default_factory=list, init=False)
    _values: dict[tuple[str, float], np.ndarray] = field(default_factory=dict, init=False)
    units: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    hits: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = int(self.duration * self.fs)
        self.signals = [band_limited(n, self.fs, self.band, rng) for _ in range(self.n_records)]

    # ------------------------------------------------------------------ pricing
    def price(self, n_ratios: int) -> int:
        return int(n_ratios * self.n_records)

    @property
    def spent(self) -> int:
        return self.units

    # ------------------------------------------------------------------ primitives
    def values(self, feature: str, ratio: float) -> np.ndarray:
        """Per-record feature values at one ratio. Cached; billed once per ratio."""
        key = (feature, round(float(ratio), 9))
        if key in self._values:
            self.hits += 1
            return self._values[key]
        fn = REGISTRY[feature][0]
        fs_arm = float(ratio) * self.band
        out = np.array([fn(resample_fft(x, self.fs, fs_arm, lp_hz=self.band), fs_arm)
                        for x in self.signals], dtype=float)
        self.units += self.n_records
        self.misses += 1
        self._values[key] = out
        return out

    def known_ratios(self, feature: str) -> list[float]:
        """Ratios already paid for, deduplicated at the cache's own tolerance."""
        seen = {}
        for (f, r) in self._values:
            if f == feature:
                seen[round(r, 6)] = r
        return sorted(seen.values())

    # ------------------------------------------------------------------ derived quantities
    def reference(self, feature: str) -> np.ndarray:
        """The best available stand-in for F_inf: the values at TOP_RATIO."""
        return self.values(feature, TOP_RATIO)

    def deficit(self, feature: str, ratio: float) -> float:
        """Median |F(r) - F_inf|, the paper's D_F (absolute, in feature units)."""
        v = self.values(feature, ratio)
        ref = self.reference(feature)
        return float(np.median(np.abs(v - ref)))

    def deficit_rel(self, feature: str, ratio: float) -> float:
        """The same quantity divided by |F_inf|, for reporting next to relative deviations."""
        v = self.values(feature, ratio)
        ref = self.reference(feature)
        return float(np.median(np.abs(v - ref) / (np.abs(ref) + 1e-12)))

    def scale(self, feature: str) -> float:
        """|F_inf|, the denominator that turns a deficit into a relative deviation."""
        return float(np.median(np.abs(self.reference(feature))) + 1e-12)

    def deficit_curve(self, feature: str, ratios) -> np.ndarray:
        return np.array([self.deficit(feature, r) for r in ratios], dtype=float)

    def g(self, feature: str, ratio: float, anchor: float) -> float:
        """Median |F(r) - F(a)| / |F(a)|, the anchor-relative deviation of the paper."""
        v = self.values(feature, ratio)
        a = self.values(feature, anchor)
        return float(np.median(np.abs(v - a) / (np.abs(a) + 1e-12)))

    def floor(self, feature: str, anchor: float) -> float:
        """Theorem 5's floor D(a)/|F(a)|, evaluated on the record set."""
        v = self.values(feature, anchor)
        ref = self.reference(feature)
        d = np.abs(v - ref)
        return float(np.median(d / (np.abs(v) + 1e-12)))


def feature_menu() -> list[str]:
    """The audited definitions that are cheap enough for a grid search."""
    cheap = ["spec_entropy", "zcr_per_second", "turning_per_second", "hjorth_mobility_Hz",
             "zcr_per_sample", "turning_per_sample", "hjorth_mobility_sample", "std"]
    return [f for f in cheap if f in REGISTRY]
