"""Grid configurations, the cost model, and what counts as an admissible answer.

The design question this module exists for: given a tolerance on the anchor-relative
deviation and a cost model, which sampling grid should be built? The cost model is
explicit and replaceable, because the honest answer to "what is the cheapest grid"
depends entirely on how the caller prices sampling rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostModel:
    """Cost of a deployment, in arbitrary currency (mW, MB/s, ADC price — user's choice).

    ``sampling`` scales with the rate a device must sustain, which is what a cheaper ADC,
    a smaller battery and a slower bus all buy. ``fixed`` covers per-channel overhead.
    """

    fixed: float = 1.0
    per_unit_rate: float = 1.0
    gamma: float = 1.0

    def device_cost(self, ratio: float) -> float:
        """Cost of a device that must sustain rate ratio `r` (relative to the content band)."""
        return self.fixed + self.per_unit_rate * float(ratio) ** self.gamma

    def search_cost(self, ratios, n_features: int, n_records: int) -> float:
        """Cost of *finding out*, which is separate from the cost of deploying."""
        return float(len(ratios) * n_features * n_records)


@dataclass
class GridPlan:
    """A candidate answer: the ratios to measure at, and the anchor they are relative to."""

    name: str
    ratios: tuple[float, ...]
    anchor: float
    note: str = ""
    tags: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ratios = tuple(sorted(float(r) for r in self.ratios))

    @property
    def width(self) -> float:
        return self.ratios[-1] / self.ratios[0] if self.ratios else 1.0


def coarse_plan(anchor: float = 2.5) -> GridPlan:
    return GridPlan("coarse-3", (2.5, 6.25, 25.0), anchor,
                    "three log-spaced points; enough to distinguish a power from a log")


def medium_plan(anchor: float = 2.5) -> GridPlan:
    return GridPlan("medium-5", (2.5, 4.0, 6.25, 12.5, 25.0), anchor,
                    "the paper's working grid")


def fine_plan(anchor: float = 2.5) -> GridPlan:
    return GridPlan("fine-9", (2.5, 3.125, 4.0, 5.0, 6.25, 8.0, 12.5, 25.0, 50.0), anchor,
                    "dense enough for a two-point Richardson extrapolation")


def sweep_plan(anchor: float = 2.5, top: float = 100.0) -> GridPlan:
    return GridPlan("sweep-12", tuple(np.geomspace(2.5, top, 12)), anchor,
                    "what a defensive auditor runs when nothing is known")


def candidates(anchor: float = 2.5) -> list[GridPlan]:
    return [coarse_plan(anchor), medium_plan(anchor), fine_plan(anchor), sweep_plan(anchor)]


def required_ratio(p: float, tau: float, g_rt: float, r_t: float) -> float:
    """The repair-test estimate r_req = r_t (tau / g(r_t))^(1/p), for p > 0."""
    if p <= 0:
        return float("inf")
    if g_rt <= tau:
        return float(r_t)
    return float(r_t * (tau / g_rt) ** (1.0 / p))


def bic_order(deficits: np.ndarray, ratios: np.ndarray) -> dict:
    """Class the measured deficit curve by BIC, the same comparison the paper uses.

    Constant is not a candidate here: a deficit of zero everywhere is reported separately
    as the invariant class, because a flat curve cannot be told from a saturated one by a
    constant fit.
    """
    d = np.asarray(deficits, float)
    r = np.asarray(ratios, float)
    ok = np.isfinite(d) & np.isfinite(r) & (d > 0)
    d, r = d[ok], r[ok]
    if d.size < 3:
        return {"class": "?", "p": None, "bic": {}}
    if float(d.max()) < 1e-9:
        return {"class": "V", "p": float("inf"), "bic": {}}
    n = d.size
    designs = {
        "S": (np.column_stack([1.0 / r]), 1),
        "D": (np.column_stack([np.ones(n), r ** -2.0]), 2),
        "L": (np.column_stack([np.ones(n), 1.0 / np.log(r)]), 0),
    }
    scores = {}
    for name, (X, p) in designs.items():
        beta, *_ = np.linalg.lstsq(X, d, rcond=None)
        rss = float(((d - X @ beta) ** 2).sum())
        bic = n * np.log(max(rss, 1e-300) / n) + X.shape[1] * np.log(n)
        scores[name] = {"bic": float(bic), "p": p, "beta": [float(b) for b in beta]}
    best = min(scores, key=lambda k: scores[k]["bic"])
    return {"class": best, "p": scores[best]["p"],
            "bic": {k: round(v["bic"], 2) for k, v in scores.items()},
            "margin": round(min(scores[k]["bic"] for k in scores if k != best) - scores[best]["bic"], 2)}


def achieved_tolerance(deficits: np.ndarray, ratios: np.ndarray, anchor: float) -> float:
    """Worst anchor-relative deviation over the grid, which is what a tolerance bounds.

    The anchor floor of the paper's Theorem 5 is folded in: the achieved deviation cannot
    be better than the anchor's own deficit, whatever the other ratios do.
    """
    deficits = np.asarray(deficits, float)
    ratios = np.asarray(ratios, float)
    if deficits.size == 0:
        return float("inf")
    worst = float(np.max(deficits))
    floor = float(np.interp(anchor, ratios, deficits)) if ratios.min() <= anchor <= ratios.max() else 0.0
    return max(worst, floor)
