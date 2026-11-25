"""Cross-track arbitration: two cheap answers disagree, and something has to say which to trust.

The Alpha track is the engineering answer: pilot, fit, predict, stop. It is cheap and it is
wrong in a particular way — it extrapolates the crossing from a curve it has only seen the
near field of. The Beta track is the mathematical answer: sweep far enough that the crossing
is bracketed by measurements, fit the order, and report it with the asymptotic tail in hand.
It is expensive and it is occasionally wrong in the opposite way, because a long sweep spends
its budget on ratios where the deficit has already died into noise.

Neither track is right in general, so this module does not rank them by principle. It fits
two models on cases where the truth is known:

* a logistic regression that predicts which track is closer, from disagreement size, the
  fitted class, the surrogate's quality and the pilot's span;
* an ordinary least squares combination of the two log-ratios, which is the posterior point
  estimate once the logistic has said how much to weight Beta.

The output always carries the disagreement and the two raw answers. An arbitration that hid
them would be indistinguishable from a guess.

``CrossTrackReader`` handles the case the design asks for: a caller who only has Alpha rights
can read a Beta result produced elsewhere and verify it before merging. The verification is a
content hash plus a seed, which is enough to detect an edited result and not enough to
detect a fabricated one; that distinction is stated in the reader's return value rather than
glossed over.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

from .grid import CostModel, bic_order
from .oracle import Oracle
from .pareto import search
from .surrogate import Surrogate

FEATURE_NAMES = ["log_disagreement", "class_L", "class_D", "class_S", "surrogate_quality",
                 "pilot_span", "crosses_measured"]


@dataclass
class TrackResult:
    """One track's answer for one (feature, tolerance) case."""

    track: str
    ratio: float
    bands: float
    order_class: str = "?"
    order_p: float | None = None
    units: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"track": self.track, "ratio": round(float(self.ratio), 4),
                "bands": float(self.bands), "order": self.order_class,
                "p": self.order_p, "units": int(self.units), "note": self.note}


# ---------------------------------------------------------------------- the tracks
def alpha_track(oracle: Oracle, feature: str, tau: float, cost: CostModel | None = None,
                top: float = 200.0) -> TrackResult:
    """Cheap: pilot, fit, predict the crossing, stop. No verification."""
    cost = cost or CostModel()
    start = oracle.spent
    pilot = np.array([2.5, 4.0, 6.25, 12.5, 25.0])
    d = np.array([oracle.deficit(feature, r) for r in pilot])
    oracle.reference(feature)
    scale = oracle.scale(feature)
    sub = Surrogate(pilot, np.maximum(d, 1e-15), feature)
    grid = np.geomspace(pilot[0], top, 120)
    pred = sub.predict(grid)
    ok = np.where(pred <= tau * scale)[0]
    ratio = float(grid[ok[0]]) if ok.size else float("nan")
    order = bic_order(d, pilot)
    return TrackResult("alpha", ratio, cost.device_cost(ratio) if np.isfinite(ratio) else float("nan"),
                       order.get("class", "?"), order.get("p"),
                       oracle.spent - start, "surrogate prediction, unverified")


def beta_track(oracle: Oracle, feature: str, tau: float, cost: CostModel | None = None,
               anchor: float = 100.0, top: float = 200.0) -> TrackResult:
    """Expensive: sweep until the crossing is bracketed, then report it."""
    cost = cost or CostModel()
    start = oracle.spent
    sweep = np.geomspace(2.5, top, 14)
    d = np.array([oracle.deficit(feature, r) for r in sweep])
    scale = oracle.scale(feature)
    order = bic_order(d, sweep)
    g = np.array([oracle.g(feature, r, anchor) for r in sweep])
    ok = np.where(g <= tau)[0]
    ratio = float(sweep[ok[0]]) if ok.size else float("nan")
    if ok.size:
        # tighten between the last failure and the first success
        lo = float(sweep[ok[0] - 1]) if ok[0] > 0 else float(sweep[0])
        hi = float(sweep[ok[0]])
        for _ in range(8):
            mid = float(np.sqrt(lo * hi))
            if oracle.g(feature, mid, anchor) <= tau:
                hi = mid
            else:
                lo = mid
        ratio = hi
    return TrackResult("beta", ratio, cost.device_cost(ratio) if np.isfinite(ratio) else float("nan"),
                       order.get("class", "?"), order.get("p"),
                       oracle.spent - start, "measured crossing")


# ---------------------------------------------------------------------- the arbiter
def _design(alpha: TrackResult, beta: TrackResult, quality: float | None,
            pilot_span: float, tau: float) -> np.ndarray:
    la = np.log(max(alpha.ratio, 1e-6)) if np.isfinite(alpha.ratio) else np.log(1e3)
    lb = np.log(max(beta.ratio, 1e-6)) if np.isfinite(beta.ratio) else np.log(1e3)
    return np.array([
        abs(la - lb),
        1.0 if beta.order_class == "L" else 0.0,
        1.0 if beta.order_class == "D" else 0.0,
        1.0 if beta.order_class == "S" else 0.0,
        float(quality) if quality is not None and np.isfinite(quality) else 0.0,
        float(pilot_span),
        1.0 if np.isfinite(beta.ratio) and np.isfinite(alpha.ratio) and beta.ratio <= 50 else 0.0,
    ], dtype=float)


@dataclass
class Arbitrator:
    """Logistic track selection plus an OLS combination, fitted on cases with known truth."""

    coef: np.ndarray | None = None
    intercept: float = 0.0
    ols: np.ndarray | None = None
    n_train: int = 0
    accuracy: float | None = None
    loo_accuracy: float | None = None
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, cases: list[dict]) -> "Arbitrator":
        """`cases` hold alpha/beta results, features and the ratio a dense sweep found."""
        self = cls()
        if len(cases) < 6:
            self.notes.append("too few cases to fit; arbitration falls back to Beta weight 0.5")
            return self
        X = np.array([c["x"] for c in cases], float)
        y = np.array([1 if c["beta_closer"] else 0 for c in cases], int)
        if _HAVE_SKLEARN and len(set(y.tolist())) > 1:
            m = LogisticRegression(C=1.0, max_iter=500)
            m.fit(X, y)
            self.coef = m.coef_.ravel()
            self.intercept = float(m.intercept_[0])
            acc = float((m.predict(X) == y).mean())
        else:  # pragma: no cover - fallback without sklearn
            self.coef = np.zeros(X.shape[1])
            self.intercept = 0.0
            acc = max(y.mean(), 1 - y.mean())
        # OLS: log truth from the two log answers, with Beta given its measured weight
        la = np.array([np.log(max(c["alpha"], 1e-6)) for c in cases])
        lb = np.array([np.log(max(c["beta"], 1e-6)) for c in cases])
        lt = np.array([np.log(max(c["truth"], 1e-6)) for c in cases])
        D = np.column_stack([la, lb, np.ones_like(la)])
        coef, *_ = np.linalg.lstsq(D, lt, rcond=None)
        self.ols = coef
        self.n_train = len(cases)
        self.accuracy = acc
        # leave-one-out, because a 9-case fit reports whatever it is told to report otherwise
        if _HAVE_SKLEARN and len(set(y.tolist())) > 1:
            hits = 0
            for i in range(len(cases)):
                keep = np.arange(len(cases)) != i
                if len(set(y[keep].tolist())) < 2:
                    continue
                m = LogisticRegression(C=1.0, max_iter=500).fit(X[keep], y[keep])
                hits += int(m.predict(X[i:i + 1])[0] == y[i])
            self.loo_accuracy = hits / len(cases)
        if len(cases) < 20:
            self.notes.append("only %d training cases: the arbitration is indicative, not "
                              "calibrated" % len(cases))
        return self

    # ------------------------------------------------------------------ use
    def p_beta(self, x: np.ndarray) -> float:
        if self.coef is None:
            return 0.5
        z = float(np.dot(self.coef, x) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))

    def combine(self, alpha: TrackResult, beta: TrackResult, x: np.ndarray) -> dict:
        p = self.p_beta(x)
        la, lb = np.log(max(alpha.ratio, 1e-6)), np.log(max(beta.ratio, 1e-6))
        if self.ols is not None:
            post = float(np.exp(self.ols[0] * la + self.ols[1] * lb + self.ols[2]))
        else:
            post = float(np.exp((1 - p) * la + p * lb))
        disagreement = abs(la - lb)
        if disagreement < 0.05:
            status = "agree"
        elif 0.4 <= p <= 0.6:
            status = "unresolved"
        else:
            status = "alpha" if p < 0.4 else "beta"
        return {"p_beta": round(p, 3), "posterior_ratio": round(post, 4),
                "disagreement_log": round(disagreement, 3), "status": status,
                "weights": {"log_alpha": None if self.ols is None else round(float(self.ols[0]), 3),
                            "log_beta": None if self.ols is None else round(float(self.ols[1]), 3)},
                "alpha": alpha.as_dict(), "beta": beta.as_dict()}

    def as_dict(self) -> dict:
        return {"coef": None if self.coef is None else [round(float(c), 4) for c in self.coef],
                "intercept": round(float(self.intercept), 4),
                "ols": None if self.ols is None else [round(float(c), 4) for c in self.ols],
                "features": FEATURE_NAMES, "n_train": self.n_train,
                "train_accuracy": self.accuracy, "loo_accuracy": self.loo_accuracy,
                "notes": self.notes}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.as_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "Arbitrator":
        with open(path, encoding="utf-8") as fh:
            body = json.load(fh)
        self = cls()
        self.coef = None if body["coef"] is None else np.array(body["coef"], float)
        self.intercept = float(body["intercept"])
        self.ols = None if body["ols"] is None else np.array(body["ols"], float)
        self.n_train = int(body["n_train"])
        self.accuracy = body["train_accuracy"]
        self.loo_accuracy = body.get("loo_accuracy")
        self.notes = list(body.get("notes", []))
        return self


# ---------------------------------------------------------------------- training data
def build_training_cases(features: list[str], taus=(0.05, 0.02, 0.01), n_records: int = 3,
                         seed: int = 101) -> list[dict]:
    """Run both tracks and a dense reference sweep on synthetic records, and keep the cases."""
    oracle = Oracle(features, n_records=n_records, seed=seed)
    cases = []
    dense = np.geomspace(2.5, 200.0, 24)
    for feature in features:
        for tau in taus:
            a = alpha_track(oracle, feature, tau)
            b = beta_track(oracle, feature, tau)
            anchor = 100.0
            g = np.array([oracle.g(feature, r, anchor) for r in dense])
            ok = np.where(g <= tau)[0]
            truth = float(dense[ok[0]]) if ok.size else float("nan")
            d = np.array([oracle.deficit(feature, r) for r in dense])
            ord_ = bic_order(d, dense)
            if not (np.isfinite(truth) and np.isfinite(a.ratio) and np.isfinite(b.ratio)):
                continue
            pilot_r = np.array([2.5, 4.0, 6.25, 12.5, 25.0])
            pilot_d = np.array([oracle.deficit(feature, r) for r in pilot_r])
            q = Surrogate(pilot_r, np.maximum(pilot_d, 1e-15), feature).quality_of_best()
            x = _design(a, b, q, np.log(25.0 / 2.5), tau)
            cases.append({"feature": feature, "tau": tau, "alpha": a.ratio, "beta": b.ratio,
                          "truth": truth, "x": x.tolist(),
                          "beta_closer": abs(np.log(b.ratio / truth)) < abs(np.log(a.ratio / truth)),
                          "order": ord_})
    return cases


# ---------------------------------------------------------------------- cross-track reader
@dataclass
class CrossTrackReader:
    """Read a Beta result produced elsewhere and decide whether it may be merged."""

    path: str

    def load(self) -> dict:
        with open(self.path, encoding="utf-8") as fh:
            body = json.load(fh)
        claimed = body.pop("content_sha256", "")
        actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        body["content_sha256"] = claimed
        body["_integrity"] = {
            "hash_ok": bool(claimed) and claimed == actual,
            "seed_present": "seed" in body,
            "detects": "an edited result, and a truncated file",
            "does_not_detect": "a fabricated result with a matching hash",
            "mergeable": bool(claimed) and claimed == actual and "seed" in body,
        }
        return body

    def merge(self, alpha: TrackResult, body: dict) -> dict:
        """Combine a local Alpha answer with a foreign Beta answer, if the foreign one verifies."""
        if not body.get("_integrity", {}).get("mergeable"):
            return {"merged": False,
                    "reason": "the foreign Beta result failed verification: %s"
                              % body.get("_integrity", {})}
        beta = TrackResult("beta", float(body["ratio"]), float(body.get("bands", float("nan"))),
                           str(body.get("order", "?")), body.get("p"), 0,
                           "read from %s" % os.path.basename(self.path))
        return {"merged": True, "alpha": alpha.as_dict(), "beta": beta.as_dict(),
                "note": "the foreign result carries a seed and a matching hash; reproducibility "
                        "can be checked by re-running that seed"}
