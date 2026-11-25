"""Surrogate pre-check: spend the compute once, on the points that decide.

Fitting the curve on a pilot grid is cheap; running the audit everywhere is not. A
surrogate fitted to the pilot can predict the rest, but only where the pilot actually
constrains it. This module therefore does three things, in this order:

1. fits several cheap regressors and scores them by leave-one-out error on the pilot;
2. refuses to extrapolate when the pilot is too small for the score to mean anything;
3. when the caller lowers the acceptance threshold anyway, records a signed risk
   agreement rather than silently letting a weak model through.

Point 2 is the part that makes this a pre-check rather than a rubber stamp: a
three-point pilot always fits a four-parameter model perfectly, and always predicts the
far field badly. The gate knows that.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - environment without sklearn
    _HAVE_SKLEARN = False

try:
    from scipy.interpolate import PchipInterpolator

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

DEFAULT_THRESHOLD = 0.80
FLOOR_THRESHOLD = 0.40
MIN_PILOT = 5


def _log(x):
    return np.log10(np.maximum(np.asarray(x, float), 1e-30))


@dataclass
class Surrogate:
    """Predicts log10(deficit) from log10(ratio), with an honest extrapolation gate."""

    ratios: np.ndarray
    deficits: np.ndarray
    feature: str = ""
    models: dict = field(default_factory=dict, init=False)
    quality: dict = field(default_factory=dict, init=False)
    best: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.ratios = np.asarray(self.ratios, float)
        self.deficits = np.asarray(self.deficits, float)
        order = np.argsort(self.ratios)
        self.ratios, self.deficits = self.ratios[order], self.deficits[order]
        self._fit()
        self._score()

    # ------------------------------------------------------------------ fitting
    def _fit(self) -> None:
        x = _log(self.ratios)
        y = _log(np.maximum(self.deficits, 1e-12))
        self.models = {}
        if x.size >= 3:
            self.models["pchip"] = ("interp", PchipInterpolator(x, y, extrapolate=True)
                                    if _HAVE_SCIPY else np.poly1d(np.polyfit(x, y, 1)))
        if x.size >= 2:
            self.models["linear"] = ("poly", np.poly1d(np.polyfit(x, y, 1)))
        if x.size >= 4:
            self.models["quad"] = ("poly", np.poly1d(np.polyfit(x, y, min(2, x.size - 1))))
        if _HAVE_SKLEARN and x.size >= 3:
            rf = RandomForestRegressor(n_estimators=64, min_samples_leaf=1, random_state=0)
            rf.fit(x.reshape(-1, 1), y)
            self.models["forest"] = ("forest", rf)

    def _predict_one(self, name, xq):
        kind, model = self.models[name]
        xq = np.atleast_1d(np.asarray(xq, float))
        if kind == "forest":
            return model.predict(xq.reshape(-1, 1))
        return model(xq)

    # ------------------------------------------------------------------ scoring
    def _score(self) -> None:
        """Leave-one-out R^2 per model. With fewer than MIN_PILOT points this is noise."""
        x = _log(self.ratios)
        y = _log(np.maximum(self.deficits, 1e-12))
        n = x.size
        if n < MIN_PILOT:
            self.quality = {name: None for name in self.models}
            self.best = ""
            return
        for name in list(self.models):
            preds = np.empty(n)
            for i in range(n):
                keep = np.arange(n) != i
                sub = Surrogate(self.ratios[keep], self.deficits[keep], self.feature)
                preds[i] = sub._predict_one(name, x[i])[0]
            ss_res = float(((y - preds) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
            self.quality[name] = 1.0 - ss_res / ss_tot
        finite = {k: v for k, v in self.quality.items()
                  if v is not None and np.isfinite(v)}
        self.best = max(finite, key=finite.get) if finite else ""

    # ------------------------------------------------------------------ querying
    def predict(self, ratios) -> np.ndarray:
        name = self.best or next(iter(self.models))
        return 10.0 ** self._predict_one(name, _log(ratios))

    def spread(self, ratios) -> np.ndarray:
        """Disagreement across models, in log10 units. Large means: measure it."""
        name = self.best or next(iter(self.models))
        xq = _log(ratios)
        if "forest" in self.models:
            rf = self.models["forest"][1]
            per_tree = np.array([t.predict(np.atleast_1d(xq).reshape(-1, 1))
                                 for t in rf.estimators_])
            return per_tree.std(axis=0)
        others = [n for n in self.models if n != name]
        if not others:
            return np.zeros(len(np.atleast_1d(xq)))
        base = self._predict_one(name, xq)
        return np.abs(np.array([self._predict_one(o, xq) for o in others]) - base).max(axis=0)

    # ------------------------------------------------------------------ the gate
    @property
    def n_pilot(self) -> int:
        return int(self.ratios.size)

    def quality_of_best(self):
        return self.quality.get(self.best) if self.best else None

    def gate(self, threshold: float = DEFAULT_THRESHOLD, target: float | None = None) -> dict:
        """Decide whether the target ratio may be answered by the surrogate or must be measured."""
        if threshold < FLOOR_THRESHOLD:
            return {"allowed": False, "reason": "threshold below the admissible floor",
                    "threshold": threshold, "floor": FLOOR_THRESHOLD}
        q = self.quality_of_best()
        if q is None:
            return {"allowed": False, "reason": "pilot too small for a trustworthy score",
                    "n_pilot": self.n_pilot, "min_pilot": MIN_PILOT, "threshold": threshold}
        if target is not None and not (self.ratios.min() <= target <= self.ratios.max()):
            factor = max(target / self.ratios.max(), self.ratios.min() / target)
            if factor > 2.0:
                return {"allowed": False,
                        "reason": "extrapolation factor %.2fx beyond the pilot" % factor,
                        "factor": factor, "quality": q, "threshold": threshold}
        if q < threshold:
            return {"allowed": False, "reason": "surrogate fit below the stated threshold",
                    "quality": q, "threshold": threshold, "best_model": self.best}
        return {"allowed": True, "reason": "fit above threshold",
                "quality": q, "threshold": threshold, "best_model": self.best}


def sign_risk_agreement(path: str, surrogate: Surrogate, threshold: float,
                        signed_by: str, reason: str = "") -> str:
    """Record that a caller chose to proceed below the default acceptance threshold.

    The file is a plain JSON record with a content hash, so a later reader can tell
    whether it was edited after signing. It is not a cryptographic signature, and does
    not pretend to be one.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = {
        "kind": "prediction-distortion-risk-agreement",
        "sra_version": "3.1.1",
        "feature": surrogate.feature,
        "pilot_points": surrogate.n_pilot,
        "pilot_ratios": [float(r) for r in surrogate.ratios],
        "chosen_threshold": float(threshold),
        "default_threshold": DEFAULT_THRESHOLD,
        "admissible_floor": FLOOR_THRESHOLD,
        "surrogate_quality": surrogate.quality_of_best(),
        "best_model": surrogate.best,
        "accepted_prediction_error": None if surrogate.quality_of_best() is None else round(
            1.0 - float(surrogate.quality_of_best()), 4),
        "reason": reason,
        "signed_by": signed_by,
        "signed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["content_sha256"] = digest
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(body, fh, indent=1, ensure_ascii=False)
    return digest


def verify_risk_agreement(path: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        body = json.load(fh)
    claimed = body.pop("content_sha256", "")
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == claimed
