"""Pareto search: find the cheapest grid that meets a tolerance, and refuse to pretend.

The question is concrete. Given a tolerance ``tau`` on the anchor-relative deviation
``g(r; a) = median |F(r) - F(a)| / |F(a)|`` and a cost model for a device that must sustain
a rate ratio ``r``, which ``(anchor, r)`` pair certifies ``tau`` at the lowest device cost —
and what is that certificate worth?

The search is a predict-then-verify loop, which is where the compute saving comes from:

1. measure a five-point pilot plus the reference point and fit the deficit curve;
2. predict the anchor requirement from Theorem 5 (:math:`D(a)/|F(a)| \\le \\tau`) and the
   crossing ratio from the fitted curve — cheap, and possibly wrong;
3. measure the predicted crossing and see whether the prediction held;
4. if it did not, walk outwards and repeat, inside a fixed budget.

Three mechanisms keep the answer honest:

* every ratio is labelled *measured*, *interpolated* or *extrapolated*, and a
  cheapest-feasible answer resting on extrapolation is not returned as a verdict;
* the anchor floor is enforced: if no candidate anchor has a floor below the tolerance, the
  verdict is *anchor-limited* and no device ratio is offered, because none can help;
* when the cheap answer and the defensible one differ, the search refuses to choose. It
  returns a :class:`ValueChoice` naming what each option costs and what each one risks. A
  tool that silently prefers the cheap answer has the same bias as the metric it criticises.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .grid import CostModel, GridPlan, bic_order, candidates
from .oracle import TOP_RATIO, Oracle
from .surrogate import Surrogate

LADDER_TOP = 200.0
MAX_EXTRAPOLATION = 2.0
MAX_ATTEMPTS = 4
ANCHORS = (10.0, 25.0, 50.0, 100.0, 200.0)
# The anchor is the high-rate reference arm; it must sit strictly above the device rate,
# otherwise g(a; a) = 0 by construction and any grid would "certify" any tolerance.
ANCHOR_MARGIN = 2.0
# The anchor is the high-rate reference arm; it must sit strictly above the device rate,
# otherwise g(a; a) = 0 by construction and any grid would "certify" any tolerance.
ANCHOR_MARGIN = 2.0


@dataclass
class Certificate:
    """A claim of the form: an anchor at ``a`` and a device at ``r`` meet ``tau``."""

    anchor: float
    device_ratio: float
    predicted_g: float
    measured_g: float | None
    anchor_floor: float
    achieved: float
    evidence: str
    extrapolation_factor: float
    device_cost: float
    admissible: bool
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "anchor": round(float(self.anchor), 4),
            "device_ratio": round(float(self.device_ratio), 4),
            "predicted_g": float(self.predicted_g),
            "measured_g": None if self.measured_g is None else float(self.measured_g),
            "anchor_floor": float(self.anchor_floor),
            "achieved_tolerance": float(self.achieved),
            "evidence": self.evidence,
            "extrapolation_factor": round(float(self.extrapolation_factor), 3),
            "device_cost": round(float(self.device_cost), 3),
            "admissible": bool(self.admissible),
            "flags": list(self.flags),
        }


@dataclass
class ValueChoice:
    """A decision the system declines to make on the caller's behalf."""

    question: str
    options: list[dict]
    context: dict = field(default_factory=dict)
    chosen: str | None = None

    def choose(self, name: str) -> dict:
        for opt in self.options:
            if opt["name"] == name:
                self.chosen = name
                return opt
        raise KeyError("unknown option %r; available: %s"
                       % (name, [o["name"] for o in self.options]))

    def as_dict(self) -> dict:
        return {"question": self.question, "options": self.options,
                "context": self.context, "chosen": self.chosen}


@dataclass
class SearchReport:
    feature: str
    tau: float
    certificate: Certificate | None = None
    attempts: list[dict] = field(default_factory=list)
    frontier: list[dict] = field(default_factory=list)
    choice: ValueChoice | None = None
    spent_units: int = 0
    baseline_units: int = 0
    order: dict = field(default_factory=dict)
    surrogate_model: str = ""
    surrogate_quality: float | None = None
    measured: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    plan_name: str = ""
    anchor_scan: list[dict] = field(default_factory=list)

    @property
    def saving(self) -> float:
        if self.baseline_units <= 0:
            return 0.0
        return 1.0 - self.spent_units / self.baseline_units

    def verdict(self) -> str:
        c = self.certificate
        if c is None:
            if self.anchor_scan and not any(a["feasible"] for a in self.anchor_scan):
                best = min(self.anchor_scan, key=lambda a: a["floor"])
                return ("anchor-limited: the finest admissible anchor has a floor of %.4g, above "
                        "tau=%g; no device ratio can certify this tolerance (raise the anchor)"
                        % (best["floor"], self.tau))
            return ("infeasible within the audited range: the fitted order does not bring the "
                    "deviation to tau=%g before r=%g" % (self.tau, LADDER_TOP))
        if c.admissible:
            return ("a=%.0f, r*=%.1f (device cost %.1f) certifies tau=%g from %d measured ratios; "
                    "%.0f%% of the defensive sweep's compute"
                    % (c.anchor, c.device_ratio, c.device_cost, self.tau, len(self.measured),
                       100 * (1 - self.saving)))
        return ("a=%.0f, r*=%.1f would certify tau=%g, but the certificate is %s (%.1fx beyond the "
                "measured range); a value choice is required"
                % (c.anchor, c.device_ratio, self.tau, c.evidence, c.extrapolation_factor))

    def as_dict(self) -> dict:
        return {
            "feature": self.feature, "tau": self.tau, "plan": self.plan_name,
            "spent_units": self.spent_units, "baseline_units": self.baseline_units,
            "compute_saved": round(self.saving, 3),
            "order": self.order,
            "surrogate": {"model": self.surrogate_model, "quality": self.surrogate_quality},
            "measured_ratios": [float(r) for r in self.measured],
            "predicted_ratios": [float(r) for r in self.predicted],
            "anchor_scan": self.anchor_scan,
            "attempts": self.attempts,
            "certificate": None if self.certificate is None else self.certificate.as_dict(),
            "frontier": self.frontier,
            "choice": None if self.choice is None else self.choice.as_dict(),
            "verdict": self.verdict(),
        }


def _frontier(points: list[dict]) -> list[dict]:
    pts = [p for p in points if np.isfinite(p["achieved_tolerance"])]
    pts.sort(key=lambda p: p["device_cost"])
    out, best = [], float("inf")
    for p in pts:
        if p["achieved_tolerance"] < best - 1e-12:
            best = p["achieved_tolerance"]
            out.append(dict(p))
    return out


def _crossing(sub: Surrogate, target: float, lo: float, hi: float) -> float | None:
    """Smallest r in [lo, hi] with predicted absolute deficit <= target."""
    grid = np.geomspace(lo, hi, 90)
    pred = sub.predict(grid)
    ok = np.where(pred <= target)[0]
    if ok.size == 0:
        return None
    i = int(ok[0])
    if i == 0:
        return float(grid[0])
    a, b = float(grid[i - 1]), float(grid[i])
    for _ in range(40):
        mid = float(np.sqrt(a * b))
        if float(sub.predict([mid])[0]) <= target:
            b = mid
        else:
            a = mid
    return b


def search(oracle: Oracle, feature: str, tau: float = 0.05,
           cost: CostModel | None = None, budget_units: int | None = None,
           threshold: float = 0.80, pilot: GridPlan | None = None,
           anchors=ANCHORS, ladder_top: float = LADDER_TOP,
           accept_choice: str | None = None) -> SearchReport:
    """Budgeted predict-then-verify search for the cheapest certifiable (anchor, rate)."""
    cost = cost or CostModel()
    pilot = pilot or candidates(2.5)[1]
    budget_units = budget_units or oracle.price(len(pilot.ratios) + 8)
    spent0 = oracle.spent

    rep = SearchReport(feature=feature, tau=tau, plan_name=pilot.name)

    # ---- phase 1: pilot plus reference
    pilot_deficits = np.array([oracle.deficit(feature, r) for r in pilot.ratios])
    oracle.reference(feature)                      # paid for once, reused by every deficit
    sub = Surrogate(np.array(pilot.ratios), np.maximum(pilot_deficits, 1e-15), feature)
    rep.surrogate_quality = sub.quality_of_best()
    rep.surrogate_model = sub.best
    scale = oracle.scale(feature)

    # ---- phase 2: anchors, Theorem 5 in deficit form (predicted, then one measurement)
    # ---- phase 3: predict the cheapest device, then the cheapest anchor that can back it
    target = tau * scale
    r_hat = _crossing(sub, target, pilot.ratios[0], ladder_top)
    cert = None
    if r_hat is None:
        rep.measured = oracle.known_ratios(feature)
        rep.spent_units = oracle.spent - spent0
        rep.baseline_units = oracle.price(14)
        return rep

    needed = r_hat * ANCHOR_MARGIN
    ranked = sorted(a for a in anchors if a >= needed)
    predicted_floors = {a: float(sub.predict([a])[0]) / scale for a in anchors}
    # the anchor floor must sit well below the tolerance, not merely below it
    chosen, floor = None, float("inf")
    for a in ranked:
        rep.anchor_scan.append({"anchor": float(a), "predicted_floor": predicted_floors[a],
                                "feasible": bool(predicted_floors[a] <= tau / 2)})
        if predicted_floors[a] > tau / 2 and chosen is None:
            continue
        if (oracle.spent - spent0) + oracle.price(1) > budget_units:
            break
        floor = oracle.floor(feature, a)
        rep.anchor_scan[-1].update({"floor": floor, "measured": True,
                                    "feasible": bool(floor <= tau)})
        if floor <= tau:
            chosen = a
            break
    if chosen is None:
        best_a = ranked[-1] if ranked else max(anchors)
        shown = min(rep.anchor_scan, key=lambda x: x.get("floor", x["predicted_floor"])) if rep.anchor_scan else None
        rep.certificate = Certificate(
            anchor=best_a, device_ratio=r_hat, predicted_g=0.0, measured_g=None,
            anchor_floor=(shown.get("floor") if shown else float("nan")) or float("nan"),
            achieved=float("inf"), evidence="anchor-limited",
            extrapolation_factor=float(needed / best_a) if best_a else 0.0,
            device_cost=cost.device_cost(r_hat), admissible=False,
            flags=["no anchor at or above %.0f has a floor below tau=%g" % (needed, tau)])
        rep.measured = oracle.known_ratios(feature)
        rep.spent_units = oracle.spent - spent0
        rep.baseline_units = oracle.price(14)
        rep.choice = ValueChoice(
            question=("The cheapest device needs r=%.1f, so the anchor must be at least %.0f, but "
                      "no anchor that fine has a floor below tau=%g. The tolerance cannot be "
                      "certified at this device rate; the three ways out are not equivalent."
                      % (r_hat, needed, tau)),
            options=[
                {"name": "measure-finer-anchor",
                 "label": "pay to measure an anchor at r=%.0f" % (needed * 2),
                 "search_units": int(oracle.price(1)),
                 "gives": "a chance that the floor clears tau at a finer anchor",
                 "risk": "the floor may not clear it: D(a) decays too slowly for this definition"},
                {"name": "accept-unsupported", "label": "report the tolerance as met anyway",
                 "search_units": 0, "gives": "device cost %.1f now" % cost.device_cost(r_hat),
                 "risk": "the anchor floor is above tau, so the claim is arithmetically false"},
                {"name": "relax-tolerance", "label": "relax tau to the measured floor",
                 "search_units": 0,
                 "gives": "an honest, weaker requirement",
                 "risk": "states a tolerance the caller did not ask for"},
            ],
            context={"r_star": r_hat, "needed_anchor": needed, "tau": tau,
                     "anchor_scan": rep.anchor_scan})
        return rep
    anchor = chosen

    for _ in range(MAX_ATTEMPTS):
        known = oracle.known_ratios(feature)
        # avoid paying twice for a ratio we already have: snap when it is within 3%
        near = [k for k in known if abs(k - r_hat) / r_hat < 0.03]
        if near:
            r_hat = min(near, key=lambda k: abs(k - r_hat))
        extrap = max(r_hat / max(known), 1.0)
        # predict g(r; a), not D(r): the certificate is about the anchor-relative deviation.
        # g = |F(r) - F(a)| / |F(a)| <= (D(r) + D(a)) / |F_inf|, with equality when the two
        # deficits have opposite signs; the bound is the conservative prediction.
        d_r = float(sub.predict([r_hat])[0])
        d_a = oracle.floor(feature, anchor) * scale      # D(a), recovered from the floor
        predicted = (d_r + d_a) / scale
        rep.attempts.append({"r": float(r_hat), "predicted_deficit": predicted,
                             "extrapolation_factor": round(float(extrap), 3)})
        if extrap > MAX_EXTRAPOLATION and accept_choice is None:
            rep.certificate = Certificate(
                anchor=anchor, device_ratio=r_hat, predicted_g=predicted, measured_g=None,
                anchor_floor=oracle.floor(feature, anchor), achieved=predicted,
                evidence="extrapolated", extrapolation_factor=extrap,
                device_cost=cost.device_cost(r_hat), admissible=False,
                flags=["%.1fx beyond the measured range" % extrap])
            rep.measured = oracle.known_ratios(feature)
            rep.spent_units = oracle.spent - spent0
            rep.baseline_units = oracle.price(14)
            rep.choice = _make_choice(rep, anchor, r_hat, predicted, extrap, tau, cost, oracle)
            return rep
        if (oracle.spent - spent0) + oracle.price(1) > budget_units:
            break
        g = oracle.g(feature, r_hat, anchor)
        rep.attempts[-1]["measured_g"] = g
        rep.attempts[-1]["held"] = bool(g <= tau)
        cert = Certificate(anchor=anchor, device_ratio=r_hat, predicted_g=predicted,
                           measured_g=g, anchor_floor=oracle.floor(feature, anchor),
                           achieved=g, evidence="measured", extrapolation_factor=extrap,
                           device_cost=cost.device_cost(r_hat), admissible=bool(g <= tau))
        if g <= tau:
            break
        r_hat = min(ladder_top, r_hat * 1.5)

    rep.measured = oracle.known_ratios(feature)
    rep.spent_units = oracle.spent - spent0
    if len(rep.measured) >= 3:
        rs = np.array(rep.measured)
        ds = np.array([oracle.deficit(feature, r) for r in rs])
        rep.order = bic_order(ds, rs)
    rep.certificate = cert
    floor = oracle.floor(feature, anchor)
    for r in rep.measured:
        rep.frontier.append({"device_ratio": round(r, 4), "device_cost": cost.device_cost(r),
                             "achieved_tolerance": max(oracle.g(feature, r, anchor), floor),
                             "evidence": "measured"})
    rep.frontier = _frontier(rep.frontier)

    # the defensive baseline: a twelve-point sweep out to the anchor, the reference, and
    # every anchor candidate — what an auditor who does not trust a surrogate would spend
    top = max([cert.device_ratio if cert else ladder_top, anchor])
    sweep = np.geomspace(float(min(pilot.ratios)), top, 12)
    rep.baseline_units = oracle.price(len(sweep) + 1 + len(anchors))
    return rep


def _make_choice(rep, anchor, r_hat, predicted, extrap, tau, cost, oracle) -> ValueChoice:
    extra = oracle.price(1)
    return ValueChoice(
        question=("The fitted curve puts the crossing at r=%.1f, which is %.1fx beyond anything "
                  "measured. Believing it is a judgement about risk, not a measurement."
                  % (r_hat, extrap)),
        options=[
            {"name": "measure-it", "label": "pay to measure r=%.1f" % r_hat,
             "search_units": int(extra),
             "gives": "an admissible certificate at device cost %.1f" % cost.device_cost(r_hat),
             "risk": "none beyond the audit's own scope"},
            {"name": "accept-extrapolation", "label": "accept the predicted crossing",
             "search_units": 0,
             "gives": "device cost %.1f now" % cost.device_cost(r_hat),
             "risk": "predicted deviation %.4g from a %s fit of quality %s"
                     % (predicted, rep.surrogate_model,
                        None if rep.surrogate_quality is None
                        else round(rep.surrogate_quality, 3))},
            {"name": "relax-tolerance", "label": "relax tau to %.0f%%" % (tau * 4 * 100),
             "search_units": 0, "gives": "an admissible plan inside the measured range",
             "risk": "states a weaker requirement than the caller wanted"},
        ],
        context={"anchor": anchor, "r_star": r_hat, "tau": tau, "order": rep.order},
    )


def record_decision(report: SearchReport, option: str, path: str, decided_by: str = "") -> dict:
    """Persist a value choice so it can be audited later."""
    if report.choice is None:
        raise ValueError("this report did not require a value choice")
    chosen = report.choice.choose(option)
    payload = {"feature": report.feature, "tau": report.tau,
               "decision": chosen, "chosen_by": decided_by,
               "chosen_from": [o["name"] for o in report.choice.options],
               "report": report.as_dict()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return payload
