"""End-to-end run of the 3.1.1 optimiser layer.

Produces, in ``out/``:

* ``search.json`` — the per-feature certificates, with measured/predicted labels and the
  value choices the search declined to make;
* ``arbitrator.json`` — the fitted track-selection model and its training report;
* ``eld.json`` — the sandbox and Evil-Lock-Detector demonstration;
* ``budget.json`` — the reserve, the hidden set, and where the reserve was spent;
* ``heatmap.png``, ``waterfall.png`` — the two figures;
* ``.sra_bug_report`` — seed, config, versions and the replay command.

Run with the bundled interpreter; see V311_DESIGN.md for the dependency list.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sra_opt import Oracle, feature_menu  # noqa: E402
from sra_opt.arbitrator import Arbitrator, build_training_cases  # noqa: E402
from sra_opt.budget import Budget, hidden_set, save as save_json, spend_option  # noqa: E402
from sra_opt.deadlock import DeadlockGuard, DeadlockTriggered  # noqa: E402
from sra_opt.eld_demo import demo as eld_demo  # noqa: E402
from sra_opt.pareto import record_decision, search  # noqa: E402
from sra_opt.surrogate import Surrogate  # noqa: E402
from sra_opt.visualizer_v3 import heatmap, waterfall, write_bug_report  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEED = 7
FEATURES = ["hjorth_mobility_Hz", "turning_per_second", "spec_entropy", "zcr_per_sample", "std"]
TAUS = {"hjorth_mobility_Hz": 0.02, "turning_per_second": 0.02,
        "spec_entropy": 0.05, "zcr_per_sample": 0.05, "std": 0.01}


def run_searches() -> dict:
    oracle = Oracle(FEATURES, n_records=6, seed=SEED)
    budget = Budget(total_units=2400)
    out = {}
    for feature in FEATURES:
        rep = search(oracle, feature, tau=TAUS[feature], budget_units=budget.main_left)
        budget.charge(rep.spent_units, "main")
        entry = rep.as_dict()
        # a required value choice is recorded with the conservative option, not silently skipped
        if rep.choice is not None:
            path = os.path.join(OUT, "decisions", "%s_choice.json" % feature)
            record_decision(rep, "measure-it", path, decided_by="default-conservative")
            entry["decision_record"] = os.path.relpath(path, OUT)
        out[feature] = entry
    out["_budget"] = budget.as_dict()
    out["_spent"] = oracle.spent
    return out


def run_arbitration() -> dict:
    path = os.path.join(OUT, "arbitrator.json")
    report_path = os.path.join(OUT, "arbitrator_report.json")
    if os.path.exists(path) and os.path.exists(report_path):
        return json.load(open(report_path, encoding="utf-8"))
    cases = build_training_cases(["hjorth_mobility_Hz", "turning_per_second", "std"],
                                 taus=(0.05, 0.02, 0.01), n_records=3, seed=SEED + 1)
    arb = Arbitrator.fit(cases)
    arb.save(path)
    closer = sum(1 for c in cases if c["beta_closer"])
    return {"loaded_from": "fresh fit", "n_cases": len(cases),
            "beta_closer": closer, "alpha_closer": len(cases) - closer,
            "cases": [{"feature": c["feature"], "tau": c["tau"],
                       "alpha": round(c["alpha"], 3), "beta": round(c["beta"], 3),
                       "truth": round(float(c["truth"]), 3),
                       "beta_closer": bool(c["beta_closer"])}
                      for c in cases],
            **arb.as_dict()}


def run_figures(search_out: dict) -> dict:
    oracle = Oracle(FEATURES, n_records=3, seed=SEED)
    ratios = np.array([2.5, 4.0, 6.25, 12.5, 25.0, 50.0, 100.0, 200.0, 400.0])
    # relative deficit, so features on different scales are comparable on one ramp
    matrix = np.zeros((len(FEATURES), len(ratios)))
    for i, f in enumerate(FEATURES):
        scale = oracle.scale(f)
        for j, r in enumerate(ratios):
            d = oracle.deficit(f, r) / scale
            matrix[i, j] = np.log10(max(d, 1e-9))
    hm = heatmap(os.path.join(OUT, "heatmap.png"), FEATURES, ratios, matrix,
                 vmin=-5.0, vmax=1.0)

    # waterfall: how much each measurement moved the reported ratio, by leave-one-out
    focus = "hjorth_mobility_Hz"
    tau = TAUS[focus]
    pilot = np.array([2.5, 4.0, 6.25, 12.5, 25.0])
    d = np.array([oracle.deficit(focus, r) for r in pilot])
    full = Surrogate(pilot, np.maximum(d, 1e-15), focus)
    scale = oracle.scale(focus)

    def r_star(sub) -> float:
        """Continuous crossing by bisection, so leave-one-out differences are not quantised."""
        lo, hi = 2.5, 200.0
        if float(sub.predict([hi])[0]) > tau * scale:
            return float("nan")
        if float(sub.predict([lo])[0]) <= tau * scale:
            return float(lo)
        for _ in range(60):
            mid = float(np.sqrt(lo * hi))
            if float(sub.predict([mid])[0]) <= tau * scale:
                hi = mid
            else:
                lo = mid
        return float(hi)

    base = r_star(full)
    contribs = []
    for k, r in enumerate(pilot):
        keep = np.arange(len(pilot)) != k
        loo = Surrogate(pilot[keep], np.maximum(d[keep], 1e-15), focus)
        delta = np.log10(r_star(loo)) - np.log10(base)
        contribs.append({"label": "without the r=%g measurement" % r, "delta": float(delta)})
    contribs.sort(key=lambda c: -abs(c["delta"]))
    wf = waterfall(os.path.join(OUT, "waterfall.png"), contribs, baseline=base)

    return {"heatmap": os.path.basename(hm), "waterfall": os.path.basename(wf),
            "waterfall_focus": focus, "base_ratio": round(base, 3),
            "contributions": contribs}


def run_deadlock(searches: dict) -> dict:
    """Exercise DP on the real reports: three exports, two of which must be refused."""
    guard = DeadlockGuard(root=OUT)
    results = []
    for feature in ("hjorth_mobility_Hz", "zcr_per_sample", "spec_entropy"):
        class _R:
            pass
        rep = _R()
        rep.feature = feature
        rep.tau = TAUS[feature]
        rep.measured = searches[feature]["measured_ratios"]
        rep.verdict = lambda f=feature: searches[f]["verdict"]
        cert = searches[feature]["certificate"]
        if cert is None:
            rep.certificate = None
        else:
            class _C:
                pass
            c = _C()
            for k, v in cert.items():
                setattr(c, k, v)
            rep.certificate = c
        try:
            out = guard.guard_export(rep)
            results.append({"feature": feature, "outcome": "emitted", **out})
        except DeadlockTriggered as exc:
            results.append({"feature": feature, "outcome": "deadlock",
                            "reason": exc.reason, "exit_code": exc.exit_code,
                            "detail": {k: v for k, v in exc.detail.items()
                                       if k in ("verdict", "evidence", "flags")}})
    # and one dirty input, to show the input guard as well
    try:
        guard.guard_inputs({"tau": -1.0})
        dirty = {"outcome": "not caught"}
    except DeadlockTriggered as exc:
        dirty = {"outcome": "deadlock", "reason": exc.reason,
                 "violations": exc.detail["violations"]}
    return {"exports": results, "dirty_input": dirty,
            "exit_code": DeadlockGuard.exit_code_value if hasattr(DeadlockGuard, "exit_code_value") else 3,
            "trace": os.path.basename(guard.trace_path),
            "boundary": "exploration rolls back; only dirty input and unbacked export deadlock"}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print("1/5  search")
    searches = run_searches()
    print("2/5  arbitration")
    arb = run_arbitration()
    print("3/5  sandbox, ELD and deadlock protection")
    eld = eld_demo(os.path.join(OUT, "eld"))
    dp = run_deadlock(searches)
    print("4/5  budget option and figures")
    figs = run_figures(searches)
    oracle = Oracle(FEATURES, n_records=6, seed=SEED)
    pilot_d = np.array([oracle.deficit("hjorth_mobility_Hz", r) for r in (2.5, 4.0, 6.25, 12.5, 25.0)])
    sub = Surrogate(np.array([2.5, 4.0, 6.25, 12.5, 25.0]), np.maximum(pilot_d, 1e-15),
                    "hjorth_mobility_Hz")
    hidden = hidden_set(sub.spread, np.geomspace(2.5, 200.0, 40))
    budget = Budget(total_units=2400)
    budget.spent_main = searches["_spent"]
    option = spend_option(budget, "math", hidden and budget.reserve_units,
                          "hjorth_mobility_Hz", "refine the crossing after the report")
    budget_doc = {"hidden": hidden, "option": option, "budget": budget.as_dict()}
    save_json(budget_doc, os.path.join(OUT, "budget.json"))

    print("5/5  write")
    save_json(searches, os.path.join(OUT, "search.json"))
    save_json(arb, os.path.join(OUT, "arbitrator_report.json"))
    save_json(eld, os.path.join(OUT, "eld.json"))
    save_json(dp, os.path.join(OUT, "deadlock.json"))

    report = {"searches": searches, "arbitration": arb, "eld": eld, "deadlock": dp,
              "budget": budget_doc, "figures": figs}
    write_bug_report(os.path.join(OUT, ".sra_bug_report"), report, SEED,
                     "python examples/optimizer_demo.py",
                     inputs={"features": FEATURES, "taus": TAUS,
                             "engine": "leakage_audit 1.0.1"})

    print("\nDP exports:", [(r["feature"], r["outcome"]) for r in dp["exports"]])
    ok = sum(1 for f in FEATURES if searches[f]["certificate"] and
             searches[f]["certificate"]["admissible"])
    print("\n%-22s %-10s %s" % ("feature", "tau", "verdict"))
    for f in FEATURES:
        print("%-22s %-10g %s" % (f, TAUS[f], searches[f]["verdict"][:100]))
    print("\nadmissible certificates: %d/%d" % (ok, len(FEATURES)))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
