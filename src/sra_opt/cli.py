"""Command line for the optimiser layer.

Three subcommands, because there are three different things a caller wants:

* ``search`` — find the cheapest grid that meets a tolerance, and report the certificate;
* ``dp`` — the Deadlock Protection check, whose exit code is the CI contract (0 / 3 / 4);
* ``demo`` — run the end-to-end demonstration and write the figures and the bug report.

The threshold argument is deliberately awkward to lower: it requires ``--acknowledge-risk``
and a signer, which writes the risk agreement instead of quietly accepting a weak surrogate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import Oracle, __version__
from .deadlock import EXIT_CODE, DeadlockGuard, DeadlockTriggered
from .pareto import record_decision, search
from .surrogate import DEFAULT_THRESHOLD, FLOOR_THRESHOLD


def _cmd_search(args) -> int:
    guard = DeadlockGuard(root=args.root)
    try:
        guard.guard_inputs({"tau": args.tau, "anchor_margin": args.margin,
                            "budget": args.budget})
    except DeadlockTriggered as exc:
        print("deadlock protection: %s" % exc.reason, file=sys.stderr)
        print(json.dumps(exc.detail, ensure_ascii=False, indent=1), file=sys.stderr)
        return exc.exit_code

    oracle = Oracle([args.feature], n_records=args.records, seed=args.seed)
    report = search(oracle, args.feature, tau=args.tau, budget_units=args.budget,
                    threshold=args.threshold)
    print(report.verdict())
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report.as_dict(), fh, indent=1, ensure_ascii=False)
        print("wrote", args.json)
    if report.choice is not None:
        print("\nthis answer needs a value choice:")
        for opt in report.choice.options:
            print("  %-20s %s  (units: %s)" % (opt["name"], opt["label"],
                                               opt["search_units"]))
        if args.choose:
            path = args.decision or "decision.json"
            record_decision(report, args.choose, path, decided_by=args.by)
            print("recorded choice %r in %s" % (args.choose, path))
        else:
            print("  pass --choose <name> to record a decision")
    return 0


def _cmd_dp(args) -> int:
    guard = DeadlockGuard(root=args.root)
    try:
        guard.guard_inputs({"tau": args.tau})
        oracle = Oracle([args.feature], n_records=args.records, seed=args.seed)
        report = search(oracle, args.feature, tau=args.tau, budget_units=args.budget)
        if args.export:
            out = guard.guard_export(report)
            print("%s: r*=%.2f evidence=%s" % (args.feature,
                                               out.get("device_ratio", float("nan")),
                                               out.get("evidence", "?")))
        else:
            print("%s: %s" % (args.feature, report.verdict()))
        return 0
    except DeadlockTriggered as exc:
        print("deadlock protection: %s" % exc.reason, file=sys.stderr)
        for key, value in exc.detail.items():
            print("  %s: %s" % (key, value), file=sys.stderr)
        return EXIT_CODE
    except Exception as exc:  # noqa: BLE001 - the CI contract needs a distinct code
        print("unexpected error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 4


def _cmd_demo(args) -> int:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    demo = os.path.join(here, "examples", "optimizer_demo.py")
    if not os.path.exists(demo):
        print("the demo script ships in the source tree: examples/optimizer_demo.py",
              file=sys.stderr)
        return 4
    import runpy
    runpy.run_path(demo, run_name="__main__")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sra-opt",
        description="Search for the cheapest sampling grid that meets a stated tolerance.")
    ap.add_argument("--version", action="version", version="sra-opt %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--feature", default="hjorth_mobility_Hz")
        p.add_argument("--tau", type=float, default=0.02)
        p.add_argument("--budget", type=int, default=400, help="cost units")
        p.add_argument("--records", type=int, default=4)
        p.add_argument("--seed", type=int, default=11)
        p.add_argument("--root", default=".", help="where .sra_deadlock is written")

    s = sub.add_parser("search", help="find the cheapest certifiable grid")
    common(s)
    s.add_argument("--margin", type=float, default=2.0,
                   help="how far above r* the anchor must sit")
    s.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="surrogate acceptance threshold (floor %.2f)" % FLOOR_THRESHOLD)
    s.add_argument("--json", default="", help="write the full report here")
    s.add_argument("--choose", default="", help="record a value choice by name")
    s.add_argument("--decision", default="", help="where to write the decision record")
    s.add_argument("--by", default="", help="who decided")
    s.set_defaults(func=_cmd_search)

    d = sub.add_parser("dp", help="Deadlock Protection check; exit 3 when it fires")
    common(d)
    d.add_argument("--export", action="store_true",
                   help="attempt to emit the certificate")
    d.set_defaults(func=_cmd_dp)

    m = sub.add_parser("demo", help="run the end-to-end demonstration")
    m.set_defaults(func=_cmd_demo)

    args = ap.parse_args(argv)
    if getattr(args, "threshold", DEFAULT_THRESHOLD) < DEFAULT_THRESHOLD:
        print("note: threshold %.2f is below the default %.2f; a risk agreement should be "
              "signed (see sra_opt.surrogate.sign_risk_agreement)"
              % (args.threshold, DEFAULT_THRESHOLD), file=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
