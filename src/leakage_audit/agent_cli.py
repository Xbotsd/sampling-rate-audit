"""Command line front-end for the invariance agent.

    invariance-agent plan    --expr "crossings divided by len(x)"
    invariance-agent profile --fs 500 --band 40
    invariance-agent audit   --npy rec.npy --fs 500 --band 40
    invariance-agent full    --npy rec.npy --fs 500 --band 40 --tolerance 0.01
    invariance-agent schema
    invariance-agent wizard
"""
from __future__ import annotations

import argparse
import os
import sys

from . import agent


def _print_plan(p):
    print("signature      : %s  (%s confidence, %s)" % (p["signature"], p["confidence"], p["reason"]))
    print("structural     : %s" % p["class"])
    prior = p["prior_exponent"]
    print("prior exponent : %s" % ("inf" if prior == float("inf") else "%g" % prior))
    print("expected       : %s" % p["expected"])
    print("do first       : %s" % p["before_measuring"])
    if not p["ratio_helps"]:
        print("ratio helps?   : NO - a higher sampling rate does not remove this dependence")
    else:
        print("ratio helps?   : possibly - measure the curve")
    print("note           : %s" % p["note"])


def _print_advice(a):
    print("tolerance      : %.3g" % a["tolerance"])
    print("audited ratios : %s" % ", ".join("%g" % x for x in a.get("audited_ratios", [])))
    for w in a.get("warnings", []):
        print("WARNING        : %s" % w)
    for f in a["features"]:
        print("-" * 72)
        print("%-24s class=%s" % (f["feature"], f["structural_class"]))
        exp = f["measured_exponent"]
        print("  measured exponent : %s" % ("-" if exp is None else exp))
        dev = f["deviation_at_top_ratio"]
        print("  deviation at r=%.3g : %s" % (
            f["top_ratio"] or 0, "-" if dev is None else "%.3e" % dev))
        need = f["required_ratio_for_tolerance"]
        if f["no_ratio_helps"]:
            print("  required ratio     : none (no ratio helps)")
        else:
            print("  required ratio     : %s" % ("-" if need is None else need))
        print("  verdict            : %s" % f["verdict"])
        print("  action             : %s" % f["action"])
    print("-" * 72)
    print(a["caveat"])


def _print_profile(d):
    print("sampling rate  : %.4g Hz" % d["fs_hz"])
    print("declared band  : %.4g Hz" % d["band_hz"])
    print("ratio r = fs/B : %.3f" % d["oversampling_ratio"])
    print("nyquist ok     : %s" % d["nyquist_ok"])
    if d["samples_per_window"]:
        print("samples/window : %d" % d["samples_per_window"])
    for n in d["notes"]:
        print("WARNING        : %s" % n)
    print("\nB is a claim: %s" % d["declared_band_is_the_users_claim"])
    print(d["scope"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # `wizard` is interactive and takes no flags
    if argv and argv[0] == "wizard":
        return _wizard()

    ap = argparse.ArgumentParser(
        prog="invariance-agent",
        description="Decide whether a feature is comparable across sampling grids.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="predict the structural class from the definition")
    p_plan.add_argument("--feature", help="registry name, e.g. zcr_per_sample")
    p_plan.add_argument("--expr", help="free-text definition")
    p_plan.add_argument("--json", action="store_true")

    p_prof = sub.add_parser("profile", help="check acquisition settings")
    p_prof.add_argument("--fs", type=float, required=True)
    p_prof.add_argument("--band", type=float, required=True)
    p_prof.add_argument("--window", type=float, default=None)
    p_prof.add_argument("--lpf", type=float, default=None)
    p_prof.add_argument("--json", action="store_true")

    for name, helptext in (("audit", "measure the grid sensitivity"),
                           ("advise", "audit + per-feature recommendation"),
                           ("full", "plan + audit + recommendation")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--npy")
        sp.add_argument("--csv")
        sp.add_argument("--fs", type=float, required=True)
        sp.add_argument("--band", type=float, default=None)
        sp.add_argument("--rates", type=float, nargs="*", default=None)
        sp.add_argument("--tolerance", type=float, default=0.01)
        sp.add_argument("--features", nargs="*", default=None)
        sp.add_argument("--window", type=float, default=None)
        sp.add_argument("--json", action="store_true")

    sub.add_parser("schema", help="print the task/response contract")

    a = ap.parse_args(argv)

    if a.cmd == "schema":
        print(agent.dumps(agent.describe()))
        return 0

    if a.cmd == "plan":
        try:
            p = agent.plan(feature=a.feature, expr=a.expr)
        except (KeyError, ValueError) as exc:
            print("error:", exc, file=sys.stderr)
            return 2
        if a.json:
            print(agent.dumps(p))
        else:
            _print_plan(p)
        return 0

    if a.cmd == "profile":
        d = agent.device_profile(a.fs, a.band, window_s=a.window, analog_lpf_hz=a.lpf)
        if a.json:
            print(agent.dumps(d))
        else:
            _print_profile(d)
        return 0

    path = a.npy or a.csv
    if not path:
        print("error: pass --npy or --csv", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print("error: no such file %r" % path, file=sys.stderr)
        return 2

    task = {
        "action": a.cmd,
        "device": {"fs_hz": a.fs, "band_hz": a.band, "window_s": a.window},
        "data": {"path": path},
        "tolerance": a.tolerance,
    }
    if a.rates:
        task["rates"] = a.rates
    if a.features:
        task["features"] = a.features

    out = agent.run_task(task)
    if a.json:
        print(agent.dumps(out))
        return 0

    if a.cmd == "audit":
        print("band B = %.3f Hz   fs = %g Hz" % (out["audit"]["band_hz"], out["audit"]["fs_hz"]))
        print(out["audit"]["markdown"])
        print(out["audit"]["scope"])
    elif a.cmd == "advise":
        _print_advice(out["advice"])
    else:
        for p in out["plan"]:
            print("=" * 72)
            print("FEATURE:", p["class"])
            _print_plan(p)
        print("=" * 72)
        _print_advice(out["advice"])
    return 0


def _wizard():
    print("Invariance agent - first run\n" + "-" * 40)
    print("This walks you from your instrument to a recommendation.\n")

    def ask(prompt, cast, default=None):
        while True:
            raw = input("%s%s: " % (prompt, " [%s]" % default if default is not None else ""))
            raw = raw.strip()
            if not raw and default is not None:
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                print("  please enter a number")

    fs = ask("Sampling rate of your device (Hz)", float)
    band = ask("Content band you rely on (Hz)", float)
    window = ask("Analysis window (s)", float, default=30.0)
    path = input("Path to one record (.npy or .csv): ").strip().strip('"')
    tol = ask("Relative tolerance you can accept (e.g. 0.01 for 1%)", float, default=0.01)

    if not os.path.exists(path):
        print("Cannot find %r. Stopping." % path)
        return 2

    prof = agent.device_profile(fs, band, window_s=window)
    print()
    _print_profile(prof)
    print()

    out = agent.run_task({
        "action": "full",
        "device": {"fs_hz": fs, "band_hz": band, "window_s": window},
        "data": {"path": path},
        "tolerance": tol,
    })
    _print_advice(out["advice"])
    print("\nNext: repeat this on 5-20 records, then report the paired deviation "
          "margins together with the class of each feature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
