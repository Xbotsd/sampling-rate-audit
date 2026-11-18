"""Command line interface:  leakage-audit --selftest  |  --npy ... --fs ..."""
import argparse
import math
import os
import sys

import numpy as np

from . import Audit, REGISTRY, ORDER_OF, EPS, __version__, resample_fft

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def selftest(n_real=20):
    rng = np.random.default_rng(7)
    T, B, fs_hi = 10.0, 20.0, 8000.0
    n = int(T * fs_hi)
    t = np.arange(n) / fs_hi
    sigs = []
    for _ in range(n_real):
        x = np.zeros(n)
        for _ in range(25):
            k = rng.integers(5, 201)
            x += rng.rayleigh(1.0) * np.cos(2 * np.pi * (k / T) * t + rng.uniform(0, 2 * np.pi))
        sigs.append(x)
    print("%-26s%15s   expected" % ("feature", "bias exponent"))
    for f in ["turning_per_second", "zcr_per_second", "hjorth_mobility_Hz",
              "spec_entropy", "turning_per_sample", "zcr_per_sample", "std"]:
        fn = REGISTRY[f][0]
        refs = [fn(x, fs_hi) for x in sigs]
        rs, bs = [], []
        any_above = False
        for r in [2.5, 3.125, 5, 6.25, 12.5, 25, 50]:
            vals = []
            for x, ref in zip(sigs, refs):
                y = resample_fft(x, fs_hi, r * B, lp_hz=B)
                vals.append(abs(fn(y, r * B) - ref) / (abs(ref) + EPS))
            b = float(np.median(vals))
            if b > 1e-7:
                any_above = True
                rs.append(math.log(r))
                bs.append(math.log(b))
        if not any_above:
            p = float("inf")
        else:
            p = -np.polyfit(rs, bs, 1)[0] if len(rs) >= 3 else float("nan")
        exp = ORDER_OF[REGISTRY[f][1]]
        ps = "inf" if p == float("inf") else "%.2f" % p
        print("%-26s%15s   %s" % (f, ps, "inf" if exp == float("inf") else "%g" % exp))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Sampling-rate leakage audit")
    ap.add_argument("--npy", help="1-D .npy of samples")
    ap.add_argument("--csv", help="single-column CSV of samples")
    ap.add_argument("--dir", help="directory of .npy files, one per record")
    ap.add_argument("--fs", type=float, default=None, help="sampling rate, Hz")
    ap.add_argument("--band", type=float, default=None, help="content bandwidth B, Hz")
    ap.add_argument("--rates", type=float, nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="enable deadlock protection: abort when a fitted curve "
                         "predicts a deficit above the tolerance")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="admissible paired deficit for --strict (default 0.01)")
    ap.add_argument("--i-understand-risks", action="store_true",
                    help="skip the protocol prompt (records an explicit "
                         "acknowledgement); the sandbox and the guard stay on")
    ap.add_argument("--agreement-cache", default=None,
                    help="where to keep the protocol acknowledgement "
                         "(default: .sra_agreement)")
    ap.add_argument("--deadlock-file", default=None,
                    help="where to append deadlock events (default: .sra_deadlock)")
    ap.add_argument("--version", action="version", version="leakage-audit %s" % __version__)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.fs:
        ap.error("--fs is required")

    guard = None
    if a.strict:
        from .formalguard import FormalGuard, require_agreement

        if not a.i_understand_risks:
            try:
                require_agreement(cache_path=a.agreement_cache)
            except RuntimeError as exc:
                # The protocol was refused (no terminal, or the paste was too
                # fast). Print the reason once and stop; no traceback for a
                # predictable refusal.
                print("strict mode refused: %s" % exc, file=sys.stderr)
                raise SystemExit(2)
        guard = FormalGuard(a.tolerance, state_path=a.deadlock_file,
                            strict=True)

    eng = Audit(formal_guard=guard, tolerance=a.tolerance if a.strict else None)

    from .formalguard import DeadlockError

    try:
        if a.dir:
            files = sorted(f for f in os.listdir(a.dir) if f.endswith(".npy"))
            sigs = [np.load(os.path.join(a.dir, f)).astype(float).ravel() for f in files]
            res = eng.run_many(sigs, a.fs, band=a.band, rates=a.rates)
            print("records:", len(sigs))
        elif a.npy:
            res = eng.run(np.load(a.npy).astype(float).ravel(), a.fs,
                          band=a.band, rates=a.rates)
        elif a.csv:
            res = eng.run(np.loadtxt(a.csv, delimiter=",").astype(float).ravel(),
                          a.fs, band=a.band, rates=a.rates)
        else:
            ap.error("need --npy, --csv or --dir (or --selftest)")
    except DeadlockError:
        # 沙盒已经回档，库内也打印过说明。命令行这里只负责给出一个固定的非零
        # 退出码，避免把契约异常埋进 traceback 里。程序化调用方拿到的仍然是
        # DeadlockError 本身——库没有吞掉它。
        print("strict mode: run aborted, exit code 3", file=sys.stderr)
        raise SystemExit(3)

    print("band B = %.3f Hz   fs = %g Hz   mode = %s" % (res.band, res.fs, res.mode))
    print(res.markdown())
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        res.to_csv(os.path.join(a.out, "audit.csv"))
        res.to_json(os.path.join(a.out, "audit.json"))
        print("wrote", os.path.join(a.out, "audit.csv"))
