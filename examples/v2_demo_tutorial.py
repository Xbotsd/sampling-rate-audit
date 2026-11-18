#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V2.1.1-demo tutorial: run one audit, fit the decay, check the tolerance.

中文说明
========
这是一个"保姆级"教程脚本，带一个完整的叙事：

1. 造一段假的生理/工程信号（若干带限正弦 + 白噪声，模拟真实记录）；
2. 用 ``Audit`` 把同一个信号重采样到一串更低的采样率，量出每个特征定义的
   配对亏空（这一步是 V1 就有的，逻辑没有改动）；
3. 用 ``estimator`` 把亏空对采样率拟合成 ``d = C·r^(−p)``，打印经验阶数；
4. 用 ``DemoGuard`` 把公差之外的预测标出来——**只打印，不拦截**；
5. 用 ``visualizer`` 画出散点、拟合曲线和三色背景区，存成 PNG。

运行
----
    python examples/v2_demo_tutorial.py                # 默认：出图到当前目录
    python examples/v2_demo_tutorial.py --out fig.png  # 指定输出文件
    python examples/v2_demo_tutorial.py --no-plot      # 只打印，不出图

依赖
----
    numpy, scipy（拟合）, matplotlib（绘图）
    pip install -e ".[demo]"

预期
----
你会看到一张表（每个特征的 p、C、残差）、一条或几条黄色 Demo 警告，以及
一张三色背景的衰减曲线图。

重要提醒
--------
这个脚本里没有任何东西会被"保护"或"拦截"。DemoGuard 只负责告诉你"这条
预测线越界了"，然后继续往下跑。真正部署时，越界的操作点应当**重新测量**，
而不是相信一个外推出来的指数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许直接从仓库根目录运行（不必先 pip install）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakage_audit import Audit, __version__                      # noqa: E402
from leakage_audit.demoguard import DemoGuard                      # noqa: E402
from leakage_audit.visualizer import plot_decay                    # noqa: E402

TOLERANCE = 0.01        # 用户公差：配对亏空不超过 1 %
FS_NATIVE = 500.0       # 记录的原生采样率，Hz
BAND_HZ = 20.0          # 声明的信号内容带宽，Hz
DURATION_S = 10.0       # 记录时长，秒
N_RECORDS = 8           # 用于 AUC 的独立记录数


def make_signal(fs=FS_NATIVE, band=BAND_HZ, duration=DURATION_S, seed=0):
    """A band-limited multi-tone signal plus white noise.

    多频段 + 噪声：既有"跟着网格走"的成分（过零点、转折点），也有
    "跟着信号走"的成分（标准差、谱质心），这样一次审计就能同时看到
    强泄露和弱泄露两类特征，教学对比才明显。
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    x = np.zeros(n)
    # 六个分量：三个固定在低频，三个扫到带宽边缘
    for f, a in [(1.2, 1.0), (3.7, 0.7), (8.2, 0.5), (12.5, 0.35),
                 (17.0, 0.25), (band * 0.95, 0.15)]:
        x += a * np.cos(2.0 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x += 0.05 * rng.standard_normal(n)          # 测量噪声
    return x


def make_signals(count=N_RECORDS, **kwargs):
    return [make_signal(seed=k, **kwargs) for k in range(count)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="v2_demo_decay.png", help="figure path")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    ap.add_argument("--method", default="mle", choices=["nls", "mle", "huber"])
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    bar = "=" * 72
    print(bar)
    print("  sampling-rate-audit  %s   (order fit: descriptive, not evidence)" % __version__)
    print("  This script runs without strict mode: the fit is descriptive only.")
    print(bar)

    # ---------------------------------------------------------------- signal
    signal = make_signal(seed=args.seed)
    print("\n[1/5] signal")
    print("      native fs = %g Hz, declared band B = %g Hz, duration = %g s"
          % (FS_NATIVE, BAND_HZ, DURATION_S))
    print("      samples = %d, RMS = %.4f" % (signal.size, float(np.sqrt(np.mean(signal ** 2)))))

    # ----------------------------------------------------------------- audit
    # 从原生速率向下走：r = fs / B 从 25 一直降到 1.56
    rates = [250.0, 125.0, 62.5, 50.0, 40.0, 31.25]
    engine = Audit(estimator_method=args.method)
    result = engine.run(signal, fs=FS_NATIVE, band=BAND_HZ, rates=rates)
    print("\n[2/5] paired audit (%d rate cells, band B = %.1f Hz)"
          % (len(rates), result.band))
    print("      " + result.markdown().replace("\n", "\n      "))

    # ------------------------------------------------------------------- fit
    print("\n[3/5] empirical order  d = C * r^(-p)   estimator = %s" % args.method)
    print("      " + result.fit_markdown().replace("\n", "\n      "))
    fits = {f: fit for f, fit in result.fits.items() if fit is not None and fit.ok}
    if not fits:
        print("      no feature had two informative rate cells; nothing to fit.")
    for feature in sorted(fits)[:5]:
        print("      " + "%s -> %s" % (feature, fits[feature].summary()))

    # ---------------------------------------------------------------- guard
    print("\n[4/5] demo guard (tolerance = %.4g)" % args.tolerance)
    guard = DemoGuard(tolerance=args.tolerance)
    per_feature = {}
    for feature, fit in fits.items():
        # 最坏点是最低的 r：亏空随 r 上升而下降
        r_grid = np.array([min(rates) / result.band])
        code = guard.check_fit(fit, r_grid=r_grid)
        per_feature[feature] = code
    warned = sorted(f for f, c in per_feature.items() if c == "WARNING")
    print("      %d of %d fitted features predict a deficit above tolerance"
          % (len(warned), len(fits)))
    if warned:
        print("      " + ", ".join(warned))
    print("      " + guard.summary())

    # ----------------------------------------------------------------- plot
    if args.no_plot:
        print("\n[5/5] plotting skipped (--no-plot)")
    else:
        # Let the visualiser choose: it prefers well-supported fits (three or
        # more rate cells) and then the largest median deficit, which keeps
        # degenerate two-point fits out of the picture.
        plot_decay(result, features=None, tolerance=args.tolerance,
                   out_path=args.out, x_key="r")
        print("\n[5/5] figure -> %s" % Path(args.out).resolve())
        print("      green = predicted deficit under tolerance; amber = up to 3x;")
        print("      red = beyond that. Right of the dotted line is extrapolation.")

    print("\n" + bar)
    print("  Reminder: the exponent describes the rate range you audited.")
    print("  Re-measure at any rate you intend to deploy at.")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
