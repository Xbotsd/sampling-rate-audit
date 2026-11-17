"""Plot the paired decay: deficit versus sampling rate, with the fitted curve.

中文说明
--------
V1 只往终端打表。Demo 版加这一层，是为了让"亏空随采样率衰减"这件事能被
一眼看到：

* 散点 = 真实审计出来的配对亏空，每个点对应一个被审计过的采样率；
* 实线 = Estimator 拟合出的 ``C·r^(−p)``；
* 背景三色带 = 以用户公差为界划分的安全区 / 模糊区 / 死锁区。

三色带的边界不是拍脑袋画的，它来自拟合曲线与公差的两条水平线的交点：
亏空小于公差的地方是绿的，公差到 3 倍公差之间是黄的，再往外是红的。

Important
---------
背景色带表达的是**拟合曲线**的位置，不是测量结果的位置。在审计区间之外，
曲线是外推，色带也随之变成预测。图上会画一条竖虚线标出"审计到此为止"，
免得把外推当成测量。

matplotlib 是可选依赖，只在调用绘图函数时导入；核心包仍然只需要 numpy。
"""

from __future__ import annotations

import numpy as np

# Colour scheme, deliberately the same three names used by the demo docs.
SAFE_COLOUR = "#2e7d32"        # green
MARGINAL_COLOUR = "#f9a825"    # amber
DEAD_COLOUR = "#c62828"        # red
POINT_COLOUR = "#1565c0"       # blue
FIT_COLOUR = "#37474f"         # graphite


def _deficit_of(row):
    """The deficit column of one audit row.

    Both ``run`` and ``run_many`` write ``rel_bias``; older exports may carry
    only ``auc``, in which case the orientation-free leakage score
    ``2·|AUC − 0.5|`` is the comparable quantity.
    """
    if "rel_bias" in row and np.isfinite(row["rel_bias"]):
        return float(row["rel_bias"])
    if "auc" in row:
        return float(abs(row["auc"] - 0.5) * 2.0)
    return float("nan")


# A deficit of exactly zero cannot be drawn on a log axis.  Such a point is
# informative — it says the feature did not move at all — so it is drawn at a
# floor rather than dropped. 1e-16 is below any real measurement here.
ZERO_FLOOR = 1e-16


def _series(result, feature, x_key):
    """``(x, d)`` for one feature, sorted by x.

    Deficits that are non-finite are dropped; deficits that are exactly zero
    are kept and plotted at :data:`ZERO_FLOOR`, because "did not move" is a
    result, not missing data.
    """
    xs, ds = [], []
    for row in result.rows:
        if row.get("feature") != feature:
            continue
        d = _deficit_of(row)
        if not np.isfinite(d):
            continue
        if x_key == "r":
            xs.append(float(row["r"]))
        else:
            xs.append(float(row.get("fs", row["r"] * result.band)))
        ds.append(d if d > 0 else ZERO_FLOOR)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ds)[order]


def _bands(ax, r_grid, curve, tolerance, marginal_factor=3.0):
    """Shade safe / marginal / dead regions from the fitted curve.

    把网格点按"曲线值 vs 公差"分成三类，再把连续同类点合并成一段 axvspan。
    这样色带的边界就是曲线穿过公差线的位置，而不是某个写死的比例。
    """
    if tolerance is None or tolerance <= 0 or curve is None:
        return []
    with np.errstate(invalid="ignore"):
        ratio = curve / float(tolerance)
    cls = np.where(~np.isfinite(ratio), -1,
                   np.where(ratio <= 1.0, 0,
                            np.where(ratio <= marginal_factor, 1, 2)))
    palette = {0: SAFE_COLOUR, 1: MARGINAL_COLOUR, 2: DEAD_COLOUR}
    labels = {0: "safe (pred. deficit ≤ tolerance)",
              1: "marginal (≤ %.0f× tolerance)" % marginal_factor,
              2: "dead zone (> %.0f× tolerance)" % marginal_factor}

    spans, seen = [], set()
    start = 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[start]:
            kind = int(cls[start])
            if kind >= 0 and i > start:
                lo = r_grid[start] if start == 0 else 0.5 * (r_grid[start - 1] + r_grid[start])
                hi = r_grid[i - 1] if i == len(r_grid) else 0.5 * (r_grid[i - 1] + r_grid[i])
                ax.axvspan(lo, hi, color=palette[kind], alpha=0.13, zorder=0,
                           label=labels[kind] if kind not in seen else None)
                seen.add(kind)
                spans.append((kind, float(lo), float(hi)))
            start = i
    return spans


def plot_decay(result, features=None, x_key="r", tolerance=None,
               out_path=None, ax=None, log_log=True, title=None, dpi=180,
               annotate=True, figsize=(7.2, 4.4), band_feature=None,
               ylim=None, min_points=3):
    """Draw the decay curve for one or more features of an audit result.

    Parameters
    ----------
    result : leakage_audit.AuditResult
        The object returned by ``Audit.run`` / ``Audit.run_many``.
    features : sequence of str, optional
        Which features to draw.  Defaults to the three with the largest
        median deficit among those with at least ``min_points`` informative
        cells, which is usually what the reader wants to see first.
    x_key : {"r", "fs"}
        Plot the over-sampling ratio (default) or the raw sampling rate.
    tolerance : float, optional
        Draws the coloured bands and the tolerance line.  Pass the same number
        you gave to :class:`~leakage_audit.demoguard.DemoGuard`.
    band_feature : str, optional
        Which feature's fitted curve defines the colour bands.  With several
        curves on one axes the bands can only follow one of them; the default
        picks the best-supported fit (most points, then median exponent) and
        the legend says so.  Pass a name to overrule that.
    ylim : (low, high), optional
        Explicit ordinate limits.  Without it the limits come from the measured
        points, not from the fitted curves: a two-point fit with a large
        exponent otherwise drags the axis to 1e-16 and flattens everything.
    out_path : path, optional
        Save the figure here and close it.  Without it the figure is returned
        for further use (``fig.savefig(...)``).
    ax : matplotlib Axes, optional
        Draw into an existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if out_path is not None:
        # 服务器/CI 上没有显示设备，必须显式选一个不依赖 DISPLAY 的后端
        matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    rows = getattr(result, "rows", None) or []
    if not rows:
        raise ValueError("the audit result has no rows to plot")

    all_features = sorted({row["feature"] for row in rows})
    fits = getattr(result, "fits", {}) or {}

    def _usable(feature):
        fit = fits.get(feature)
        return fit is not None and getattr(fit, "ok", False) and fit.n >= min_points

    if features is None:
        ranked = []
        for f in all_features:
            x_f, d = _series(result, f, x_key)
            if d.size:
                ranked.append((_usable(f), float(np.median(d)), f))
        # Prefer well-supported fits over degenerate ones.
        ranked.sort(key=lambda t: (t[0], t[1]))
        # Then take a *spread* rather than the three largest: the point of the
        # figure is that definitions behave differently, and three features all
        # sitting in the same decade would hide that.
        spread = [t for t in ranked if t[0]] or ranked
        if len(spread) <= 3:
            features = [f for _, _, f in spread]
        else:
            idx = np.linspace(0, len(spread) - 1, 3).round().astype(int)
            features = [spread[i][2] for i in sorted(set(idx.tolist()))]
        features = features or all_features[:3]
    features = list(features)
    unknown = [f for f in features if f not in all_features]
    if unknown:
        raise KeyError("feature(s) not in this result: %s" % ", ".join(unknown))

    # Global abscissa range, so bands and the extrapolation marker agree.
    series_cache = {f: _series(result, f, x_key) for f in features}
    usable_x = [s[0] for s in series_cache.values() if s[0].size]
    if not usable_x:
        raise ValueError("none of the requested features has a plottable point: %s"
                         % ", ".join(features))
    all_x = np.concatenate(usable_x)
    x_lo, x_hi = float(all_x.min()), float(all_x.max())

    # ---- the curve that defines the bands ---------------------------------
    # The bands are only informative for a curve that actually crosses the
    # tolerance inside the plotted range: that is where "the tolerance is met
    # from this rate rightward" becomes a visible statement.  A curve that sits
    # wholly above or wholly below just paints the whole axes one colour.
    if band_feature is None and tolerance:
        candidates = [f for f in features if _usable(f)]
        crossing = []
        for f in candidates:
            fit = fits[f]
            probe = np.geomspace(x_lo, x_hi, 128)
            with np.errstate(invalid="ignore"):
                vals = fit.predict(probe)
            if not np.any(np.isfinite(vals)):
                continue
            inside = bool(np.any(vals <= tolerance) and np.any(vals > tolerance))
            crossing.append((inside, fit.n, -abs(fit.p - 1.0), f))
        if crossing:
            crossing.sort(reverse=True)
            band_feature = crossing[0][3]
    if band_feature is not None and band_feature not in features:
        band_feature = None

    if ax is None:
        fig = Figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    span_labels = {}
    if band_feature is not None:
        fit = fits.get(band_feature)
        if fit is not None and getattr(fit, "ok", False):
            grid = np.geomspace(x_lo * 0.85, x_hi * 1.15, 400)
            curve = fit.predict(grid)
            spans = _bands(ax, grid, curve, tolerance)
            span_labels = {kind: (lo, hi) for kind, lo, hi in spans}
            ax.plot([], [], " ", label="bands follow %s's fit" % band_feature)

    for feature in features:
        x, d = _series(result, feature, x_key)
        if not x.size:
            continue
        ax.scatter(x, d, s=38, zorder=3, edgecolor="white", linewidth=0.6,
                   label="%s (measured)" % feature)

        fit = fits.get(feature)
        if fit is not None and getattr(fit, "ok", False):
            # Only draw the curve across the range that was actually audited,
            # widened by 15 %: beyond that it is extrapolation, and the figure
            # should not imply otherwise.
            lo, hi = float(x.min()), float(x.max())
            grid = np.geomspace(lo * 0.85, hi * 1.15, 240)
            ax.plot(grid, fit.predict(grid), "-", linewidth=1.9,
                    color=FIT_COLOUR, zorder=4,
                    label="%s: d = %.3g·r^(−%.2f) (n=%d)"
                          % (feature, fit.C, fit.p, fit.n))

    if annotate and tolerance and band_feature is not None:
        ax.axvline(x_hi, color="#9e9e9e", linestyle=":", linewidth=1.0, zorder=2)
        ax.annotate("audited up to %s = %g\nright of this line: extrapolation"
                    % ("r" if x_key == "r" else "fs", x_hi),
                    xy=(0.985, 0.985), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7, color="#616161",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#e0e0e0", alpha=0.85))

    if tolerance is not None and tolerance > 0:
        ax.axhline(tolerance, color=DEAD_COLOUR, linestyle="--", linewidth=1.1,
                   zorder=2, label="tolerance = %.3g" % tolerance)

    ax.set_xlabel("over-sampling ratio r = fs / B" if x_key == "r"
                  else "sampling rate fs (Hz)")
    ax.set_ylabel("paired deficit  |Δfeature| / |reference|")
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif log_log:
        # Anchor the ordinate to the measurements: fits may run far below the
        # smallest measured deficit without that being informative.
        ys = np.concatenate([_series(result, f, x_key)[1] for f in features
                             if _series(result, f, x_key)[1].size])
        ys = ys[np.isfinite(ys) & (ys > 0)]
        if ys.size:
            ax.set_ylim(float(ys.min()) * 0.4, float(ys.max()) * 4.0)
    ax.grid(True, which="both", alpha=0.18, zorder=1)
    ax.set_title(title or "Paired decay of feature deficit with the sampling grid")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.92)

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    return fig


__all__ = ["plot_decay", "SAFE_COLOUR", "MARGINAL_COLOUR", "DEAD_COLOUR"]
