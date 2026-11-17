"""Finite-range power-law estimation for the paired audit (DEMO build).

中文说明
--------
本模块把"亏空随采样率衰减"这件事拟合成一条幂律曲线

    d(r) = C · r^(−p)                                   (E1)

其中 ``d`` 是配对亏空（paired deficit），``r`` 是过采样比 ``r = fs / B``，
``p`` 是经验阶数（empirical order），``C`` 是比例常数。

**这是有限区间的描述性拟合，不是渐近定理。** 论文与 V1.x 引擎反复强调的
那一条在这里同样成立：``p`` 只对被审计过的那段 ``r`` 区间、那套处理路径、
那一批记录负责；把它外推到区间之外没有依据。Demo 版提供这个接口是为了
教学与可视化，不是为了让谁引用一个"阶数"。

Three estimators are provided:

``nls``
    Ordinary non-linear least squares in the *linear* space of ``d``.  This is
    what most people mean by "fit a power law", and it weights large deficits
    more heavily than small ones.

``mle``
    Maximum likelihood under a *log-normal* error model, i.e. a Gaussian fit in
    ``log d`` space with the noise scale profiled out.  For a curve that spans
    orders of magnitude this is usually the better-behaved choice, because a
    relative error of 10 % counts the same at both ends of the curve.

``huber``
    The same log-space model, minimised with the Huber loss, so that one
    outlier rate cell cannot drag the exponent on its own.

All three treat deficits that are zero, negative or non-finite as uninformative
and drop them, recording how many were dropped in ``PowerLawFit.note``.

依赖：numpy 必需；scipy 仅在调用估计函数时导入，因此 ``import leakage_audit``
仍然只需要 numpy。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# A deficit at or below this floor carries no information about the exponent:
# it is either an exactly invariant feature or numerical dust.
DEFICIT_FLOOR = 1e-12

# Minimum number of usable (rate, deficit) pairs.  Two points determine a line,
# so two is the mathematical minimum, but a two-point exponent is a slope through
# two dots, not a fit; we allow it and say so in ``note``.
MIN_POINTS = 2
COMFORTABLE_POINTS = 3


@dataclass
class PowerLawFit:
    """Result of fitting ``d = C · r^(−p)`` to a set of paired deficits.

    ``ok`` is the only field callers should branch on before using ``p``; when
    it is False the numeric fields are NaN and ``note`` says why.
    """

    p: float = float("nan")
    C: float = float("nan")
    rss: float = float("nan")          # residual sum of squares, in the fit's own space
    sigma: float = float("nan")        # residual scale (per point)
    n: int = 0                         # points actually used
    n_dropped: int = 0                 # points discarded as uninformative
    method: str = "nls"
    space: str = "linear"              # "linear" (nls) or "log" (mle / huber)
    ok: bool = False
    note: str = ""
    r_min: float = float("nan")
    r_max: float = float("nan")

    # ------------------------------------------------------------------ use
    def predict(self, r):
        """Evaluate the fitted curve. Returns NaN when the fit did not succeed.

        拟合曲线的取值。未成功拟合时返回 NaN，而不是抛异常：调用方（绘图、
        守卫）应当能安全地把"没拟合出来"当作一种正常状态。
        """
        if not self.ok:
            return np.full(np.shape(r), np.nan, dtype=float)
        r = np.asarray(r, dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self.C * np.power(r, -self.p)

    def as_dict(self):
        """Plain dict, for JSON output."""
        return {
            "p": self.p, "C": self.C, "rss": self.rss, "sigma": self.sigma,
            "n": self.n, "n_dropped": self.n_dropped, "method": self.method,
            "space": self.space, "ok": self.ok, "note": self.note,
            "r_min": self.r_min, "r_max": self.r_max,
        }

    def summary(self):
        """One human-readable line, used by the CLI and the demo tutorial."""
        if not self.ok:
            return "%s: no fit (%s)" % (self.method, self.note or "insufficient data")
        return ("%s: p = %.3f, C = %.3g, RSS = %.3g over %d points "
                "(r = %g–%g)" % (self.method, self.p, self.C, self.rss,
                                 self.n, self.r_min, self.r_max))

    def __str__(self):  # pragma: no cover - convenience only
        return self.summary()


# --------------------------------------------------------------------------
# input hygiene
# --------------------------------------------------------------------------


def _sanitise(deficits, rates):
    """Drop uninformative points and return ``(r, d, n_dropped)``, sorted by r.

    只保留"亏空为正、采样率为正、两者都有限"的点。这一步是数值稳定性的
    第一道防线：幂律在对数空间里对 0 和负数没有定义，而真实的配对审计里
    恰好会出现亏空精确为 0 的情况（某个特征在该网格上完全不变）。
    """
    d = np.asarray(deficits, dtype=float).ravel()
    r = np.asarray(rates, dtype=float).ravel()
    if d.shape != r.shape:
        raise ValueError("deficits and rates must have the same length "
                         "(got %d and %d)" % (d.size, r.size))
    keep = (np.isfinite(d) & np.isfinite(r) & (d > DEFICIT_FLOOR) & (r > 0.0))
    n_dropped = int((~keep).sum())
    d, r = d[keep], r[keep]
    order = np.argsort(r)
    return r[order], d[order], n_dropped


def _start_values(r, d):
    """Closed-form starting guess for ``(C, p)`` from the log–log line.

    一条过两端点的直线给初值：比 p0 = 1 更稳，也不会在曲线陡的时候把
    ``curve_fit`` 送进一个坏局部极小。
    """
    lr, ld = np.log(r), np.log(d)
    if lr.size >= 2 and np.ptp(lr) > 0:
        slope, intercept = np.polyfit(lr, ld, 1)
        p0 = float(np.clip(-slope, 1e-3, 1e3))
        C0 = float(np.exp(intercept))
    else:  # pragma: no cover - guarded by MIN_POINTS
        p0, C0 = 1.0, float(np.median(d) * np.median(r))
    if not (np.isfinite(C0) and C0 > 0):
        C0 = float(np.median(d) * np.median(r) ** p0) or 1e-12
    return C0, p0


def _empty(method, note, n=0, n_dropped=0, space="linear"):
    return PowerLawFit(method=method, space=space, note=note,
                       n=n, n_dropped=n_dropped)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def estimate_order_nls(deficits, rates):
    """Non-linear least squares on ``d = C · r^(−p)`` in linear space.

    最小的、也最容易被误读的那个估计量：它最小化的是绝对残差平方和，所以
    曲线高处的点（低采样率、亏空大）权重远高于曲线低处。当亏空跨越两三个
    数量级时，低处几乎不影响结果。
    """
    r, d, n_drop = _sanitise(deficits, rates)
    if r.size < MIN_POINTS:
        return _empty("nls", "only %d usable point(s); need >= %d"
                      % (r.size, MIN_POINTS), r.size, n_drop)

    from scipy.optimize import curve_fit

    def model(x, C, p):
        return C * np.power(x, -p)

    C0, p0 = _start_values(r, d)
    fit_note = ""
    try:
        # With exactly two points and two parameters the covariance is
        # undefined; curve_fit warns about it and the fit is still the exact
        # two-point solution, so the warning is noise here.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (C, p), _cov = curve_fit(model, r, d, p0=(C0, p0),
                                     bounds=([0.0, 0.0], [np.inf, np.inf]),
                                     maxfev=20000)
        ok = True
    except Exception as exc:  # noqa: BLE001 - fall back, never propagate
        # 兜底：对数空间线性回归。仍然是幂律，只是换了一个误差度量。
        lr, ld = np.log(r), np.log(d)
        slope, intercept = np.polyfit(lr, ld, 1)
        C, p = float(np.exp(intercept)), float(-slope)
        ok = np.isfinite(C) and np.isfinite(p)
        fit_note = "curve_fit failed (%s); used log-linear fallback" % type(exc).__name__

    resid = d - model(r, C, p)
    rss = float(np.sum(resid ** 2))
    fit = PowerLawFit(
        p=float(p), C=float(C), rss=rss,
        sigma=float(np.sqrt(rss / max(r.size - 2, 1))),
        n=int(r.size), n_dropped=n_drop, method="nls", space="linear",
        ok=bool(ok), note=fit_note,
        r_min=float(r.min()), r_max=float(r.max()),
    )
    if fit.ok and not fit.note:
        if fit.n == MIN_POINTS:
            fit.note = "two points; exponent is a slope, not a verified fit"
        elif fit.n < COMFORTABLE_POINTS:
            fit.note = "fewer than %d points" % COMFORTABLE_POINTS
    return fit


def estimate_order_mle(deficits, rates):
    """Maximum likelihood under a log-normal error model.

    在 ``log d`` 空间里的高斯极大似然：最小化

        Σ (log dᵢ − log C + p · log rᵢ)²                        (E2)

    噪声尺度 σ 被 profile 掉（解析解），所以这里只需要对 (log C, p) 做二维
    极小化。对跨数量级的衰减曲线，这是三者里最稳的一个。
    """
    r, d, n_drop = _sanitise(deficits, rates)
    if r.size < MIN_POINTS:
        return _empty("mle", "only %d usable point(s); need >= %d"
                      % (r.size, MIN_POINTS), r.size, n_drop, space="log")

    from scipy.optimize import minimize

    lr, ld = np.log(r), np.log(d)
    C0, p0 = _start_values(r, d)
    n = r.size

    def negloglik(theta):
        logC, p = theta
        resid = ld - (logC - p * lr)
        # profile out sigma: sigma^2 = RSS / n, so the concentrated
        # log-likelihood is -(n/2) * log(RSS / n) plus constants.
        rss = float(np.dot(resid, resid))
        rss = max(rss, 1e-300)
        return 0.5 * n * math.log(rss / n)

    def grad(theta):
        """Analytic gradient of the concentrated log-likelihood.

        With ``rss = Σ rᵢ²`` and ``−2·logL = n·log(rss/n)``, the chain rule gives

            ∂/∂logC = −n·Σrᵢ / rss        ∂/∂p = +n·Σ(rᵢ·log rᵢ) / rss

        给出解析梯度后 L-BFGS-B 一步收敛；Nelder-Mead 在这个尺度上会因浮点
        噪声一路撞到迭代上限，且 ``success`` 无法反映真实收敛情况。
        """
        logC, p = theta
        resid = ld - (logC - p * lr)
        rss = max(float(np.dot(resid, resid)), 1e-300)
        return np.array([-n * resid.sum() / rss,
                         n * float(np.dot(resid, lr)) / rss])

    res = minimize(negloglik, x0=np.array([math.log(C0), p0]), method="L-BFGS-B",
                   jac=grad, options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
    logC, p = float(res.x[0]), float(res.x[1])

    # L-BFGS-B reports "ABNORMAL_TERMINATION_IN_LNSRCH" when the data fit the
    # model to machine precision and the line search cannot improve further.
    # That is convergence, not a failure.  A raw gradient test is unusable here:
    # the gradient scales as 1/rss, so at rss ~ 1e-30 it amplifies rounding dust.
    # 判据用残差本身的绝对尺度：残差已在浮点噪声水平即视为收敛。
    resid = ld - (logC - p * lr)
    converged = bool(res.success) or float(np.max(np.abs(resid))) < 1e-8
    status_note = "" if converged else "optimiser reported: %s" % res.message
    rss_log = float(np.dot(resid, resid))
    C = float(math.exp(logC))
    fit = PowerLawFit(
        p=p, C=C, rss=rss_log,
        sigma=float(np.sqrt(rss_log / max(r.size, 1))),
        n=int(r.size), n_dropped=n_drop, method="mle", space="log",
        ok=bool(converged and np.isfinite(p)),
        note=status_note,
        r_min=float(r.min()), r_max=float(r.max()),
    )
    if fit.ok and fit.n == MIN_POINTS and not fit.note:
        fit.note = "two points; exponent is a slope, not a verified fit"
    return fit


def estimate_order_huber(deficits, rates, delta=1.345):
    """Huber-robust fit of the same log-space model.

    ``delta=1.345`` is the usual 95 %-efficiency choice for the Huber loss under
    Gaussian noise. 残差先用 MAD 标准化，所以 ``delta`` 与数据的量纲无关：
    一个离谱的采样率单元最多贡献一条线性而非二次的残差，拉不动指数。
    """
    r, d, n_drop = _sanitise(deficits, rates)
    if r.size < MIN_POINTS:
        return _empty("huber", "only %d usable point(s); need >= %d"
                      % (r.size, MIN_POINTS), r.size, n_drop, space="log")

    from scipy.optimize import minimize

    lr, ld = np.log(r), np.log(d)
    C0, p0 = _start_values(r, d)

    # The scale is fixed once, from the residuals at the starting point.  If it
    # were recomputed inside the objective the loss surface would shift under
    # the optimiser's feet, which makes a derivative-free method stall at a
    # point that is not a minimum. 固定尺度也是 IRLS 的标准做法。
    resid0 = ld - (math.log(C0) - p0 * lr)
    scale = 1.4826 * float(np.median(np.abs(resid0 - np.median(resid0))))
    if not (np.isfinite(scale) and scale > 1e-12):
        scale = 1.0

    def loss(theta):
        logC, p = theta
        z = np.abs(ld - (logC - p * lr)) / scale
        quad = np.minimum(z, delta)
        lin = z - quad
        return float(np.sum(0.5 * quad ** 2 + delta * lin))

    res = minimize(loss, x0=np.array([math.log(C0), p0]), method="Nelder-Mead",
                   options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
    if not res.success:
        # Powell has a better stopping rule on this piecewise-smooth loss.
        res2 = minimize(loss, x0=res.x, method="Powell",
                        options=dict(maxiter=4000, xtol=1e-8, ftol=1e-10))
        if res2.success or res2.fun <= res.fun:
            res = res2
    logC, p = float(res.x[0]), float(res.x[1])
    resid = ld - (logC - p * lr)
    fit = PowerLawFit(
        p=p, C=float(math.exp(logC)), rss=float(np.dot(resid, resid)),
        sigma=1.4826 * float(np.median(np.abs(resid - np.median(resid)))),
        n=int(r.size), n_dropped=n_drop, method="huber", space="log",
        ok=bool(res.success or np.isfinite(p)),
        note="" if res.success else "optimiser reported: %s" % res.message,
        r_min=float(r.min()), r_max=float(r.max()),
    )
    if fit.ok and fit.n == MIN_POINTS and not fit.note:
        fit.note = "two points; exponent is a slope, not a verified fit"
    return fit


_ESTIMATORS = {
    "nls": estimate_order_nls,
    "mle": estimate_order_mle,
    "huber": estimate_order_huber,
}


def available_methods():
    """Names accepted by :func:`estimate_order`."""
    return tuple(_ESTIMATORS)


def estimate_order(deficits, rates, method="nls", **kwargs):
    """Dispatch to one of the three estimators.

    Parameters
    ----------
    deficits : sequence of float
        Paired deficits, one per audited rate cell.
    rates : sequence of float
        The abscissa, normally the over-sampling ratio ``r = fs / B``.  Any
        positive, monotonically meaningful scale works (``fs`` itself, for
        instance) but ``C`` then carries that scale's units.
    method : {"nls", "mle", "huber"}
    **kwargs
        Passed through (``delta`` for ``huber``).

    Returns
    -------
    PowerLawFit

    Notes
    -----
    失败从不抛异常：点数不足、全为 0、全为常数，都返回 ``ok=False`` 的
    :class:`PowerLawFit`，并在 ``note`` 里写清原因。Demo 版的调用方（可视化、
    守卫、示例脚本）因此可以无条件继续。
    """
    key = str(method).lower().strip()
    if key not in _ESTIMATORS:
        raise ValueError("unknown estimator %r; choose from %s"
                         % (method, ", ".join(available_methods())))
    return _ESTIMATORS[key](deficits, rates, **kwargs)


__all__ = [
    "PowerLawFit", "estimate_order", "estimate_order_nls", "estimate_order_mle",
    "estimate_order_huber", "available_methods", "DEFICIT_FLOOR",
]
