"""Demo-only safety guard: warn loudly, never block, never touch your data.

中文说明
--------
正式的质量体系里，"把采样率降到公差之外"这样的操作应当在流程层面被拦住。
本模块**不拦**。它只做三件事：

1. 把预测亏空与用户给定的公差比较；
2. 越界时在终端打印一条醒目的警告；
3. 返回一个状态码，让调用方自己决定要不要继续。

它刻意不做的事（这是设计，不是遗漏）：

* 不抛异常——审计跑到一半被异常打断，比看到一条警告更糟；
* 不退出进程；
* 不写、不改、不删任何文件；
* 不回滚任何数据。

也就是说：**这个类没有任何强制力**。它是一块提示牌，不是一道闸门。
如果你的场景真的需要"越界即拒绝"，那要在你自己的流程里实现，不能把
责任交给这里的一个 print。

关于版本的一句话：本分支（V2.1.1-demo）是一个沙盒。仓库里没有、这个类里
也没有任何"生产版"的拦截、回档或隔离机制；不要把下面那条警告文字理解成
"Our production build does this"。它只是提醒你：**这条曲线是用外推算出来
的，真实部署时应当重新测量，而不是相信一个指数。**
"""

from __future__ import annotations

import sys

import numpy as np

YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

SAFE = "SAFE"
WARNING = "WARNING"

# The band between tolerance and this multiple is "marginal": a prediction
# within it is a judgement call rather than a clear violation.
MARGINAL_FACTOR = 3.0


class DemoGuard:
    """Tolerance checker for the demo build.

    Parameters
    ----------
    tolerance : float
        The user's admissible paired deficit (relative, dimensionless).  A
        typical teaching value is 0.01 (1 %).
    marginal_factor : float
        Above ``tolerance`` but at most ``marginal_factor × tolerance`` the
        message is "marginal"; above that it is "well outside".
    verbose : bool
        Print the warning to ``stream``.  Set False in tests.
    stream : file-like
        Defaults to ``sys.stdout``; captured at call time so pytest's capsys
        still sees the text.
    """

    def __init__(self, tolerance=0.01, marginal_factor=MARGINAL_FACTOR,
                 verbose=True, stream=None):
        tol = float(tolerance)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tolerance must be a positive finite number, got %r"
                             % (tolerance,))
        self.tolerance = tol
        self.marginal_factor = float(marginal_factor)
        self.verbose = bool(verbose)
        self._stream = stream
        self.history = []          # list of dicts, for a session summary

    # ------------------------------------------------------------------
    def check_extrapolation(self, predicted_deficit):
        """Alias of :meth:`check_risk`, kept for the wording in the demo notes.

        同一个检查的两个名字：``check_risk`` 描述"这个预测值有多危险"，
        ``check_extrapolation`` 描述调用场景"你在用外推的曲线做决定"。
        两者行为完全一致，都只打印、都返回 ``SAFE`` / ``WARNING``。
        """
        return self.check_risk(predicted_deficit)

    # ------------------------------------------------------------------
    def check_risk(self, predicted_deficit):
        """Compare one predicted deficit with the tolerance.

        Returns
        -------
        str
            ``"SAFE"`` or ``"WARNING"``.  Never raises for a bad input: a
            non-finite prediction is reported as a WARNING, because "cannot
            verify" is not the same statement as "verified safe".
        """
        try:
            value = float(predicted_deficit)
        except (TypeError, ValueError):
            value = float("nan")

        if not np.isfinite(value):
            code = WARNING
            ratio_text = "the predicted deficit is not a finite number"
            severe = True
        elif value < 0.0:
            # A deficit is |Δfeature| / |reference|, so a negative value is a
            # programming error upstream, not a comfortable margin. Refusing to
            # call it SAFE is the whole point of this branch: "cannot verify"
            # must never read as "verified safe".
            code = WARNING
            ratio_text = ("the predicted deficit is negative (%.4g), which cannot "
                          "be a relative magnitude" % value)
            severe = True
        else:
            ratio = value / self.tolerance
            code = SAFE if value <= self.tolerance else WARNING
            severe = ratio > self.marginal_factor
            ratio_text = ("predicted deficit %.3g is %.1f× the tolerance %.3g"
                          % (value, ratio, self.tolerance))

        self.history.append(dict(predicted=value, tolerance=self.tolerance,
                                 status=code, severe=bool(severe)))

        if code == WARNING and self.verbose:
            self._emit(value, severe, ratio_text)
        return code

    # ------------------------------------------------------------------
    def check_fit(self, fit, r_grid=None):
        """Check a fitted curve at the worst point of ``r_grid``.

        The deficit decays with ``r``, so the worst point is the smallest rate
        in the grid: that is where a production deployment would be asking the
        most of the instrument.
        """
        curve = fit.predict(r_grid) if r_grid is not None else None
        if curve is None:
            return self.check_risk(float("nan"))
        curve = np.asarray(curve, dtype=float)
        if curve.size == 0 or not np.any(np.isfinite(curve)):
            return self.check_risk(float("nan"))
        return self.check_risk(float(np.nanmax(curve)))

    # ------------------------------------------------------------------
    @staticmethod
    def _colour(text, colour=YELLOW):
        """ANSI colour, but only when writing to a terminal."""
        return "%s%s%s" % (colour, text, RESET)

    def _emit(self, value, severe, ratio_text):
        stream = self._stream if self._stream is not None else sys.stdout
        bar = "=" * 66
        lines = [
            "",
            self._colour(bar),
            self._colour(BOLD + "  [Demo warning] prediction is outside the tolerance"
                         + RESET),
            self._colour(bar),
            "  %s" % ratio_text,
            "  Estimated at the lowest rate you asked about, i.e. this is an",
            "  extrapolation from the cells you actually audited.",
            "",
            "  What a production quality system would do here: refuse the",
            "  operating point until it has been measured, not extrapolated.",
            "  What this demo does: print this block and keep going. Nothing",
            "  has been blocked, nothing has been rolled back, and your data",
            "  have not been modified.",
            "",
            "  A power-law exponent is a description of a finite range of",
            "  rates. Re-audit at the rate you intend to use before you rely",
            "  on the number.",
            self._colour(bar),
            "",
        ]
        if severe:
            lines.insert(4, self._colour(
                "  (this is well outside: more than %.0f× the tolerance)"
                % self.marginal_factor))
        print("\n".join(lines), file=stream)

    # ------------------------------------------------------------------
    def summary(self):
        """One line for the end of a run."""
        if not self.history:
            return "DemoGuard: no check performed."
        warned = sum(1 for h in self.history if h["status"] == WARNING)
        return ("DemoGuard: %d check(s), %d warning(s), tolerance = %.3g. "
                "No action was blocked."
                % (len(self.history), warned, self.tolerance))


def check(predicted_deficit, tolerance=0.01, verbose=True):
    """One-shot convenience wrapper around :class:`DemoGuard`."""
    return DemoGuard(tolerance=tolerance, verbose=verbose).check_risk(predicted_deficit)


__all__ = ["DemoGuard", "check", "SAFE", "WARNING", "MARGINAL_FACTOR"]
