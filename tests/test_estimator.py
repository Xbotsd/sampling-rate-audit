"""Tests for the demo-build order estimator, guard and plotting hooks.

中文说明
--------
这一组测试守的是三件事：

1. **已知真值能否被还原**：合成一条 ``p = 0.8`` 的幂律，三种估计量的回收误差
   都要落在 ±0.15 之内（题目给定的验收线）。
2. **退化输入不许抛异常**：亏空为 0、只有一两个点、采样率重复——必须返回
   ``ok=False`` 并写清楚原因，而不是让审计流程崩掉。
3. **Demo 版绝不越权**：``DemoGuard`` 只打印、只返回状态码，不能抛致命异常、
   不能改写输入、不能退出进程。

Run:  pytest -q tests/test_estimator.py
"""

from __future__ import annotations

import numpy as np
import pytest

from leakage_audit import Audit, DEMO_BUILD
from leakage_audit.demoguard import DemoGuard, SAFE, WARNING, check
from leakage_audit.estimator import (
    PowerLawFit,
    available_methods,
    estimate_order,
    estimate_order_huber,
    estimate_order_mle,
    estimate_order_nls,
)

TRUE_P = 0.8
TRUE_C = 0.5
RATES = np.array([2.5, 3.125, 5.0, 6.25, 12.5, 25.0, 50.0])


def synthetic_deficits(p=TRUE_P, C=TRUE_C, rates=RATES, noise=0.0, seed=0):
    """Exact (or slightly noisy) power law, ``d = C·r^(−p)``."""
    d = C * rates.astype(float) ** (-p)
    if noise:
        rng = np.random.default_rng(seed)
        d = d * (1.0 + noise * rng.standard_normal(d.size))
    return d


# ---------------------------------------------------------------- recovery
@pytest.mark.parametrize("method", ["nls", "mle", "huber"])
def test_recovers_known_order(method):
    """^p must land within ±0.15 of the true 0.8 on clean data."""
    fit = estimate_order(synthetic_deficits(), RATES, method=method)
    assert fit.ok, fit.note
    assert abs(fit.p - TRUE_P) <= 0.15, "%s gave p=%.3f" % (method, fit.p)


@pytest.mark.parametrize("method", ["nls", "mle", "huber"])
def test_recovers_known_C(method):
    fit = estimate_order(synthetic_deficits(), RATES, method=method)
    assert fit.ok
    assert abs(fit.C - TRUE_C) / TRUE_C < 0.05, "%s gave C=%.4f" % (method, fit.C)


@pytest.mark.parametrize("method", ["nls", "mle", "huber"])
def test_recovers_order_with_small_noise(method):
    fit = estimate_order(synthetic_deficits(noise=0.02, seed=3), RATES, method=method)
    assert fit.ok
    assert abs(fit.p - TRUE_P) <= 0.15, "%s gave p=%.3f" % (method, fit.p)


def test_huber_resists_a_single_outlier():
    """One grossly wrong rate cell must not drag the robust exponent away."""
    d = synthetic_deficits(noise=0.05, seed=0)
    d[3] *= 4.0
    huber = estimate_order(d, RATES, method="huber")
    nls = estimate_order(d, RATES, method="nls")
    assert huber.ok
    assert abs(huber.p - TRUE_P) <= 0.15, "huber p=%.3f" % huber.p
    # The point of the comparison: plain least squares moves further.
    assert abs(nls.p - TRUE_P) > abs(huber.p - TRUE_P)


def test_rss_is_reported_in_the_fit_space():
    fit = estimate_order(synthetic_deficits(), RATES, method="mle")
    assert np.isfinite(fit.rss) and fit.rss >= 0.0
    assert fit.space == "log"
    assert estimate_order(synthetic_deficits(), RATES, method="nls").space == "linear"


def test_predict_matches_the_model():
    fit = estimate_order(synthetic_deficits(), RATES, method="nls")
    grid = np.array([1.0, 10.0, 100.0])
    expect = TRUE_C * grid ** (-TRUE_P)
    assert np.allclose(fit.predict(grid), expect, rtol=1e-3)


# ---------------------------------------------------------------- robustness
def test_all_zero_deficits_do_not_raise():
    fit = estimate_order(np.zeros(5), RATES[:5], method="nls")
    assert fit.ok is False
    assert "usable" in fit.note


def test_single_point_do_not_raise():
    fit = estimate_order([0.1], [2.5], method="mle")
    assert fit.ok is False
    assert "1 usable" in fit.note


def test_two_points_fit_but_are_labelled():
    fit = estimate_order([0.3, 0.1], [2.5, 12.5], method="nls")
    assert fit.ok
    assert "two points" in fit.note


def test_negative_and_nan_points_are_dropped():
    d = synthetic_deficits()
    d[1] = -1.0
    d[4] = np.nan
    fit = estimate_order(d, RATES, method="mle")
    assert fit.ok
    assert fit.n_dropped == 2
    assert abs(fit.p - TRUE_P) <= 0.15


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        estimate_order([0.1, 0.2], [2.5, 5.0, 10.0])


def test_unknown_method_raises_with_the_menu():
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate_order(synthetic_deficits(), RATES, method="magic")
    assert set(available_methods()) == {"nls", "mle", "huber"}


def test_individual_entry_points_are_public():
    for fn in (estimate_order_nls, estimate_order_mle, estimate_order_huber):
        fit = fn(synthetic_deficits(), RATES)
        assert isinstance(fit, PowerLawFit)
        assert fit.ok


# ------------------------------------------------------------------ guard
def test_guard_returns_safe_below_tolerance():
    g = DemoGuard(tolerance=0.05, verbose=False)
    assert g.check_risk(0.01) == SAFE


def test_guard_returns_warning_above_tolerance():
    g = DemoGuard(tolerance=0.05, verbose=False)
    assert g.check_risk(0.5) == WARNING


def test_guard_never_raises_and_only_prints(capsys):
    g = DemoGuard(tolerance=0.01)
    code = g.check_risk(10.0)          # extreme, must still be a return value
    assert code == WARNING
    out = capsys.readouterr().out
    assert "Demo warning" in out
    # The warning must say plainly that nothing was blocked.
    assert "Nothing" in out or "nothing" in out


def test_guard_survives_a_non_numeric_input():
    g = DemoGuard(tolerance=0.01, verbose=False)
    assert g.check_risk("not a number") == WARNING
    assert g.check_risk(float("nan")) == WARNING


def test_guard_rejects_a_nonsense_tolerance():
    with pytest.raises(ValueError):
        DemoGuard(tolerance=0.0)


def test_guard_does_not_mutate_its_input():
    """The guard must be a read-only observer of the caller's data."""
    data = np.array([0.1, 0.2, 0.3])
    before = data.copy()
    DemoGuard(tolerance=0.01, verbose=False).check_risk(1.0)
    DemoGuard(tolerance=0.01, verbose=False).check_fit(
        estimate_order(synthetic_deficits(), RATES), r_grid=data)
    assert np.array_equal(data, before)


def test_guard_summary_counts_warnings():
    g = DemoGuard(tolerance=0.05, verbose=False)
    g.check_risk(0.01)
    g.check_risk(0.5)
    g.check_risk(0.001)
    assert "1 warning" in g.summary()


def test_module_level_check_helper():
    assert check(0.001, tolerance=0.01, verbose=False) == SAFE
    assert check(0.5, tolerance=0.01, verbose=False) == WARNING


# -------------------------------------------------------------------- demo
def test_demo_build_flag_is_set():
    """The branch identity must be visible from code, not only from the README."""
    assert DEMO_BUILD is True


def test_audit_attaches_fits_and_keeps_v1_behaviour():
    fs, band = 500.0, 20.0
    rng = np.random.default_rng(1)
    t = np.arange(int(4 * fs)) / fs
    x = sum(rng.rayleigh(1.0) * np.cos(2 * np.pi * rng.uniform(1, band) * t
                                  + rng.uniform(0, 2 * np.pi)) for _ in range(12))

    demo = Audit(estimator_method="mle").run(x, fs=fs, band=band,
                                             rates=[250, 125, 62.5])
    assert demo.fits, "no fits attached"
    assert any(f is not None and f.ok for f in demo.fits.values())
    assert demo.markdown().startswith("| feature | order |")

    v1 = Audit(fit_order=False).run(x, fs=fs, band=band, rates=[250, 125, 62.5])
    assert v1.fits == {}
    assert v1.rows == demo.rows, "fitting must not change the audit itself"
    assert "no order fit" in v1.fit_markdown().replace("requested", "requested")


def test_audit_rejects_an_unknown_estimator_when_fitting():
    with pytest.raises(ValueError, match="unknown estimator_method"):
        Audit(estimator_method="nope").run(np.random.default_rng(0).normal(size=800),
                                           fs=500, band=20, rates=[250, 125])


def test_audit_degrades_gracefully_without_scipy(monkeypatch):
    """The core package depends only on numpy; a missing scipy must not raise.

    核心包只声明 numpy。若环境里没有 scipy，拟合应当安静地跳过并留下说明，
    而不是让审计本身失败。
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    x = np.random.default_rng(0).normal(size=2000)
    result = Audit(estimator_method="nls").run(x, fs=500, band=20,
                                               rates=[250.0, 125.0, 62.5])
    assert result.rows, "the audit itself must still run"
    assert "scipy" in result.fit_note
    assert "scipy" in result.fit_markdown()


# ------------------------------------------------------------- visualizer
def _toy_result(method="mle"):
    fs, band = 500.0, 20.0
    rng = np.random.default_rng(5)
    t = np.arange(int(6 * fs)) / fs
    x = sum(rng.rayleigh(1.0) * np.cos(2 * np.pi * rng.uniform(1, band) * t
                                  + rng.uniform(0, 2 * np.pi)) for _ in range(10))
    return Audit(estimator_method=method).run(
        x, fs=fs, band=band, rates=[250.0, 125.0, 62.5, 50.0])


def test_plot_decay_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    from leakage_audit.visualizer import plot_decay

    out = tmp_path / "decay.png"
    fig = plot_decay(_toy_result(), tolerance=0.05, out_path=out)
    assert out.exists() and out.stat().st_size > 1000
    assert fig is not None


def test_plot_decay_rejects_an_unknown_feature():
    pytest.importorskip("matplotlib")
    from leakage_audit.visualizer import plot_decay

    with pytest.raises(KeyError, match="not in this result"):
        plot_decay(_toy_result(), features=["no_such_feature"])


def test_plot_decay_follows_the_requested_band_feature(tmp_path):
    pytest.importorskip("matplotlib")
    from leakage_audit.visualizer import plot_decay

    result = _toy_result()
    name = sorted(result.fits)[0]
    out = tmp_path / "bands.png"
    plot_decay(result, features=[name], tolerance=0.05, band_feature=name,
               out_path=out)
    assert out.exists()


def test_plot_decay_without_fits_still_plots(tmp_path):
    """A result audited with fit_order=False must still be plottable."""
    pytest.importorskip("matplotlib")
    from leakage_audit.visualizer import plot_decay

    fs, band = 500.0, 20.0
    x = np.random.default_rng(2).normal(size=int(4 * fs))
    result = Audit(fit_order=False).run(x, fs=fs, band=band,
                                        rates=[250.0, 125.0, 62.5])
    out = tmp_path / "nofit.png"
    plot_decay(result, features=["std"], out_path=out)
    assert out.exists()
