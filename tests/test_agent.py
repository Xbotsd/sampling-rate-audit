"""Tests for the invariance agent (plan / profile / advise / task contract)."""
import json
import math

import numpy as np
import pytest

from leakage_audit import agent


def _record(fs=1000.0, seconds=8.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    x = np.zeros_like(t)
    for k in range(1, 16):
        x += rng.rayleigh(1.0) * np.cos(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
    return x


# ---------------------------------------------------------------- plan
def test_plan_registry_feature():
    p = agent.plan(feature="zcr_per_sample")
    assert p["signature"] == "PER_SAMPLE"
    assert p["class"] == "sample-indexed"
    assert p["ratio_helps"] is False
    assert p["confidence"] == "high"


def test_plan_registry_difference_feature():
    p = agent.plan(feature="hjorth_mobility_Hz")
    assert p["signature"] == "DIFF_COUNT"
    assert p["ratio_helps"] is True
    assert p["prior_exponent"] == 2.0


def test_plan_from_expression():
    p = agent.plan(expr="number of zero crossings divided by len(x)")
    assert p["signature"] == "PER_SAMPLE"
    assert p["confidence"] == "heuristic"


def test_plan_unknown_feature_raises():
    with pytest.raises(KeyError):
        agent.plan(feature="not_a_feature")


def test_plan_requires_an_argument():
    with pytest.raises(ValueError):
        agent.plan()


# ------------------------------------------------------------- profile
def test_device_profile_ratio_and_nyquist():
    d = agent.device_profile(500, 40, window_s=30)
    assert d["oversampling_ratio"] == pytest.approx(12.5)
    assert d["nyquist_ok"] is True
    assert d["samples_per_window"] == 15000
    assert d["notes"] == []


def test_device_profile_flags_impossible_band():
    d = agent.device_profile(100, 80)
    assert d["nyquist_ok"] is False
    assert any("Nyquist" in n or "band" in n for n in d["notes"])


def test_device_profile_flags_analog_filter_above_nyquist():
    d = agent.device_profile(500, 40, analog_lpf_hz=400)
    assert any("alias" in n for n in d["notes"])


# ------------------------------------------------------- required_ratio
def test_required_ratio_quadratic():
    # 1% at r=10 with p=2 needs r=10*sqrt(10) for 0.1%
    need = agent.required_ratio(0.01, 10.0, 2.0, 0.001)
    assert need == pytest.approx(10.0 * math.sqrt(10.0), rel=1e-9)


def test_required_ratio_infinite_for_scale_factor():
    assert agent.required_ratio(0.5, 10.0, float("inf"), 0.01) == math.inf
    assert agent.required_ratio(0.5, 10.0, float("nan"), 0.01) == math.inf


def test_required_ratio_already_within_tolerance():
    assert agent.required_ratio(0.001, 10.0, 2.0, 0.01) == pytest.approx(10.0)


# --------------------------------------------------------------- advise
def test_advise_scale_factor_is_not_called_stable():
    audit = {
        "band_hz": 30.0,
        "ratios": [3.125, 6.25],
        "exponents": {"zcr_per_sample": float("nan")},
        "rows": [
            {"feature": "zcr_per_sample", "r": 3.125, "rel_bias": 14.9},
            {"feature": "zcr_per_sample", "r": 6.25, "rel_bias": 7.0},
        ],
    }
    a = agent.advise(audit, tolerance=0.01)
    f = a["features"][0]
    assert f["no_ratio_helps"] is True
    assert f["verdict"] == "no ratio helps"
    assert "scale factor" in f["action"]


def test_advise_reports_tolerance_already_met():
    audit = {
        "band_hz": 30.0,
        "ratios": [3.125, 6.25],
        "exponents": {"std": float("nan")},
        "rows": [
            {"feature": "std", "r": 3.125, "rel_bias": 1e-16},
            {"feature": "std", "r": 6.25, "rel_bias": 1e-16},
        ],
    }
    f = agent.advise(audit, tolerance=0.01)["features"][0]
    assert f["verdict"] == "met within audited range"
    assert f["required_ratio_for_tolerance"] == pytest.approx(3.125)


def test_advise_warns_when_too_few_ratios():
    audit = {
        "band_hz": 30.0,
        "ratios": [3.125, 6.25],
        "exponents": {"hjorth_mobility_Hz": float("nan")},
        "rows": [
            {"feature": "hjorth_mobility_Hz", "r": 3.125, "rel_bias": 0.05},
            {"feature": "hjorth_mobility_Hz", "r": 6.25, "rel_bias": 0.011},
        ],
    }
    f = agent.advise(audit, tolerance=0.001)["features"][0]
    assert f["verdict"] == "unresolved"
    assert "no exponent can be fitted" in f["action"]


# ------------------------------------------------------------- run_task
def test_run_task_profile():
    out = agent.run_task({"action": "profile", "device": {"fs_hz": 256, "band_hz": 20}})
    assert out["device"]["oversampling_ratio"] == pytest.approx(12.8)


def test_run_task_audit_returns_rows(tmp_path):
    x = _record()
    p = tmp_path / "rec.npy"
    np.save(p, x)
    out = agent.run_task({
        "action": "audit",
        "device": {"fs_hz": 1000.0, "band_hz": 30.0},
        "data": {"path": str(p)},
        "features": ["zcr_per_second", "hjorth_mobility_Hz"],
    })
    assert out["audit"]["rows"]
    assert {r["feature"] for r in out["audit"]["rows"]} <= {"zcr_per_second", "hjorth_mobility_Hz"}


def test_run_task_full_is_json_serialisable(tmp_path):
    x = _record()
    p = tmp_path / "rec.npy"
    np.save(p, x)
    out = agent.run_task({
        "action": "full",
        "device": {"fs_hz": 1000.0, "band_hz": 30.0},
        "data": {"path": str(p)},
        "tolerance": 0.01,
        "features": ["zcr_per_second", "zcr_per_sample"],
    })
    text = agent.dumps(out)          # NaN/Inf must become null
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
    assert out["advice"]["features"]


def test_describe_contract():
    d = agent.describe()
    assert set(d["actions"]) == {"plan", "profile", "audit", "advise", "full"}
    assert "structural_classes" in d
    json.dumps(d, default=str)
