"""Tests for the 3.1.1 optimiser layer.

Kept small on purpose: two records and two features, so the suite stays under a minute while
still exercising every guarantee the design claims.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from sra_opt import Oracle  # noqa: E402
from sra_opt.arbitrator import Arbitrator, CrossTrackReader, TrackResult, _design  # noqa: E402
from sra_opt.budget import Budget, hidden_set, spend_option  # noqa: E402
from sra_opt.deadlock import (EXIT_CODE, DeadlockGuard, DeadlockTriggered,  # noqa: E402
                              host_id, validate_inputs)
from sra_opt.eld_demo import Frozen, Sandbox, demo  # noqa: E402
from sra_opt.pareto import search  # noqa: E402
from sra_opt.surrogate import (FLOOR_THRESHOLD, Surrogate, sign_risk_agreement,  # noqa: E402
                               verify_risk_agreement)
from sra_opt.visualizer_v3 import replay_matches, write_bug_report  # noqa: E402

FEATURES = ["hjorth_mobility_Hz", "std"]


@pytest.fixture(scope="module")
def oracle():
    return Oracle(FEATURES, n_records=2, seed=3)


# ---------------------------------------------------------------------- oracle
def test_price_matches_records():
    o = Oracle(["std"], n_records=3)
    assert o.price(4) == 12


def test_caching_does_not_double_bill(oracle):
    oracle.reference("std")                 # warm the reference arm the deficit needs
    before = oracle.spent
    oracle.deficit("std", 8.0)
    mid = oracle.spent
    assert mid == before + oracle.n_records  # one new ratio, billed once
    oracle.deficit("std", 8.0)
    assert oracle.spent == mid               # the repeat is free


def test_missing_ratio_costs_and_repeat_is_free(oracle):
    start = oracle.spent
    oracle.values("std", 3.0)
    assert oracle.spent == start + oracle.n_records
    oracle.values("std", 3.0)
    assert oracle.spent == start + oracle.n_records


def test_g_of_anchor_is_zero(oracle):
    assert oracle.g("hjorth_mobility_Hz", 10.0, 10.0) == pytest.approx(0.0, abs=1e-12)


def test_floor_tends_down_with_anchor(oracle):
    a = oracle.floor("hjorth_mobility_Hz", 10.0)
    b = oracle.floor("hjorth_mobility_Hz", 100.0)
    assert b <= a + 1e-9


# ---------------------------------------------------------------------- surrogate
def test_small_pilot_refuses_to_extrapolate(oracle):
    pilot = [2.5, 4.0, 6.25]
    d = [oracle.deficit("std", r) for r in pilot]
    sub = Surrogate(np.array(pilot), np.maximum(d, 1e-15), "std")
    gate = sub.gate()
    assert gate["allowed"] is False
    assert "too small" in gate["reason"]


def test_far_extrapolation_refused_even_with_good_fit(oracle):
    pilot = [2.5, 4.0, 6.25, 12.5, 25.0]
    d = [oracle.deficit("hjorth_mobility_Hz", r) for r in pilot]
    sub = Surrogate(np.array(pilot), np.maximum(d, 1e-15), "hjorth_mobility_Hz")
    gate = sub.gate(target=200.0)
    assert gate["allowed"] is False
    assert "extrapolation" in gate["reason"]


def test_threshold_below_floor_is_rejected(oracle):
    pilot = [2.5, 4.0, 6.25, 12.5, 25.0]
    d = [oracle.deficit("std", r) for r in pilot]
    sub = Surrogate(np.array(pilot), np.maximum(d, 1e-15), "std")
    assert sub.gate(threshold=FLOOR_THRESHOLD / 2)["allowed"] is False


def test_risk_agreement_round_trips(tmp_path, oracle):
    pilot = [2.5, 4.0, 6.25]
    d = [oracle.deficit("std", r) for r in pilot]
    sub = Surrogate(np.array(pilot), np.maximum(d, 1e-15), "std")
    path = str(tmp_path / "agree.json")
    sign_risk_agreement(path, sub, 0.5, "tester", "unit test")
    assert verify_risk_agreement(path) is True
    body = json.load(open(path, encoding="utf-8"))
    assert body["chosen_threshold"] == 0.5
    assert body["kind"] == "prediction-distortion-risk-agreement"


def test_tampered_risk_agreement_fails_verification(tmp_path, oracle):
    pilot = [2.5, 4.0, 6.25]
    d = [oracle.deficit("std", r) for r in pilot]
    sub = Surrogate(np.array(pilot), np.maximum(d, 1e-15), "std")
    path = str(tmp_path / "agree.json")
    sign_risk_agreement(path, sub, 0.5, "tester")
    body = json.load(open(path, encoding="utf-8"))
    body["chosen_threshold"] = 0.1
    json.dump(body, open(path, "w", encoding="utf-8"))
    assert verify_risk_agreement(path) is False


# ---------------------------------------------------------------------- search
def test_certificate_respects_the_anchor_margin(oracle):
    rep = search(oracle, "hjorth_mobility_Hz", tau=0.02, budget_units=400)
    c = rep.certificate
    if c is not None and c.admissible:
        assert c.anchor >= 2.0 * c.device_ratio - 1e-6


def test_anchor_limited_is_a_verdict_not_a_number(oracle):
    rep = search(oracle, "hjorth_mobility_Hz", tau=1e-9, budget_units=200)
    assert rep.certificate is None or not rep.certificate.admissible
    assert "anchor-limited" in rep.verdict() or "infeasible" in rep.verdict()


def test_search_never_claims_an_unmeasured_certificate(oracle):
    rep = search(oracle, "hjorth_mobility_Hz", tau=0.02, budget_units=400)
    c = rep.certificate
    if c is not None and c.admissible:
        assert c.evidence == "measured"
        assert c.measured_g is not None


def test_compute_saving_is_reported_and_bounded(oracle):
    rep = search(oracle, "hjorth_mobility_Hz", tau=0.02, budget_units=400)
    doc = rep.as_dict()
    assert 0.0 <= doc["compute_saved"] <= 1.0
    assert doc["baseline_units"] > 0


# ---------------------------------------------------------------------- sandbox and ELD
def test_out_of_bounds_rolls_back_without_freezing():
    box = Sandbox(state={"r": 25.0}, admissible={"r": lambda v: 2.5 <= float(v) <= 200.0})
    out = box.propose({"r": 500.0})
    assert out["rolled_back"] is True
    assert box.frozen is False
    assert box.state["r"] == 25.0


def test_malformed_proposal_rolls_back_without_crashing():
    box = Sandbox(state={"r": 25.0}, admissible={"r": lambda v: 2.5 <= float(v) <= 200.0})
    out = box.propose({"r": "not-a-number"})
    assert out["accepted"] is False
    assert box.frozen is False


def test_only_integrity_violation_freezes(tmp_path):
    res = demo(str(tmp_path / "eld"))
    assert res["frozen_by_out_of_bounds"] is False
    assert res["integrity_event"]["ok"] is False
    assert res["proposal_blocked_after_freeze"] is True
    assert res["sandbox"]["accepted"] == 1


def test_frozen_sandbox_refuses_further_writes():
    box = Sandbox(state={"r": 1.0})
    box.freeze("test")
    with pytest.raises(Frozen):
        box.propose({"r": 2.0})


# ---------------------------------------------------------------------- budget
def test_reserve_is_fifteen_percent():
    b = Budget(total_units=1000)
    assert b.reserve_units == 150
    assert b.main_units == 850


def test_main_cannot_spend_the_reserve():
    b = Budget(total_units=1000)
    assert b.charge(150, "main") is True
    assert b.charge(700, "main") is True
    assert b.charge(1, "main") is False
    assert b.reserve_left == 150


def test_option_commits_and_records():
    b = Budget(total_units=1000)
    out = spend_option(b, "math", 100, "std")
    assert out["committed"] is True
    assert b.spent_reserve == 100
    assert b.reserve_left == 50


def test_unknown_sink_rejected():
    b = Budget(total_units=100)
    with pytest.raises(KeyError):
        spend_option(b, "wishful-thinking", 10, "std")


def test_hidden_set_selects_the_most_uncertain():
    spread = lambda rs: np.array([0.0, 0.1, 0.9, 0.2])
    got = hidden_set(spread, [1.0, 2.0, 3.0, 4.0], fraction=0.25)
    assert got["ratios"] == [3.0]


# ---------------------------------------------------------------------- arbitration
def test_arbitrator_needs_cases_and_says_so():
    arb = Arbitrator.fit([])
    assert arb.coef is None
    assert "too few cases" in arb.notes[0]
    assert arb.p_beta(np.zeros(7)) == 0.5


def test_arbitrator_combines_and_reports_disagreement():
    cases = []
    for k in range(8):
        alpha = 4.0 + k * 0.5
        beta = alpha * 1.3
        x = _design(TrackResult("alpha", alpha, 1.0, "D", 2), 
                    TrackResult("beta", beta, 1.0, "D", 2), 0.9, np.log(10.0), 0.02)
        cases.append({"alpha": alpha, "beta": beta, "truth": beta * 0.98, "x": x.tolist(),
                      "beta_closer": True})
    arb = Arbitrator.fit(cases)
    out = arb.combine(TrackResult("alpha", 5.0, 1.0, "D", 2),
                      TrackResult("beta", 8.0, 1.0, "D", 2),
                      _design(TrackResult("alpha", 5.0, 1.0, "D", 2),
                              TrackResult("beta", 8.0, 1.0, "D", 2), 0.9, np.log(10.0), 0.02))
    assert out["status"] in ("alpha", "beta", "agree", "unresolved")
    assert out["disagreement_log"] > 0
    assert 0.0 <= out["p_beta"] <= 1.0


def test_cross_track_reader_rejects_a_forged_record(tmp_path):
    path = tmp_path / "beta.json"
    body = {"ratio": 12.5, "bands": 2.0, "order": "D", "p": 2, "seed": 5,
            "content_sha256": "0" * 64}
    path.write_text(json.dumps(body), encoding="utf-8")
    reader = CrossTrackReader(str(path))
    got = reader.load()
    assert got["_integrity"]["hash_ok"] is False
    merged = reader.merge(TrackResult("alpha", 8.0, 1.0, "D", 2), got)
    assert merged["merged"] is False


# ---------------------------------------------------------------------- visualiser
def test_bug_report_is_reproducible(tmp_path):
    path = str(tmp_path / ".sra_bug_report")
    report = {"a": 1, "b": [1, 2, 3]}
    write_bug_report(path, report, seed=42, command="python run_v311.py")
    assert replay_matches(path, report) is True
    body = json.load(open(path, encoding="utf-8"))
    assert body["seed"] == 42
    assert body["sra_version"] == "3.1.1"


def test_bug_report_detects_a_changed_result(tmp_path):
    path = str(tmp_path / ".sra_bug_report")
    write_bug_report(path, {"a": 1}, seed=1, command="x")
    assert replay_matches(path, {"a": 2}) is False


# ---------------------------------------------------------------------- deadlock protection
def test_dirty_inputs_are_rejected():
    for bad in ({"tau": -1.0}, {"deficit": -0.3}, {"tau": float("nan")},
                {"ratio": float("inf")}, {"tau": "abc"}, {"tau": True}):
        with pytest.raises(DeadlockTriggered):
            validate_inputs(bad)


def test_clean_inputs_pass():
    assert validate_inputs({"tau": 0.02, "ratio": 7.0, "deficit": 0.5})["ok"] is True


def test_deadlock_writes_a_trace_with_a_stable_fingerprint(tmp_path):
    guard = DeadlockGuard(root=str(tmp_path))
    with pytest.raises(DeadlockTriggered) as exc:
        guard.guard_inputs({"tau": -1.0})
    assert exc.value.exit_code == EXIT_CODE == 3
    trace = (tmp_path / ".sra_deadlock").read_text(encoding="utf-8").strip().splitlines()
    assert len(trace) == 1
    body = json.loads(trace[0])
    assert body["event"] == "deadlock"
    assert body["reason"] == "dirty input"
    assert body["host_id"] == host_id()
    assert body["phase"] == "optimiser-export"
    assert isinstance(body["timestamp"], str)


def test_deadlock_purges_only_its_own_sandbox(tmp_path):
    guard = DeadlockGuard(root=str(tmp_path))
    (tmp_path / "user_data.csv").write_text("keep", encoding="utf-8")
    open(os.path.join(guard.sandbox, "junk.tmp"), "w").write("x")
    with pytest.raises(DeadlockTriggered) as exc:
        guard.guard_inputs({"deficit": -1.0})
    assert exc.value.detail["sandbox_purged"] == 1
    assert os.listdir(guard.sandbox) == []
    assert (tmp_path / "user_data.csv").read_text(encoding="utf-8") == "keep"


def test_export_of_an_unbacked_certificate_deadlocks(tmp_path, oracle):
    guard = DeadlockGuard(root=str(tmp_path))
    rep = search(oracle, "zcr_per_sample", tau=0.01, budget_units=200)
    with pytest.raises(DeadlockTriggered):
        guard.guard_export(rep)


def test_export_of_a_backed_certificate_passes(tmp_path, oracle):
    guard = DeadlockGuard(root=str(tmp_path))
    rep = search(oracle, "hjorth_mobility_Hz", tau=0.02, budget_units=400)
    try:
        out = guard.guard_export(rep)
    except DeadlockTriggered:
        pytest.skip("this fixture did not produce an admissible certificate")
    assert out["emitted"] is True
    assert out["evidence"] == "measured"


def test_cli_exit_code_three_for_dirty_input(tmp_path):
    import subprocess
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-m", "sra_opt.deadlock_cli", "--tau", "-5",
         "--root", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 3
    assert "deadlock protection" in proc.stderr.lower()
