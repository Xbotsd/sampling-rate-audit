"""Tests for strict mode: deadlock protection, protocol gate, sandbox rollback.

中文说明
--------
这一组测试守的是 V2.1.1 严格模式的四条契约：

1. **越界必须中断**：`FormalGuard.enforce` 抛出 `DeadlockError`，消息固定为
   `SRA_DEADLOCK_PROTECTION_TRIGGERED`，且事件文件里留下时间戳与原因。
2. **垃圾输入等于最高风险**：负数、nan、inf、非数字类型一律 BLOCKED——
   "读不懂"绝不等于"安全"。
3. **协议门不能靠时间糊弄**：阅读时间不足 3 秒直接报
   `Scripting detected. Protocol reading time too short.`；确权后写缓存，
   第二次不再询问。
4. **回档只销毁沙盒**：触发死锁后临时目录被删除，调用方自己的数据逐字节不变。

时间检查用注入的 clock，测试不 sleep。
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest

from leakage_audit import Audit
from leakage_audit.formalguard import (
    AGREEMENT_PHRASE,
    BLOCKED,
    DEADLOCK_MESSAGE,
    SCRIPTING_MESSAGE,
    SAFE,
    WARNING,
    DeadlockError,
    FormalGuard,
    Sandbox,
    format_rollback_message,
    require_agreement,
)


# --------------------------------------------------------------------------
# 1. classification inherits the fixed demo logic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1.0, -1e-9, float("nan"), float("inf"),
                                   float("-inf"), "banana", None, [1, 2]])
def test_garbage_is_blocked(value):
    g = FormalGuard(0.01, verbose=False)
    assert g.classify(value) == BLOCKED


def test_plain_thresholds():
    g = FormalGuard(0.01, verbose=False)
    assert g.classify(0.0) == SAFE
    assert g.classify(0.01) == SAFE          # equal is admissible
    assert g.classify(0.0100001) == BLOCKED


def test_check_risk_keeps_the_demo_vocabulary():
    g = FormalGuard(0.01, verbose=False)
    assert g.check_risk(0.001) == SAFE
    assert g.check_risk(0.5) == WARNING       # report-only, no raise
    assert g.check_risk(-1) == WARNING


def test_nonsense_tolerance_is_refused():
    for bad in (0, -1, float("nan"), None):
        with pytest.raises((ValueError, TypeError)):
            FormalGuard(bad)


def test_numeric_string_tolerance_is_coerced():
    # A CLI or a config file may hand us "0.01"; that is unambiguous and accepted.
    assert FormalGuard("0.01", verbose=False).tolerance == 0.01


# --------------------------------------------------------------------------
# 2. deadlock: record + exception
# --------------------------------------------------------------------------


def test_enforce_raises_the_contract_message(tmp_path):
    g = FormalGuard(0.01, state_path=tmp_path / ".sra_deadlock", verbose=False)
    with pytest.raises(DeadlockError) as info:
        g.enforce(0.5)
    assert str(info.value) == DEADLOCK_MESSAGE
    # It must still be a RuntimeError for callers that only know that contract.
    assert isinstance(info.value, RuntimeError)


def test_deadlock_file_records_timestamp_reason_and_host(tmp_path):
    path = tmp_path / ".sra_deadlock"
    g = FormalGuard(0.01, state_path=path, verbose=False)
    with pytest.raises(DeadlockError):
        g.enforce(0.5, context="feature=zcr r=2.5")

    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["event"] == "deadlock"
    assert event["tolerance"] == 0.01
    assert event["predicted_deficit"] == 0.5
    assert event["context"] == "feature=zcr r=2.5"
    assert event["timestamp"].endswith("+00:00")
    assert isinstance(event["host_id"], str) and event["host_id"]


def test_events_append_rather_than_overwrite(tmp_path):
    path = tmp_path / ".sra_deadlock"
    g = FormalGuard(0.01, state_path=path, verbose=False)
    for value in (0.1, 0.2, 0.3):
        with pytest.raises(DeadlockError):
            g.enforce(value)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(x)["predicted_deficit"] for x in lines] == [0.1, 0.2, 0.3]


def test_host_id_can_be_disabled_or_set(tmp_path):
    path = tmp_path / ".sra_deadlock"
    g = FormalGuard(0.01, state_path=path, host_id=False, verbose=False)
    with pytest.raises(DeadlockError):
        g.enforce(1.0)
    assert "host_id" not in json.loads(path.read_text(encoding="utf-8"))

    path2 = tmp_path / ".sra_deadlock_2"
    g2 = FormalGuard(0.01, state_path=path2, host_id="lab-machine-3", verbose=False)
    with pytest.raises(DeadlockError):
        g2.enforce(1.0)
    assert json.loads(path2.read_text(encoding="utf-8"))["host_id"] == "lab-machine-3"


def test_non_strict_guard_never_raises():
    g = FormalGuard(0.01, strict=False, verbose=False)
    assert g.enforce(0.9) == WARNING
    assert g.history[-1]["level"] == BLOCKED


def test_unwritable_state_path_still_raises(tmp_path):
    """'Could not record' must never become 'let it through'."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    g = FormalGuard(0.01, state_path=blocker / ".sra_deadlock", verbose=False)
    with pytest.raises(DeadlockError):
        g.enforce(0.5)


# --------------------------------------------------------------------------
# 3. protocol agreement
# --------------------------------------------------------------------------


class FakeClock:
    """Monotonic clock we can advance by hand."""

    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t


def test_fast_paste_is_rejected_as_scripting(tmp_path):
    cache = tmp_path / ".sra_agreement"
    clock = FakeClock(0.0)

    with pytest.raises(RuntimeError) as info:
        require_agreement(cache_path=cache, input_fn=lambda: AGREEMENT_PHRASE,
                          clock=clock, stream=io.StringIO())
    assert str(info.value) == SCRIPTING_MESSAGE
    assert not cache.exists(), "a rejected run must not leave an agreement"


def test_slow_enough_paste_writes_the_cache(tmp_path):
    cache = tmp_path / ".sra_agreement"

    def slow_clock():
        # 4 s on the first call (start), 0 s on the second (elapsed) is wrong;
        # return increasing values instead.
        slow_clock.t += 0.0
        return slow_clock.t

    values = iter([0.0, 4.0])
    require_agreement(cache_path=cache, input_fn=lambda: AGREEMENT_PHRASE,
                      clock=lambda: next(values), stream=io.StringIO())
    assert cache.exists()
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["accepted"] is True and payload["mode"] == "typed"


def test_cached_agreement_skips_the_prompt(tmp_path):
    cache = tmp_path / ".sra_agreement"
    values = iter([0.0, 5.0])
    require_agreement(cache_path=cache, input_fn=lambda: AGREEMENT_PHRASE,
                      clock=lambda: next(values), stream=io.StringIO())

    # The second call must not ask: a failing input_fn would raise.
    def explode():
        raise AssertionError("prompted again despite a valid cache")

    require_agreement(cache_path=cache, input_fn=explode,
                      clock=FakeClock(), stream=io.StringIO())


def test_wrong_phrase_is_refused_after_attempts(tmp_path):
    with pytest.raises(ValueError, match="not matched"):
        require_agreement(cache_path=tmp_path / ".sra_agreement",
                          input_fn=lambda: "yes",
                          clock=FakeClock(), stream=io.StringIO(),
                          max_attempts=2)


def test_closed_stdin_is_a_clear_refusal(tmp_path):
    """Cron or CI has no terminal; that must not be an EOFError traceback."""
    def eof():
        raise EOFError()

    with pytest.raises(RuntimeError, match="i-understand-risks"):
        require_agreement(cache_path=tmp_path / ".sra_agreement",
                          input_fn=eof, clock=FakeClock(), stream=io.StringIO())


def test_skip_flag_records_an_explicit_acknowledgement(tmp_path):
    cache = tmp_path / ".sra_agreement"
    require_agreement(cache_path=cache, skip=True, stream=io.StringIO())
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["accepted"] is True
    assert payload["mode"] == "skipped"


# --------------------------------------------------------------------------
# 4. sandbox and rollback
# --------------------------------------------------------------------------


def test_sandbox_rollback_destroys_its_directory():
    box = Sandbox()
    path = box.path
    box.write("fits.json", "{}")
    assert path.exists() and (path / "fits.json").exists()

    assert box.rollback() is True
    assert not path.exists()
    assert box.rolled_back is True


def test_sandbox_rolls_back_on_exception():
    path = None
    with pytest.raises(ValueError):
        with Sandbox() as box:
            path = box.path
            box.write("a.txt", "hello")
            raise ValueError("boom")
    assert path is not None and not path.exists()


def test_sandbox_publish_copies_out(tmp_path):
    with Sandbox() as box:
        box.write("result.json", '{"ok": true}')
        dest = box.publish(tmp_path / "out")
    assert (dest / "result.json").exists()


def test_default_sandbox_lives_in_the_system_temp_dir():
    """Regression: dir='' put every sandbox in the user's working directory."""
    box = Sandbox()
    try:
        assert not str(box.path).startswith(os.getcwd()), (
            "the sandbox must not be created inside the caller's working directory: %s"
            % box.path)
    finally:
        box.rollback()


def test_sandbox_is_cleaned_up_on_any_exception(tmp_path):
    """A failure that is not a deadlock must still not leak a temp directory."""
    engine = Audit(features=TOY_FEATURES, formal_guard=FormalGuard(
        1e6, state_path=tmp_path / ".sra_deadlock", verbose=False),
        tolerance=1e6)
    with pytest.raises(ValueError):
        # every arm is filtered out by the Nyquist rule -> ValueError from _prep
        engine.run(_toy_signal(), fs=500, band=240, rates=[250.0, 125.0])
    assert engine.last_sandbox is not None
    assert engine.last_sandbox.rolled_back is True
    assert not engine.last_sandbox.path.exists()


# --------------------------------------------------------------------------
# 5. Audit integration
# --------------------------------------------------------------------------


def _toy_signal(fs=500.0, seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    x = np.zeros_like(t)
    for k in range(1, 12):
        x += rng.rayleigh(1.0) * np.cos(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
    return x


TOY_FEATURES = ["zcr_per_sample", "turning_per_sample", "spec_entropy", "std"]


def test_strict_audit_raises_and_rolls_back(tmp_path):
    guard = FormalGuard(1e-9, state_path=tmp_path / ".sra_deadlock", verbose=False)
    engine = Audit(features=TOY_FEATURES, formal_guard=guard,
                   tolerance=1e-9)

    with pytest.raises(DeadlockError):
        engine.run(_toy_signal(), fs=500, band=20, rates=[250.0, 125.0, 62.5])

    assert engine.last_sandbox is not None
    assert engine.last_sandbox.rolled_back is True
    assert not engine.last_sandbox.path.exists()
    assert (tmp_path / ".sra_deadlock").exists()


def test_strict_audit_passes_when_the_tolerance_is_loose(tmp_path):
    guard = FormalGuard(1e6, state_path=tmp_path / ".sra_deadlock", verbose=False)
    engine = Audit(features=TOY_FEATURES, formal_guard=guard, tolerance=1e6)
    result = engine.run(_toy_signal(), fs=500, band=20,
                        rates=[250.0, 125.0, 62.5])
    assert result.rows
    assert engine.last_sandbox is not None
    assert engine.last_sandbox.rolled_back is False


def test_deadlock_is_not_swallowed_by_the_fit_loop(tmp_path):
    """_attach_fits catches Exception around the fit; the guard must sit outside it."""
    guard = FormalGuard(1e-9, state_path=tmp_path / ".sra_deadlock", verbose=False)
    engine = Audit(features=TOY_FEATURES, formal_guard=guard, tolerance=1e-9)
    with pytest.raises(DeadlockError):
        engine.run(_toy_signal(), fs=500, band=20, rates=[250.0, 125.0, 62.5])


def test_caller_data_is_untouched_by_a_deadlock(tmp_path):
    guard = FormalGuard(1e-9, state_path=tmp_path / ".sra_deadlock", verbose=False)
    signal = _toy_signal()
    before = signal.tobytes()
    with pytest.raises(DeadlockError):
        Audit(features=TOY_FEATURES, formal_guard=guard, tolerance=1e-9).run(
            signal, fs=500, band=20, rates=[250.0, 125.0, 62.5])
    assert signal.tobytes() == before


def test_rollback_message_is_bilingual():
    text = format_rollback_message()
    assert "数据已回档" in text and "not modified" in text


def test_demo_mode_is_unchanged_without_a_guard():
    result = Audit(features=TOY_FEATURES).run(_toy_signal(), fs=500, band=20,
                                              rates=[250.0, 125.0, 62.5])
    assert result.rows and result.fits
