"""Strict mode: refuse an operating point that the audit did not measure.

中文说明
--------
`DemoGuard` 只打印。本模块是它的严格版本：越界时写入一条事件记录并抛出
`DeadlockError`，由调用方决定是否中断。

**它不是安全机制。** 没有任何东西能阻止一个脚本忽略异常、删掉记录文件，
或者直接调用拟合器。这里提供的三样东西是工程约束，不是防护：

1. **记录**（`_trigger_deadlock` 写一个 JSON 事件文件）——事后可查；
2. **中断**（抛 `DeadlockError`）——让越界的那次调用无法悄悄返回结果；
3. **沙盒回档**（`Sandbox`）——本次运行产生的中间文件随临时目录一起销毁。

关于"硬件指纹"
--------------
`uuid.getnode()` 返回的是**网卡 MAC 派生的主机号**，不是硬件 UUID：它可被
伪造、会随网卡变化，并且属于个人数据。写进事件文件只是为了区分"同一台机器
上的多次事件"，不构成任何身份认证。不需要时传 ``host_id=False`` 关掉。

关于"脚本检测"
--------------
3 秒阅读时间拦不住脚本（脚本 sleep 4 秒即可），但会误伤从剪贴板粘贴的人。
保留它是为了满足"必须有人读过协议"的流程要求，不是为了技术防护。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__

#: Raised when a strict guard trips. Messages are contractual: tests and the
#: CLI match on them, so do not reword without changing the tests.
DEADLOCK_MESSAGE = "SRA_DEADLOCK_PROTECTION_TRIGGERED"
SCRIPTING_MESSAGE = "Scripting detected. Protocol reading time too short."

SAFE = "SAFE"
WARNING = "WARNING"
BLOCKED = "BLOCKED"

DEFAULT_DEADLOCK_FILE = ".sra_deadlock"
DEFAULT_AGREEMENT_FILE = ".sra_agreement"
AGREEMENT_PHRASE = ("I understand that extrapolation beyond the audited range "
                    "is not evidence")
MIN_READING_SECONDS = 3.0


class DeadlockError(RuntimeError):
    """Strict-mode violation.

    Derives from ``RuntimeError`` so that callers who only know the documented
    contract still catch it, but is its own class so the message stays exact and
    so it can never be confused with an unrelated runtime failure.
    """


# --------------------------------------------------------------------------
# strict guard
# --------------------------------------------------------------------------


class FormalGuard:
    """Tolerance checker with a real consequence: a record and an exception.

    Parameters
    ----------
    tolerance : float
        Admissible paired deficit. Anything strictly above it is a violation.
    strict : bool
        When False the guard only classifies and never raises; use it for
        reporting, not for gating.
    state_path : path, optional
        Where the deadlock event file goes. Defaults to ``.sra_deadlock`` in the
        current working directory. Pass an explicit path in a directory you
        control; a read-only working directory would otherwise make the guard
        raise at the worst possible moment.
    host_id : bool or str
        True (default) writes ``uuid.getnode()`` as a hex string, False omits
        it, a string is written verbatim. See the module docstring: this is a
        host label, not an identity.
    """

    def __init__(self, tolerance, *, strict=True, state_path=None,
                 host_id=True, verbose=True, stream=None):
        tol = float(tolerance)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tolerance must be a positive finite number, got %r"
                             % (tolerance,))
        self.tolerance = tol
        self.strict = bool(strict)
        self.state_path = Path(state_path) if state_path else Path(DEFAULT_DEADLOCK_FILE)
        self.host_id = host_id
        self.verbose = bool(verbose)
        self._stream = stream
        self.history = []

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce(value):
        """Return ``(float_value, problem)`` without ever raising.

        非数字、布尔以外的奇怪类型一律视为未知风险，交给调用方按最高等级处理。
        """
        if isinstance(value, bool):
            return float(value), None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float("nan"), "not a number: %r" % (value,)
        if not np.isfinite(v):
            return v, "not a finite number: %r" % (v,)
        if v < 0.0:
            return v, "negative deficit: %r" % (v,)
        return v, None

    def classify(self, predicted_deficit):
        """``SAFE`` / ``BLOCKED``. Unknown, negative or non-finite is BLOCKED."""
        value, problem = self._coerce(predicted_deficit)
        if problem is not None:
            return BLOCKED
        return SAFE if value <= self.tolerance else BLOCKED

    def check_risk(self, predicted_deficit):
        """Classify without raising. Keeps the ``SAFE``/``WARNING`` vocabulary.

        ``WARNING`` is returned for a value that would be blocked in strict mode
        but is being reported in non-strict mode, so a caller can render a
        warning without the guard tripping.
        """
        level = self.classify(predicted_deficit)
        self.history.append(dict(predicted=predicted_deficit, level=level,
                                 tolerance=self.tolerance))
        if level == SAFE:
            return SAFE
        return WARNING

    def enforce(self, predicted_deficit, context=""):
        """Classify, then raise :class:`DeadlockError` on a violation.

        The exception is raised *outside* any try/except in this module: the
        caller's ``except Exception`` may still catch it (Python has no way to
        forbid that), but nothing in ``leakage_audit`` will swallow it.
        """
        value, problem = self._coerce(predicted_deficit)
        level = self.classify(predicted_deficit)
        self.history.append(dict(predicted=value, level=level,
                                 tolerance=self.tolerance))
        if level == SAFE:
            return SAFE
        if not self.strict:
            return WARNING
        self._trigger_deadlock(value, problem, context)

    # ------------------------------------------------------------------
    def _trigger_deadlock(self, value, problem, context=""):
        """Write the event file, print, then raise. Never returns."""
        reason = problem or ("predicted deficit %.6g exceeds tolerance %.6g"
                             % (value, self.tolerance))
        event = {
            "event": "deadlock",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "predicted_deficit": (None if not np.isfinite(value) else float(value)),
            "tolerance": self.tolerance,
            "context": context,
            "version": __version__,
        }
        if self.host_id is True:
            event["host_id"] = "%012x" % uuid.getnode()
        elif isinstance(self.host_id, str):
            event["host_id"] = self.host_id

        written = self._write_event(event)
        if self.verbose:
            self._emit(reason, event, written)
        raise DeadlockError(DEADLOCK_MESSAGE)

    def _write_event(self, event):
        """Append one JSON line. Returns the path, or None if it could not be written.

        写失败（只读目录、权限）**不会**改变结果：守卫仍然抛出，因为"记不下来"
        不能变成"放行"。失败原因写进事件对象并在终端提示。
        """
        line = json.dumps(event, ensure_ascii=False) + "\n"
        try:
            path = self.state_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
            return path
        except OSError as exc:
            event["write_error"] = str(exc)
            return None

    def _emit(self, reason, event, written):
        stream = self._stream if self._stream is not None else sys.stdout
        bar = "=" * 68
        lines = [
            "",
            bar,
            "  [STRICT] deadlock protection triggered",
            bar,
            "  %s" % reason,
            "  tolerance: %.6g" % self.tolerance,
        ]
        if written:
            lines.append("  event recorded in: %s" % written)
        else:
            lines.append("  event could NOT be recorded (see file error above)")
        lines += [
            "  The run is being aborted; the audit result is not returned.",
            bar,
            "",
        ]
        print("\n".join(lines), file=stream)

    def summary(self):
        blocked = sum(1 for h in self.history if h["level"] == BLOCKED)
        return ("FormalGuard: %d check(s), %d blocked, tolerance = %.3g, strict = %s"
                % (len(self.history), blocked, self.tolerance, self.strict))


# --------------------------------------------------------------------------
# protocol agreement
# --------------------------------------------------------------------------


def protocol_text(version=__version__):
    """The text a user must read before strict mode will run."""
    return f"""
{'=' * 72}
  sampling-rate-audit {version} - STRICT MODE PROTOCOL
{'=' * 72}

  You are about to run the audit with deadlock protection enabled.

  What strict mode does
  ---------------------
  * It fits d = C*r^(-p) to the rates you audited.
  * If the fitted curve predicts a deficit above your tolerance, it writes an
    event record and aborts the run with DeadlockError.
  * Nothing is deleted and nothing outside the sandbox is modified.

  What it cannot do
  -----------------
  * It cannot make an extrapolation true. A power law fitted between r=2.5 and
    r=12.5 says nothing about r=1. Re-measure at the rate you intend to use.
  * It cannot stop a script. The reading-time check is a process requirement,
    not a security control.
  * It cannot identify a machine. The host id is a MAC-derived label.

  Operating point rule
  --------------------
  An out-of-tolerance prediction means: this rate has not been measured for this
  feature. Do not deploy at it on the strength of the fit.

  To continue, type the following line exactly:

      {AGREEMENT_PHRASE}

{'=' * 72}
""".strip("\n")


def require_agreement(*, cache_path=None, input_fn=input, clock=time.monotonic,
                      min_reading_seconds=MIN_READING_SECONDS, skip=False,
                      version=__version__, stream=None, max_attempts=3):
    """Ask for the protocol acknowledgement, or accept a cached one.

    Returns the path of the agreement file (or None when ``skip`` is set and no
    cache is wanted). Raises ``RuntimeError(SCRIPTING_MESSAGE)`` when the whole
    exchange took less than ``min_reading_seconds``: that is the documented
    contract, and the clock is injectable so tests need not sleep.

    ``skip`` (the CLI's ``--i-understand-risks``) records an explicit
    acknowledgement instead of asking: the caller has said, in the command line,
    that they accept the risk. It does not weaken the sandbox or the guard.
    """
    out = stream if stream is not None else sys.stdout
    cache = Path(cache_path) if cache_path else Path(DEFAULT_AGREEMENT_FILE)

    if skip:
        _write_agreement(cache, mode="skipped", version=version)
        return cache

    if _valid_agreement(cache, version):
        return cache

    print(protocol_text(version), file=out)
    started = clock()
    phrase = None
    for attempt in range(max_attempts):
        prompt = "> " if attempt == 0 else "> (try again) "
        print(prompt, end="", file=out, flush=True)
        try:
            phrase = input_fn()
        except EOFError:
            # No stdin (cron, CI, a closed pipe). That is a refusal, not a crash:
            # give the caller a message that says what to do instead.
            raise RuntimeError(
                "the strict-mode protocol needs an interactive confirmation; "
                "pass --i-understand-risks to acknowledge it explicitly, or "
                "run with a terminal") from None
        if phrase.strip() == AGREEMENT_PHRASE:
            break
        print("  That does not match. Copy the line exactly.", file=out)
    else:
        raise ValueError("agreement phrase was not matched after %d attempts"
                         % max_attempts)

    elapsed = clock() - started
    if elapsed < min_reading_seconds:
        raise RuntimeError(SCRIPTING_MESSAGE)

    _write_agreement(cache, mode="typed", version=version)
    return cache


def _valid_agreement(path, version):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("version") == version and data.get("accepted") is True


def _write_agreement(path, *, mode, version):
    payload = {
        "accepted": True,
        "mode": mode,
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # 缓存写不了不是拒绝运行的理由：协议本身已经确认过。
        return None
    return path


# --------------------------------------------------------------------------
# sandbox
# --------------------------------------------------------------------------


class Sandbox:
    """A temporary directory that is destroyed unless the run succeeds.

    Strict mode computes into here and only publishes results after the guard
    has passed. On a deadlock the directory is removed, which is the only
    "rollback" that exists: files the caller wrote elsewhere are not touched,
    and cannot be.

        with Sandbox() as box:
            box.write("fits.json", data)
            guard.enforce(...)      # raises -> box is removed
    """

    def __init__(self, root=None, prefix="sra_sandbox_"):
        self._root = Path(root) if root else None
        # dir=None means "the system temporary directory". Passing "" would
        # mean "the current working directory", which would litter the user's
        # project with sra_sandbox_* directories and break on a read-only cwd.
        self._tmp = tempfile.TemporaryDirectory(
            prefix=prefix, dir=str(self._root) if self._root else None)
        self.path = Path(self._tmp.name)
        self.rolled_back = False
        self.published = False

    # ------------------------------------------------------------------
    def write(self, name, data):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            target.write_bytes(data)
        else:
            target.write_text(str(data), encoding="utf-8")
        return target

    def write_json(self, name, payload):
        return self.write(name, json.dumps(payload, indent=1))

    # ------------------------------------------------------------------
    def rollback(self):
        """Destroy everything this sandbox created. Idempotent."""
        if not self.rolled_back:
            self._tmp.cleanup()
            self.rolled_back = True
        return self.rolled_back

    def publish(self, target_dir=None):
        """Move the sandbox contents to ``target_dir`` and keep them.

        Only called when the run succeeded; this is the one point at which
        anything leaves the sandbox.
        """
        import shutil

        dest = Path(target_dir) if target_dir else None
        if dest is None:
            self.published = True
            return self.path
        dest.mkdir(parents=True, exist_ok=True)
        for item in self.path.iterdir():
            shutil.copy2(item, dest / item.name)
        self.published = True
        return dest

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.rollback()
        return False

    def __del__(self):  # pragma: no cover - interpreter shutdown
        try:
            self.rollback()
        except Exception:
            pass


def format_rollback_message():
    """The sentence shown after a deadlock rolled the sandbox back."""
    return ("数据已回档，原始文件未被修改。请重新阅读协议并检查参数。\n"
            "The sandbox was discarded; your original files were not modified. "
            "Re-read the protocol and check the parameters before running again.")


__all__ = [
    "FormalGuard", "DeadlockError", "Sandbox", "require_agreement",
    "protocol_text", "format_rollback_message", "DEADLOCK_MESSAGE",
    "SCRIPTING_MESSAGE", "SAFE", "WARNING", "BLOCKED", "AGREEMENT_PHRASE",
    "DEFAULT_DEADLOCK_FILE", "DEFAULT_AGREEMENT_FILE", "MIN_READING_SECONDS",
]
