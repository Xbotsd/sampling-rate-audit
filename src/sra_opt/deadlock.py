"""Deadlock Protection, phase two: the export gate on top of :mod:`leakage_audit.formalguard`.

``formalguard`` already answers one question well: *is this predicted deficit inside the
tolerance the caller declared?* It classifies, writes ``.sra_deadlock``, and raises
:class:`DeadlockError` in strict mode. It does not answer a second question, which is the one
that matters after a search has finished: *may this certificate be emitted at all?*

That is what this module adds, and it adds nothing else. The exception class, the trace file,
its JSON line schema and the host label all come from ``formalguard`` so that one file and one
``except`` clause serve both guards. Two guards writing two formats into the same directory is
the failure mode this integration exists to avoid.

The boundary the earlier design got wrong is now explicit:

**Rollback — the caller is exploring.** An out-of-bounds proposal inside a search is expected
and costs nothing but a checkpoint. It does not freeze the run, write a trace, or touch the
caller's files.

**Deadlock — the caller is asserting something unsupported.** Two cases:

* dirty input: NaN, infinity, a negative deficit, a boolean or a string where a number
  belongs. A deficit cannot be negative, so a negative one is not a tighter tolerance, it is
  a broken caller;
* unbacked export: emitting a certificate whose evidence is prediction rather than
  measurement, or whose anchor floor sits above the tolerance, or which does not exist at all
  because the fitted order never reaches the target. This is the case DP was built for — a
  result that would read as a measurement while being nothing of the kind.

Exit codes are the CI contract: ``0`` emitted, ``3`` deadlock, ``4`` anything else.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from leakage_audit import __version__
from leakage_audit.formalguard import DEFAULT_DEADLOCK_FILE, DeadlockError

EXIT_CODE = 3
TRACE_NAME = DEFAULT_DEADLOCK_FILE


class DeadlockTriggered(DeadlockError):
    """DP fired. Subclasses ``formalguard.DeadlockError`` so one ``except`` catches both."""

    exit_code = EXIT_CODE

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__("deadlock protection: %s" % reason)
        self.reason = reason
        self.detail = detail or {}


def host_id() -> str:
    """The same host label ``formalguard`` writes: ``%012x`` of ``uuid.getnode()``.

    ``getnode()`` is a MAC-derived host number, not a hardware identity, and it falls back to
    a random value when no MAC is available. It links runs on the same host and identifies
    nobody; ``formalguard``'s module docstring makes the same point at length.
    """
    return "%012x" % uuid.getnode()


def machine_fingerprint() -> str:
    """A shorter, salted label for the optimiser's own reports.

    Kept alongside :func:`host_id` because a 16-hex tag is easier to eyeball in a trace than a
    12-hex MAC-derived number, and because a caller may want a label that is not the host id.
    """
    raw = "%s|%s|%s" % (host_id(), platform.node(), platform.machine())
    import hashlib

    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_inputs(params: dict) -> dict:
    """Reject the four dirty-input classes. Mirrors ``FormalGuard._coerce`` and extends it."""
    problems = []
    for name, value in params.items():
        if isinstance(value, bool):
            problems.append({"field": name, "value": repr(value),
                             "why": "a boolean where a number belongs"})
            continue
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                problems.append({"field": name, "value": repr(value),
                                 "why": "a string that is not a number"})
                continue
        if _is_number(value):
            f = float(value)
            if math.isnan(f):
                problems.append({"field": name, "value": "nan", "why": "NaN"})
            elif math.isinf(f):
                problems.append({"field": name, "value": "inf", "why": "infinite"})
            elif name.endswith("deficit") and f < 0:
                problems.append({"field": name, "value": f,
                                 "why": "a deficit cannot be negative"})
            elif name in ("tau", "ratio", "anchor") and f <= 0:
                problems.append({"field": name, "value": f,
                                 "why": "%s must be positive" % name})
    if problems:
        raise DeadlockTriggered("dirty input", {"violations": problems})
    return {"ok": True, "checked": sorted(params)}


@dataclass
class DeadlockGuard:
    """Trace file, sandbox root, and the export gate."""

    root: str = "."
    sandbox: str | None = None
    trace: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sandbox is None:
            self.sandbox = tempfile.mkdtemp(prefix="sra_sandbox_")

    # ------------------------------------------------------------------ trace
    @property
    def trace_path(self) -> str:
        return os.path.join(self.root, TRACE_NAME)

    def _record(self, reason: str, detail: dict) -> dict:
        """Append one line in ``formalguard``'s schema, plus the optimiser's own fields."""
        entry = {
            "event": "deadlock",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "predicted_deficit": detail.get("predicted_deficit"),
            "tolerance": detail.get("tau"),
            "context": detail.get("feature", ""),
            "version": __version__,
            "host_id": host_id(),
            "phase": "optimiser-export",
            "machine": machine_fingerprint(),
            "detail": detail,
        }
        self.trace.append(entry)
        try:
            os.makedirs(self.root, exist_ok=True)
            with open(self.trace_path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # a read-only directory must not turn "cannot record" into "may proceed"
        return entry

    # ------------------------------------------------------------------ sandbox hygiene
    def purge_sandbox(self) -> int:
        """Delete the guard's own temporary directory. Nothing outside it is touched."""
        removed = 0
        if self.sandbox and os.path.isdir(self.sandbox):
            for name in os.listdir(self.sandbox):
                path = os.path.join(self.sandbox, name)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    removed += 1
                except OSError:
                    pass
        return removed

    def _fire(self, reason: str, detail: dict) -> DeadlockTriggered:
        detail = dict(detail)
        detail["sandbox_purged"] = self.purge_sandbox()
        detail["trace"] = self.trace_path
        self._record(reason, detail)
        return DeadlockTriggered(reason, detail)

    # ------------------------------------------------------------------ the gates
    def guard_inputs(self, params: dict) -> dict:
        try:
            return validate_inputs(params)
        except DeadlockTriggered as exc:
            raise self._fire("dirty input", exc.detail) from None

    def guard_export(self, report) -> dict:
        """Refuse to emit a certificate that is not backed by measurement."""
        cert = report.certificate
        if cert is None:
            verdict = report.verdict() if hasattr(report, "verdict") else "no certifiable answer"
            raise self._fire("no certifiable answer", {
                "feature": getattr(report, "feature", "?"),
                "tau": getattr(report, "tau", None),
                "verdict": verdict,
                "measured_ratios": list(getattr(report, "measured", [])),
            })
        if not getattr(cert, "admissible", False):
            raise self._fire("unbacked export", {
                "feature": getattr(report, "feature", "?"),
                "tau": getattr(report, "tau", None),
                "device_ratio": getattr(cert, "device_ratio", None),
                "predicted_deficit": getattr(cert, "predicted_g", None),
                "evidence": getattr(cert, "evidence", "?"),
                "flags": list(getattr(cert, "flags", [])),
            })
        return {"emitted": True, "feature": getattr(report, "feature", "?"),
                "device_ratio": cert.device_ratio, "evidence": cert.evidence}


def cli_report(params: dict, report, root: str = ".") -> int:
    """The CI contract: 0 on a clean emit, 3 on DP, 4 on an unexpected error."""
    guard = DeadlockGuard(root=root)
    try:
        guard.guard_inputs(params)
        out = guard.guard_export(report)
        print(json.dumps({"status": "ok", **out}, ensure_ascii=False))
        return 0
    except DeadlockTriggered as exc:
        print(json.dumps({"status": "deadlock", "reason": exc.reason,
                          "exit_code": exc.exit_code, **exc.detail}, ensure_ascii=False),
              file=os.sys.stderr)
        return exc.exit_code
