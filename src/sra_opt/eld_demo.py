"""Sandbox and Evil-Lock-Detector: try anything, but leave a trace when someone cheats.

The earlier design paired an optimiser with a hard lock: any excursion froze the machine.
That is the wrong instrument. An optimiser explores; a proposal that violates a constraint
is an expected event, not an attack, and freezing on it teaches the caller to disable the
guard. This module separates the two cases that the hard lock conflated:

* **out of bounds** — a proposal outside the admissible region. The sandbox rolls back to
  the last checkpoint, records the attempt, and continues. Nothing is frozen, because
  nothing dishonest happened.
* **integrity violation** — the thing being optimised against was edited underneath the
  run: a surrogate file whose hash changed mid-run, regression data that no longer matches
  the digest it was registered with, or a decision record altered after signing. This is
  the only case that freezes, and it freezes the run, not the machine.

Scope, stated plainly: this detects accidental and casual tampering, such as a stale file, a
hand-edited JSON, or a rerun against modified weights. It does **not** defend against a
process with write access that also rewrites the digests; that needs a trust boundary this
program does not have, and pretending otherwise would be worse than having no guard.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class Frozen(RuntimeError):
    """Raised when a run has been frozen by the Evil-Lock-Detector."""


@dataclass
class Attempt:
    n: int
    proposal: dict
    outcome: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {"n": self.n, "proposal": self.proposal, "outcome": self.outcome,
                "reason": self.reason}


@dataclass
class Sandbox:
    """Runs proposals against a state dict, rolling back instead of punishing."""

    state: dict
    admissible: dict = field(default_factory=dict)     # proposal key -> predicate
    log: list[Attempt] = field(default_factory=list)
    frozen: bool = False
    freeze_reason: str = ""
    _checkpoint: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._checkpoint = copy.deepcopy(self.state)

    # ------------------------------------------------------------------ control
    def freeze(self, reason: str) -> None:
        self.frozen = True
        self.freeze_reason = reason

    def _guard(self) -> None:
        if self.frozen:
            raise Frozen("run is frozen: %s" % self.freeze_reason)

    # ------------------------------------------------------------------ propose
    def propose(self, proposal: dict, apply=None) -> dict:
        """Try a proposal. Out of bounds -> silent rollback. Integrity -> never reached here."""
        self._guard()
        for key, pred in self.admissible.items():
            if key not in proposal:
                continue
            try:
                allowed = bool(pred(proposal[key]))
            except Exception as exc:                  # noqa: BLE001 - a bad proposal is not a crash
                reason = "%s could not be validated (%s)" % (key, type(exc).__name__)
                self.log.append(Attempt(len(self.log) + 1, dict(proposal), "rolled-back", reason))
                self.state = copy.deepcopy(self._checkpoint)
                return {"accepted": False, "rolled_back": True, "reason": reason}
            if not allowed:
                self.log.append(Attempt(len(self.log) + 1, dict(proposal), "rolled-back",
                                        "%s outside the admissible region" % key))
                self.state = copy.deepcopy(self._checkpoint)
                return {"accepted": False, "rolled_back": True,
                        "reason": "%s outside the admissible region" % key}
        try:
            if apply is not None:
                apply(self.state, proposal)
            else:
                self.state.update(proposal)
        except Exception as exc:                      # noqa: BLE001 - rollback is the point
            self.state = copy.deepcopy(self._checkpoint)
            self.log.append(Attempt(len(self.log) + 1, dict(proposal), "rolled-back",
                                    "%s: %s" % (type(exc).__name__, exc)))
            return {"accepted": False, "rolled_back": True, "reason": str(exc)}
        self._checkpoint = copy.deepcopy(self.state)
        self.log.append(Attempt(len(self.log) + 1, dict(proposal), "accepted"))
        return {"accepted": True, "rolled_back": False, "reason": ""}

    def summary(self) -> dict:
        return {"attempts": len(self.log),
                "accepted": sum(1 for a in self.log if a.outcome == "accepted"),
                "rolled_back": sum(1 for a in self.log if a.outcome == "rolled-back"),
                "frozen": self.frozen, "freeze_reason": self.freeze_reason}


@dataclass
class EvilLockDetector:
    """Watches the objects a run depends on, and freezes the run if one of them changes."""

    path: str
    watched: dict = field(default_factory=dict)       # name -> expected sha256
    events: list[dict] = field(default_factory=list)

    def register(self, name: str, file_path: str) -> str:
        d = digest(file_path)
        self.watched[name] = d
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"watched": self.watched}, fh, indent=1)
        return d

    def check(self, files: dict) -> dict:
        """Compare the current files against the registered digests."""
        bad = []
        for name, path in files.items():
            want = self.watched.get(name)
            if want is None:
                continue
            have = digest(path) if os.path.exists(path) else "<missing>"
            if have != want:
                bad.append({"name": name, "expected": want[:16], "found": have[:16]})
        event = {"kind": "integrity", "ok": not bad, "violations": bad,
                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.events.append(event)
        return event


def demo(path_dir: str = "out/eld") -> dict:
    """The three cases, side by side: legitimate retry, out of bounds, tampering."""
    os.makedirs(path_dir, exist_ok=True)

    surrogate_file = os.path.join(path_dir, "surrogate_weights.json")
    with open(surrogate_file, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"weights": [0.1, 0.2, 0.3], "n_trees": 64}, fh)

    eld = EvilLockDetector(os.path.join(path_dir, ".eld_state.json"))
    eld.register("surrogate", surrogate_file)

    # --- case 1 and 2: the sandbox
    box = Sandbox(state={"r": 25.0, "tau": 0.05, "accepted": 0},
                  admissible={"r": lambda v: 2.5 <= float(v) <= 200.0})
    legit = box.propose({"r": 18.0})
    greedy = box.propose({"r": 500.0})
    exploding = box.propose({"r": "not-a-number"},
                            apply=lambda st, p: st.update({"r": float(p["r"])}))
    frozen_by_proposal = box.frozen

    # --- case 3: tampering with a watched object
    with open(surrogate_file, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"weights": [9.9, 9.9, 9.9], "n_trees": 64}, fh)
    event = eld.check({"surrogate": surrogate_file})
    if not event["ok"]:
        box.freeze("surrogate weights changed after registration")
    try:
        box.propose({"r": 40.0})
        blocked = False
    except Frozen:
        blocked = True

    return {
        "case_legit_retry": legit,
        "case_out_of_bounds": greedy,
        "case_exception": exploding,
        "frozen_by_out_of_bounds": frozen_by_proposal,
        "integrity_event": event,
        "proposal_blocked_after_freeze": blocked,
        "sandbox": box.summary(),
        "scope": "detects stale or casually edited inputs; not a defence against a process "
                 "that can rewrite the digests",
    }
