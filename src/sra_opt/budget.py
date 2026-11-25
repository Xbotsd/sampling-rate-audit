"""Compute options: reserve a slice of the budget for what the search did not look at.

The design asks for 15% of the compute to be held back for the nodes a search tends to skip,
and for the caller to decide afterwards where it goes. The translation here is deliberately
literal, because the honest version of the idea is simple:

* the reserve is 15% of the unit budget, and it is not spendable by the main search;
* the *hidden set* is the ratios where the surrogate's models disagree most — those are the
  points whose value the search is least able to predict, and therefore the ones it is most
  likely to have got wrong;
* after the report is produced, the caller picks one of two sinks: refine the mathematical
  answer (more points around the crossing, to shrink the order's uncertainty) or deepen the
  audit (the same ratios on more records, to shrink the sampling noise). The two answer
  different questions and are not interchangeable.

The choice is recorded with its cost, so a later reader can tell what the reported numbers
were actually bought with.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

RESERVE_FRACTION = 0.15
HIDDEN_FRACTION = 0.30

SINKS = {
    "math": ("refine the order near the crossing",
             "more ratios around r*, to tighten the fitted exponent and the error bar on it"),
    "audit": ("deepen the audit",
              "the same ratios on more records, to tighten the median and its spread"),
}


@dataclass
class Budget:
    """A unit budget split into a spendable pool and a reserve the search cannot touch."""

    total_units: int
    reserve_fraction: float = RESERVE_FRACTION
    spent_main: int = 0
    spent_reserve: int = 0
    log: list[dict] = field(default_factory=list)

    @property
    def reserve_units(self) -> int:
        return int(round(self.total_units * self.reserve_fraction))

    @property
    def main_units(self) -> int:
        return self.total_units - self.reserve_units

    @property
    def main_left(self) -> int:
        return max(0, self.main_units - self.spent_main)

    @property
    def reserve_left(self) -> int:
        return max(0, self.reserve_units - self.spent_reserve)

    def charge(self, units: int, sink: str = "main") -> bool:
        if sink == "main":
            if units > self.main_left:
                return False
            self.spent_main += units
        else:
            if units > self.reserve_left:
                return False
            self.spent_reserve += units
        self.log.append({"sink": sink, "units": int(units)})
        return True

    def as_dict(self) -> dict:
        return {"total_units": self.total_units, "reserve_fraction": self.reserve_fraction,
                "reserve_units": self.reserve_units, "main_units": self.main_units,
                "spent_main": self.spent_main, "spent_reserve": self.spent_reserve,
                "reserve_left": self.reserve_left, "log": self.log}


def hidden_set(predict, ratios, fraction: float = HIDDEN_FRACTION) -> dict:
    """The ratios the surrogate is least sure about, by model disagreement or tree spread."""
    ratios = np.asarray(ratios, float)
    spread = np.asarray(predict(ratios), float)
    k = max(1, int(round(len(ratios) * fraction)))
    idx = np.argsort(spread)[-k:][::-1]
    return {"ratios": [float(ratios[i]) for i in idx],
            "spread": [float(spread[i]) for i in idx],
            "fraction": fraction}


def spend_option(budget: Budget, sink: str, units: int, feature: str, note: str = "") -> dict:
    """Commit reserve units to one of the two sinks, and record the commitment."""
    if sink not in SINKS:
        raise KeyError("unknown sink %r; choose from %s" % (sink, sorted(SINKS)))
    units = min(int(units), budget.reserve_left)
    if units <= 0:
        return {"committed": False, "reason": "reserve exhausted", "sink": sink}
    budget.charge(units, sink="reserve")
    label, what = SINKS[sink]
    return {"committed": True, "sink": sink, "label": label, "what": what,
            "units": units, "feature": feature, "note": note,
            "reserve_left": budget.reserve_left}


def save(record: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False)
    return path
