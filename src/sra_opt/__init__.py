"""SRA optimiser layer: cheap-by-default, honest-when-cornered.

Version 3.1.1. The measurement engine is `leakage_audit` (v1.0.1), unchanged; this
package adds the search, arbitration and audit-trail layers on top of it.
"""
from __future__ import annotations

__version__ = "3.1.1"

from .oracle import Oracle, feature_menu  # noqa: E402,F401
from .grid import CostModel, GridPlan, candidates, required_ratio  # noqa: E402,F401

__all__ = [
    "__version__", "Oracle", "feature_menu", "CostModel", "GridPlan",
    "candidates", "required_ratio",
]
