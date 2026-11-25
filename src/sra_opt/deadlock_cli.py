"""Deadlock Protection as its own entry point, so a pipeline can call it directly.

Exit codes: ``0`` clean, ``3`` DP fired, ``4`` something else. See
:mod:`sra_opt.deadlock` for what fires and what merely rolls back.
"""
from __future__ import annotations

import sys

from .cli import _cmd_dp, main as _cli_main

__all__ = ["main"]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return _cli_main(["dp"] + argv)


if __name__ == "__main__":
    raise SystemExit(main())
