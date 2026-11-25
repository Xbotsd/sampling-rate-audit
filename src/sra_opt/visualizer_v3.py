"""Cascade visualisation: show how one changed number produced the answer.

A report that says "r* = 7.3" is not auditable. A report that says which measurements moved
the answer, by how much, and in which direction, is. This module produces three artefacts:

* ``.sra_bug_report`` — a JSON file carrying the seed, the config, the library versions and a
  ready-to-run command that reproduces the report byte-for-byte;
* a **cascade heatmap** — feature x ratio, shaded by log deficit, showing where the grid
  dependence actually lives;
* a **waterfall** — the contributions that moved the reported ratio away from a naive
  reading, in log units, so the dominant one is visible at a glance.

Figures are drawn with Pillow, because this machine has no plotting library and adding one
to render three charts would be a worse dependency than drawing them.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont

    _HAVE_PIL = True
except Exception:  # pragma: no cover - the figures are optional
    Image = ImageDraw = ImageFont = None
    _HAVE_PIL = False

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
]

INK = (32, 32, 36)
MUTED = (112, 112, 120)
GRID = (222, 222, 228)
BG = (255, 255, 255)


def _require_pil() -> None:
    if not _HAVE_PIL:
        raise RuntimeError(
            "figure rendering needs Pillow: pip install \"sampling-rate-audit[opt]\"")


def _font(size: int):
    _require_pil()
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _colour(t: float) -> tuple[int, int, int]:
    """Blue -> white -> red, t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        u = t / 0.5
        return (int(40 + u * 215), int(80 + u * 175), int(190 + u * 65))
    u = (t - 0.5) / 0.5
    return (int(255), int(255 - u * 175), int(255 - u * 205))


def heatmap(path: str, features: list[str], ratios, matrix: np.ndarray,
            title: str = "Cascade perturbation: log10 relative deficit by feature and ratio",
            vmin: float | None = None, vmax: float | None = None) -> str:
    _require_pil()
    m, n = matrix.shape
    cw, ch = 74, 34
    left, top, right, bottom = 190, 70, 40, 96
    W = left + n * cw + right
    H = top + m * ch + bottom
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_lab, f_small = _font(17), _font(13), _font(11)

    d.text((left, 22), title, font=f_title, fill=INK)
    finite = matrix[np.isfinite(matrix)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    span = max(hi - lo, 1e-12)

    for i in range(m):
        y = top + i * ch
        d.text((16, y + ch / 2 - 7), features[i], font=f_lab, fill=INK)
        for j in range(n):
            x = left + j * cw
            v = matrix[i, j]
            if not np.isfinite(v):
                d.rectangle([x, y, x + cw - 3, y + ch - 3], fill=(240, 240, 242))
                d.text((x + 8, y + ch / 2 - 6), "n/a", font=f_small, fill=MUTED)
                continue
            vc = float(np.clip(v, lo, hi))
            t = (vc - lo) / span
            d.rectangle([x, y, x + cw - 3, y + ch - 3], fill=_colour(t))
            txt = "%.1f" % vc
            if v < lo:
                txt = "≤" + txt
            elif v > hi:
                txt = "≥" + txt
            d.text((x + 6, y + ch / 2 - 6), txt, font=f_small,
                   fill=(255, 255, 255) if t < 0.22 or t > 0.82 else INK)

    for j, r in enumerate(ratios):
        x = left + j * cw + cw / 2
        d.text((x - 12, top + m * ch + 8), "%g" % r, font=f_small, fill=MUTED)
    d.text((left, top + m * ch + 30), "rate ratio r = fs / B", font=f_small, fill=MUTED)
    d.text((16, top + m * ch + 30), "log10 D", font=f_small, fill=MUTED)

    bar_x, bar_y, bar_w, bar_h = left, H - 34, n * cw - 3, 10
    for k in range(bar_w):
        d.line([(bar_x + k, bar_y), (bar_x + k, bar_y + bar_h)],
               fill=_colour(k / max(bar_w - 1, 1)))
    d.text((bar_x, bar_y - 16), "low  (%.2f)" % lo, font=f_small, fill=MUTED)
    d.text((bar_x + bar_w - 60, bar_y - 16), "high (%.2f)" % hi, font=f_small, fill=MUTED)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def waterfall(path: str, contributions: list[dict], baseline: float,
              title: str = "Sensitivity: how far each measurement moves the reported ratio") -> str:
    """Zero-centred bars. ``contributions`` are deltas in log10 units from the baseline."""
    _require_pil()
    W = 960
    H = 120 + 44 * max(len(contributions), 1) + 60
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_lab, f_small = _font(16), _font(13), _font(11)
    d.text((28, 22), title, font=f_title, fill=INK)
    d.text((28, 46), "each row: refit the curve without that one measurement",
           font=f_small, fill=MUTED)

    vals = [c["delta"] for c in contributions] or [0.0]
    scale = max(max(abs(v) for v in vals), 1e-12)
    zero_x, half = 440, 280
    y = 82
    for c in contributions:
        delta = c["delta"]
        w = abs(delta) / scale * half
        if w < 1.2:                      # a zero-width bar still has to be visible
            w = 1.2
        x0 = zero_x if delta >= 0 else zero_x - w
        colour = (198, 62, 48) if delta > 0 else (48, 104, 186)
        d.rectangle([x0, y, x0 + w, y + 24], fill=colour)
        d.text((28, y + 4), c["label"], font=f_lab, fill=INK)
        txt = "%+.2e" % delta if abs(delta) < 1e-3 else "%+.4f" % delta
        d.text((806, y + 4), txt, font=f_small, fill=MUTED)
        y += 44
    d.line([(zero_x, 74), (zero_x, y + 4)], fill=GRID, width=2)
    d.text((28, y + 14), "baseline reported ratio  r* = %.4f" % baseline, font=f_lab, fill=INK)
    worst = max(abs(v) for v in vals)
    d.text((28, y + 38),
           "largest single-measurement influence: %.2e in log10 r (%.4f%%)"
           % (worst, 100 * (10 ** worst - 1)), font=f_small, fill=MUTED)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def write_bug_report(path: str, report: dict, seed: int, command: str,
                     inputs: dict | None = None) -> str:
    """A reproducible record of one run: seed, config, versions, and how to replay it."""
    body = {
        "kind": "sra-bug-report",
        "sra_version": "3.1.1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(seed),
        "replay": command,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs": inputs or {},
        "report": report,
    }
    payload = json.dumps(body, sort_keys=True, indent=1)
    body["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(body, fh, indent=1, ensure_ascii=False)
    return path


def replay_matches(path: str, report: dict) -> bool:
    """True when a re-run produced the same report body, ignoring the timestamp."""
    with open(path, encoding="utf-8") as fh:
        body = json.load(fh)
    a = json.dumps(body["report"], sort_keys=True)
    b = json.dumps(report, sort_keys=True)
    return a == b
