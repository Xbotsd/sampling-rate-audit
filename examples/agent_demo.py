"""Smoke test for the invariance agent on a synthetic record."""
import json
import subprocess
import sys

import numpy as np

fs, T = 500.0, 30.0
rng = np.random.default_rng(3)
t = np.arange(int(fs * T)) / fs
x = np.zeros_like(t)
for k in range(1, 25):
    x += rng.rayleigh(1.0) * np.cos(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
x += 0.02 * rng.standard_normal(x.size)

path = r"D:\new_project\repo\sampling-rate-audit\examples\_synthetic_500hz.npy"
np.save(path, x)
print("wrote", path, x.size, "samples")

exe = sys.executable
tasks = [
    ["profile", "--fs", "500", "--band", "40", "--window", "30"],
    ["full", "--npy", path, "--fs", "500", "--band", "40", "--tolerance", "0.01",
     "--features", "zcr_per_sample", "zcr_per_second", "hjorth_mobility_Hz", "std"],
]
for args in tasks:
    print("\n" + "=" * 78)
    print("$ invariance-agent", " ".join(args))
    print("=" * 78)
    out = subprocess.run([exe, "-m", "leakage_audit.agent_cli"] + args,
                         capture_output=True, text=True, encoding="utf-8")
    print(out.stdout[-2600:])
    if out.returncode:
        print("STDERR:", out.stderr[-800:])
