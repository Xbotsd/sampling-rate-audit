# Changelog

All notable changes to `sampling-rate-audit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.1] — 2026-11-25

The optimiser layer, merged onto the 2.1.x guard rather than beside it.

The engine's arithmetic is unchanged: same eighteen definitions, same `--selftest` table.
What is new is the ability to ask which grid to build, and to refuse when the answer is not
measured.

### Added

- `sra_opt`, an optional companion package:
  - `oracle` — per-record feature values on synthetic records, billed per point and cached per
    ratio, so the cost of an audit is a number rather than an adjective;
  - `surrogate` — PCHIP, polynomial and random-forest fits of the deficit curve, an
    extrapolation gate that ignores fit quality beyond 2x, and a signed risk agreement for
    callers who lower the threshold;
  - `pareto` — predict-then-verify search for the cheapest `(anchor, rate)` pair that meets a
    tolerance, returning a certificate labelled *measured*, *interpolated* or *extrapolated*,
    and a `ValueChoice` instead of a verdict when the cheap and defensible answers differ;
  - `arbitrator` — logistic track selection plus an OLS combination, with `CrossTrackReader`
    for verifying a foreign result before merging it;
  - `deadlock` — the export gate. Dirty inputs and unbacked exports exit 3; exploration rolls
    back. Built on `leakage_audit.formalguard`, so there is one exception class, one trace
    file and one JSON schema rather than two of each;
  - `budget` — a 15 % reserve the search cannot spend, and two later sinks for it;
  - `visualizer_v3` — `.sra_bug_report`, a defect heatmap and a leave-one-out sensitivity chart.
- Console scripts `sra-opt` and `sra-opt-dp`.
- `[opt]` and `[dev]` extras. `numpy` remains the only dependency of the engine.
- 35 tests for the new layer, including a subprocess check of the exit-code contract.

### Changed

- Version to 3.1.1 across `pyproject.toml`, `leakage_audit.__version__`, `CITATION.cff` and
  `.zenodo.json`.

### Notes

- `import sra_opt` works without the `opt` extra; only the figure drawing needs Pillow.
- The arbitration model ships with nine training cases. That is enough to expose a
  disagreement between the two tracks and not enough to settle it, and the module says so.
- The optimiser's trace lines extend `formalguard`'s schema with `phase`, `machine` and
  `detail`; every field the existing guard writes is still written unchanged.

## [2.1.2] — 2026-11-18 — **Formal Guard Edition**

Adds a fitted decay curve and strict mode on top of the unchanged 1.0.1 engine.
Tagged `v2.1.2`, **not archived**; the citable artefact remains **1.0.1**
(`doi:10.5281/zenodo.24154075`). See `RELEASE_NOTES_V2.1.X.md`.

### Added — order estimation and visualisation

- `leakage_audit/estimator.py` — fits `d = C·r^(−p)` with `nls` (`curve_fit`,
  linear space), `mle` (log-normal errors, scale profiled out, analytic gradient,
  L-BFGS-B) and `huber` (Huber loss, MAD-scaled and fixed at the starting
  residual). Points with zero, negative or non-finite deficit are dropped and
  counted; too few points yields `ok=False` with a reason instead of an exception.
- `leakage_audit/visualizer.py` — `plot_decay()`: measured points, fitted curve,
  and green / amber / red bands derived from where the nominated curve crosses the
  tolerance; the audited range is marked and everything right of it is labelled
  extrapolation.
- `Audit(fit_order=..., estimator_method=...)` attaches per-feature fits to
  `AuditResult.fits`; `fit_markdown()` prints them and `to_json()` includes them.
  `fit_order=False` reproduces 1.0.1 behaviour exactly.
- `examples/v2_demo_tutorial.py` — end-to-end walkthrough.

### Added — strict mode

- `leakage_audit/formalguard.py`:
  - `FormalGuard` classifies a predicted deficit and, when `strict`, appends an
    event to `.sra_deadlock` and raises `DeadlockError` (a `RuntimeError` whose
    message is exactly `SRA_DEADLOCK_PROTECTION_TRIGGERED`). Negative, non-finite
    and non-numeric values count as the highest risk. A file write failure never
    turns into a pass.
  - `require_agreement()` prints the protocol, requires the phrase to be typed,
    and raises `RuntimeError(Scripting detected. Protocol reading time too short.)`
    when the exchange takes under 3 s. The cache `.sra_agreement` skips later
    prompts; the clock is injectable so tests do not sleep.
  - `Sandbox` is a temporary directory that is destroyed on failure; `Audit`
    computes inside it and rolls it back before re-raising.
- CLI: `--strict`, `--tolerance`, `--i-understand-risks`, `--agreement-cache`,
  `--deadlock-file`. A tripped deadlock exits with code 3.

### Added — documented exit codes

`0` completed, `1` data/parameter error, `2` protocol refused, `3` deadlock.
Usage errors from the argument parser now exit 1 instead of argparse's default 2,
so 2 means the protocol gate and nothing else. Exposed as `EXIT_OK`,
`EXIT_DATA`, `EXIT_PROTOCOL`, `EXIT_DEADLOCK` in `cli.py`.

### Changed

- `DEMO_BUILD` is now `False` on this line (it was `True` on `V2.1.1-demo`);
  `STRICT_MODE_AVAILABLE` is `True`.
- `Audit.run`/`run_many` route through a sandbox when a `formal_guard` is given.

### Explicitly not provided

- No enforcement that cannot be bypassed: the event file is a log, not an
  interlock; the reading-time check cannot detect a script; nothing identifies a
  machine (the host id is a MAC-derived label, and can be omitted).
- No rollback of anything outside the sandbox. Nothing outside the sandbox is
  ever written.

### Unchanged

- The audit itself: `AuditResult.rows`, `markdown()`, `exponents()`, the CSV
  columns, the registry and the agent are the 1.0.1 behaviour.

## [1.0.1] — 2026-11-12

### Fixed

- `CITATION.cff` now cites the **concept DOI** `10.5281/zenodo.24123019` instead of a single version
  DOI. A version DOI written into a package that predates its own release always points one version
  back; the concept DOI always resolves to the latest archived version, so the file cannot go stale.

### Notes

- No code change: the engine, the registry and the command line are identical to 1.0.0. The only
  difference is the citation metadata inside the package.

## [1.0.0] — 2026-11-12

Third generation of the theory. The invariance order is now defined on the deficit from the
continuous limit rather than on a relative expansion, so it stays defined for definitions whose
value tends to zero.

### Added

- `leakage_audit.__version__`, and `leakage-audit --version`.
- Tests that the package version equals `pyproject.toml`, and that both console-script targets
  expose a callable `main()`.
- `CHANGELOG.md` and a versioning statement in the README.

### Fixed

- `katz_fd` and `petrosian_fd` were labelled `N_IN_FORMULA`, which the old look-up table read as
  order 1. Both enter the sample count through a logarithm, so their deficit decays like
  `1/log n`: they are now labelled `LOGBIN_NORM`. This changes the `plan` output for those two
  definitions — a behaviour change, hence the major version.

### Notes

- Labels returned by the engine remain heuristic structural warnings, not invariance orders. The
  order itself is measured; see the paper's Theorem 4 for the Katz case.

## [0.2.0] — 2026-11-11

### Added

- `.zenodo.json`, so the release is archived with title, description, MIT licence, ORCID and a
  link to the superseded preprint.
- DOI `10.5281/zenodo.24123020`; `CITATION.cff` now declares it.

### Changed

- README and CITATION aligned with the submitted manuscript: the preprint DOI is no longer cited
  as a software DOI, and "every number is a lower bound" is replaced by "neither an upper nor a
  lower bound".
- Documents 18 registered definitions, 16 audited in the paper, and five implementation signatures
  grouped into the paper's four structural classes.

## [0.1.2] — 2026-11-07

### Added

- `invariance-agent` for first-time users and programmatic use (guided wizard, `plan`/`audit`/
  `advise`/`full` actions, JSON task interface).

### Changed

- Package version set to 0.1.2.

## [0.1.0] — 2026-10-27

### Added

- First public release: the audit engine with 18 registered definitions, the `leakage-audit`
  command line, tests and examples.
- DOI `10.5281/zenodo.23679114` recorded for the accompanying preprint.
