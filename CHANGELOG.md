# Changelog

All notable changes to `sampling-rate-audit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1-demo] — 2026-11-17 — **branch only, never released**

A teaching sandbox layered on top of 1.0.1. No tag, no Zenodo record, no DOI, and not
supported for production or publication. `main` remains the audited line.

### Added

- `leakage_audit/estimator.py` — fits `d = C·r^(−p)` to the paired deficits with three
  estimators: `nls` (`scipy.optimize.curve_fit`, linear space), `mle` (log-normal errors with
  the scale profiled out, analytic gradient, L-BFGS-B) and `huber` (Huber loss, MAD-scaled and
  fixed at the starting residual so the surface does not shift under the optimiser). Points with
  zero, negative or non-finite deficit are dropped and counted; too few points yields
  `ok=False` with a reason rather than an exception.
- `leakage_audit/demoguard.py` — `DemoGuard.check_risk()` compares a predicted deficit with a
  tolerance and prints a warning. It returns `"SAFE"` / `"WARNING"`, never raises, never blocks,
  never writes, and touches no data.
- `leakage_audit/visualizer.py` — `plot_decay()` draws measured points, the fitted curve, and
  green / amber / red bands derived from where the nominated curve crosses the tolerance.
  The audited range is marked, and everything to the right of it is labelled extrapolation.
- `Audit(fit_order=True, estimator_method="nls", tolerance=None)` — attaches per-feature fits to
  `AuditResult.fits`; `AuditResult.fit_markdown()` prints them; `to_json()` includes them when
  present. `fit_order=False` reproduces 1.0.1 behaviour exactly.
- `examples/v2_demo_tutorial.py` — end-to-end walkthrough that prints the audit table, the fitted
  exponents, the guard's warnings, and writes a figure.
- `pyproject.toml` extra `[demo]` = `scipy` + `matplotlib`. The core package still needs only numpy;
  both are imported lazily, inside the functions that use them.
- `DEMO_BUILD` flag, so the build's identity is visible from code and not only from this file.

### Unchanged

- The audit itself. `AuditResult.rows`, `markdown()`, `exponents()`, the CSV columns, the registry,
  the CLI and the agent are byte-for-byte the 1.0.1 behaviour; the demo only adds `fits`.

### Explicitly not provided

- Any enforcement. There is no interception, no deadlock, no rollback, no sandbox, no locking.
  The guard prints. That is the whole contract.

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
