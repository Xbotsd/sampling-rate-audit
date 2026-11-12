# Changelog

All notable changes to `sampling-rate-audit` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
