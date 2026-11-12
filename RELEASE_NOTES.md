## What's new in v1.0.1

Citation metadata only; there is no code change.

- `CITATION.cff` cites the concept DOI `10.5281/zenodo.24123019`, which always resolves to the latest
  archived version. Version 1.0.0 shipped a `CITATION.cff` naming the previous version's DOI, because
  the package was built before its own release existed. That cannot be fixed in place: a published
  Zenodo record accepts metadata edits but not file replacements.

## What's new in v1.0.0

Third generation of the theory: the invariance order is defined on the deficit from the
continuous limit, `|F(r) - F_inf| = Theta(r^-p)`, instead of on the ratio `F(r)/F_inf`.
The ratio is undefined when the limit is zero, which is exactly the case that broke v0.x.

- **Corrected labels.** `katz_fd` and `petrosian_fd` enter the sample count through a logarithm,
  so their deficit decays like `1/log n`. They were labelled `N_IN_FORMULA` (read as order 1) and
  are now `LOGBIN_NORM`. This changes `plan` output for those two definitions.
- **Version is readable from code.** `leakage_audit.__version__` and `leakage-audit --version`,
  with a test that keeps it equal to `pyproject.toml`.
- **CHANGELOG.md** added, recording 0.1.0 through 1.0.0.
- **API stability declared.** From here the public surface follows semantic versioning.

Engine arithmetic is unchanged: 26 tests pass, and `--selftest` reproduces the same exponent table.

## What's new in v0.2.0

Metadata and documentation corrected to match the manuscript under review:

- README no longer carries the preprint DOI badge; `CITATION.cff` no longer declares a software DOI it does not have. This release is the first software DOI.
- Metrological scope restated: the audit isolates the digital-grid component and is **neither an upper nor a lower bound** on the device-level discrepancy.
- Documents 18 registered definitions vs 16 audited in the paper, and five implementation signatures grouped into the paper's four structural classes.
- Adds `.zenodo.json`, so this release is archived with its title, description, MIT licence, ORCID, and a link to the superseded preprint.

Engine unchanged from v0.1.2: 18 definitions, the `leakage-audit` CLI, `invariance-agent`, 24 tests.
