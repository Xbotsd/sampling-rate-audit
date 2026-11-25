## What's new in v3.1.1

An optional search layer. The measurement engine is untouched.

- **Ask which grid to build.** `sra_opt.pareto.search()` takes a tolerance and a cost model and
  returns the cheapest `(anchor, rate)` pair that meets the tolerance, with the evidence behind
  it. On the demo it lands on `r = 7.0` with an anchor at 25 for `hjorth_mobility_Hz` at a 2 %
  tolerance, for 39 % of the compute a defensive sweep costs.
- **Refuse when the answer is not measured.** Certificates are labelled *measured*,
  *interpolated* or *extrapolated*. A cheapest-feasible answer resting on extrapolation comes
  back as a `ValueChoice` with three priced options, not as a verdict.
- **Deadlock Protection as an export gate.** Dirty inputs and attempts to emit an unbacked
  certificate exit 3 and append to `.sra_deadlock`; exploration inside a search rolls back
  silently. Built on `formalguard`, so the exception class and trace schema are shared.
- **A surrogate that does not trust itself.** A five-point fit scoring a leave-one-out R2 of
  0.99996 was still wrong by 4x four octaves out, so the gate refuses extrapolation beyond 2x
  regardless of fit quality.

Install with `pip install "sampling-rate-audit[opt]"`; plain `pip install sampling-rate-audit`
still gives the engine with numpy as its only dependency.

Two limits the demo reports rather than hides: the expensive track was closer in only 2 of 9
cases, and the arbitration model has nine training cases, so it exposes disagreement and does
not settle it.

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
