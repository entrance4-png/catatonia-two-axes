# Changelog
## v3.0.0 — 2026-10-20
- Removed the preprint manuscript text, its journal-compliance audit and the docx renderer from the script; the journal manuscript is maintained in manuscript/.
- Added section 13 (run_review_additions): Remark 2 initial-condition example, Proposition 8(ii) passage-time constant, Proposition 9 exact two-stage inversion and its sensitivity, each self-checked (84 checks in total).
- Added make_jmb_figures.py (EPS/TIFF/PNG at journal width), manuscript/ and figures/.
- argparse now uses parse_known_args so the script runs inside a notebook cell.
- All earlier computations, figures and checks unchanged.

## v1.0.1 (2026-08-11)

- Recorded the Zenodo concept DOI 10.5281/zenodo.21880298 and the repository URL
  in the README, the citation file and the manuscript's Code availability.
- Corrected the Zenodo metadata: removed a placeholder ORCID that failed the
  ISO 7064 checksum and caused the v1.0.0 release to be rejected on archiving,
  and set the license identifier to the lowercase vocabulary id.
- Recorded the author ORCID in `.zenodo.json` and `CITATION.cff`.
- Manuscript title updated.

## v1.0.0 (2026-08-10)

- First public release: model, all analyses, figures, source data,
  value-by-value verification, submission-compliance audit, and the manuscript
  and Supplementary Information with the code that renders them to Word.
