# Changelog

## v3.1.1 — 2026-10-28
- Removed the manuscript/ directory. The journal manuscript, its Online Resources and the cover letter are held by the journal and are no longer kept in the repository; the README no longer lists them.
- Code, figures, source data and all 84 self-checks unchanged from v3.1.0.

## v3.1.0 — 2026-10-20
- Corrected two transcription errors in the published aggregate data of run_published_data (Fig. 5), found by checking against Bot et al. (2026) and the original report of Thamizh et al. (2016): Lee 2000 remission 67 per cent, not 0 (Bot Table 1); Thamizh 2016 case 2 BFCRS after treatment 0, not 10 (Bot Table 2; the earlier value was the lorazepam dose transcribed into the score column).
- Depth-on-dose statistics of Fig. 5a change and remain null (slope on log dose -0.04, P = 0.77; Spearman rho -0.06, P = 0.79; previously -0.14, 0.35, -0.15, 0.50). Per-patient reductions of Fig. 5b are now 0, 23, 81, 85 and six values of 100 per cent; the empty interval 23-81 and binomial P = 0.037 are unchanged. The four self-checks on these statistics carry the corrected printed values (84/84 pass).
- make_jmb_figures.py: fixed 2-5-10 tick labels on log axes at the reduced printed width.
- Online Resource 1: wording on the zero-evidence-rate exponent control revised; no numbers changed.

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
