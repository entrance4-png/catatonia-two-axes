# catatonia-two-axes

Reproduction code and figures for

> Saito H. *Catatonia recovery separates two intervention axes that constrain
> electroconvulsive therapy.* Submitted to the Journal of Mathematical Biology
> (2026). Preprint: PsyArXiv d6knq. Code: Zenodo concept DOI
> 10.5281/zenodo.21880298 (resolves to the latest version).

The model is the evidence-coupled bistable cascade of Saito (2026),
*Evidence-coupled bistability converts dependency structure into a recovery
geometry* (Research Square rs-10620384), with the illustrative parameters used
there: a0 = 0.60, a1 = 0.15, p = 2, eps = 0.02, rho = 1, kappa = 0.6.

## Contents

| File | What it is |
|---|---|
| `catatonia_two_axes_reproduce.py` | One self-contained script (v3.1): every reported value, the self-check of each against the printed value (84 checks), the five figures, the source-data workbook and the secondary analysis of published aggregate data. No inputs. |
| `make_jmb_figures.py` | Redraws Fig1–Fig5 from `two_axes_results.json` in the journal's formats (EPS with embedded fonts, TIFF at 600 dpi, PNG) at a stated printed width, with lower-case panel labels. |
| `figures/` | Fig1–Fig5 as EPS, TIFF (600 dpi) and PNG (300 dpi), 174 mm wide. |

## Running

```
pip install numpy scipy matplotlib openpyxl python-docx
python3 catatonia_two_axes_reproduce.py            # full run, ~10–20 min
python3 catatonia_two_axes_reproduce.py --quick    # reduced sample sizes, same structure
python3 make_jmb_figures.py                        # journal figure files from two_axes_results.json
```

The full run prints a self-check table of 84 lines; every line must read `OK`.
The `--quick` run reduces sample sizes and may miss one or two of the
regression-based checks by design.

## Versions

**v3.1** (this release) corrects two transcription errors in the published
aggregate data of `run_published_data` (Fig. 5), found by checking the tables
against Bot et al. (2026): the remission rate of Lee 2000 is 67 per cent, not 0
(Table 1 of the source), and the post-treatment BFCRS of Thamizh 2016 case 2 is
0, not 10 (Table 2 of the source). The depth-on-dose statistics of Fig. 5a change
(slope on log dose -0.04, P = 0.77; Spearman rho -0.06, P = 0.79; previously
-0.14, 0.35, -0.15, 0.50) and remain null; the per-patient reductions of Fig. 5b
become 0, 23, 81, 85 and six values of 100 per cent, the empty interval 23-81
and the binomial P = 0.037 are unchanged. The four self-checks on these
statistics carry the corrected printed values. Nothing else changes.

**v3** removed from the script the preprint manuscript text, its
journal-compliance audit (abstract length, numbered citations) and the docx
renderer, which belonged to the preprint and no longer match the journal
manuscript. The script now reproduces numbers, figures and source data only;
the manuscript and its Online Resources are held by the journal and are not
kept in this repository. Computations and checks are unchanged
from v2.

**v2** added section 13 of the script (`run_review_additions`) and its
self-checks, the only computational change from the preprint version v1:

* **Remark 2 / Theorem 3.** At a = 0.60 and kappa = 0.01 (subcritical), a first
  stage started above its unstable rest point r+ = 0.5828 recovers while the
  second stage arrests; started below r+, or from the collapsed configuration,
  nothing recovers. Theorem 3 is stated for the collapsed downstream start.
* **Proposition 8(ii).** Passage time of the first stage through the fold,
  compared with pi / ((1 − a/2) sqrt(delta)): ratio 0.916, 0.973, 0.992 at
  delta = 1e-4, 1e-5, 1e-6, with a constant O(1) difference.
* **Proposition 9.** The exact two-stage inversion of the threshold from the
  first two resting levels (equal coupling), its recovery of a = 0.15 to
  fourteen digits, and its sensitivity to 1 % errors in r1 and r2.

All earlier computations, figures and checks are unchanged. The sheet names in
the source-data workbook written by the script (Online Resource 2, `ESM_2.xlsx`)
use the preprint's upper-case panel letters; the journal manuscript uses lower
case.

## Licence

MIT. See `LICENSE`.
