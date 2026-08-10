# catatonia-two-axes

Reproduction code for

> **Catatonia recovers along two separable intervention axes that constrain electroconvulsive therapy**
> H. Saito

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![reproduce](https://github.com/USERNAME/catatonia-two-axes/actions/workflows/reproduce.yml/badge.svg)](https://github.com/USERNAME/catatonia-two-axes/actions/workflows/reproduce.yml)

Everything is in one file, `saito_two_axes_reproduce.py`: the model, every
analysis reported in the paper, the figures, the source-data workbook, a
value-by-value verification against the stored results, a submission-compliance
audit, and the full text of the manuscript and the Supplementary Information
together with the code that renders them to Word.

## Install

```bash
git clone https://github.com/USERNAME/catatonia-two-axes.git
cd catatonia-two-axes
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

Python 3.9 or newer. Dependencies are `numpy`, `scipy`, `matplotlib` and
`openpyxl`; nothing else, and no compiled extensions.

## Run

```bash
python saito_two_axes_reproduce.py                 # full analysis, ~15 min on one core
python saito_two_axes_reproduce.py --quick         # coarser grids, ~3 min
python saito_two_axes_reproduce.py --figures       # redraw from an existing results.json
python saito_two_axes_reproduce.py --verify        # re-check every number the paper prints
python saito_two_axes_reproduce.py --audit         # Nature Communications compliance
python saito_two_axes_reproduce.py --docx T.docx   # write the manuscript and the SI
python saito_two_axes_reproduce.py --outdir out    # put everything in ./out
```

The flags compose, and a later step produces whatever it needs if it is
missing, so a single command does a full pass:

```bash
python saito_two_axes_reproduce.py --figures --verify --audit --outdir out
```

In a notebook the command line is ignored, because the kernel appends its own
arguments. Paste the file into a cell and call the helper instead:

```python
run(quick=True)      # or run(), run(verify=True), run(figures=True)
show_figures()
```

## Outputs

| File | Contents |
|---|---|
| `results.json` | every number the manuscript reports, plus the trajectories the figures are drawn from |
| `Fig1.pdf` … `Fig4.pdf` (and `.png`) | the four display items, at the sizes and palette used in the paper |
| `Source_Data_Figs1-4.xlsx` | one sheet per figure panel, the values as plotted, plus a `Read me` sheet listing the parameters |
| `staircase.npy` | the Fig. 4d trajectory at full resolution |
| `Saito_two_axes_NatComms.docx`, `Saito_two_axes_SupplementaryInformation.docx` | written only with `--docx` |

`--docx` needs a template `.docx` for its page setup, styles and footers. None
of the template's text is used, and no template is shipped here; any Word
document whose page geometry you want to reuse will do.

## What `--verify` checks

`--verify` re-reads `results.json` and compares it against every value quoted in
the manuscript, one line per check. A full run gives **116 checks, 0 failed**; a
`--quick` run reports the same for the grid-independent subset and states how
many grid-size-dependent checks it skipped.

Values are compared at the precision the manuscript prints, not at a longer
internal precision, so a number cannot pass the check while the text is wrong by
a rounding step.

Two structural checks are also run: every source-data sheet corresponds to a
panel the figure legends declare, and every panel that carries data has a sheet.

## What `--audit` checks

The formal Nature Communications rules for an Article: title length and
punctuation, abstract length and absence of citations, main-text length,
subheading length, number of display items, absence of em dashes, references
numbered in order of first appearance with every entry cited and locatable,
figures and panels cited in order, cross-references to the Supplementary Notes,
and section order.

The audit exits non-zero while any identifier in the text is still a
placeholder, which is deliberate: it is the last thing to fix before submission.

## Reproducibility notes

- All randomness is drawn from a single seeded generator (`SEED` near the top of
  the file), so a full run is bit-for-bit repeatable on the same NumPy version.
- Integration uses an adaptive stiff solver at relative tolerance 1e-10 and
  absolute tolerance 1e-13, with event detection for crossings.
- Step 5 (intermittent modulation) is the slowest, roughly half the total, and
  prints per-cell progress so a long run is not mistaken for a hang.
- Parameters are the unfitted illustrative set of the model this work builds on:
  `a0 = 0.60`, `a1 = 0.15`, `p = 2`, `eps = 0.02`, `rho = 1`, `kappa = 0.6`.

## Citation

If you use this code, please cite the software (Zenodo DOI above) and the paper.
`CITATION.cff` is machine-readable and GitHub renders it as a "Cite this
repository" button.

## License

MIT, see [`LICENSE`](LICENSE).
