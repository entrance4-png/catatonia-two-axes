#!/usr/bin/env python3
"""Write Fig1 ... Fig5 in the formats Journal of Mathematical Biology asks for.

Run in the directory that holds catatonia_two_axes_reproduce.py and a
two_axes_results.json produced by a full run of that script:

    python3 make_jmb_figures.py                 # 174 mm wide, EPS + TIFF (600 dpi)
    python3 make_jmb_figures.py --width-mm 84   # single-column width

The figure functions of the reproduction script are reused unchanged.  Only the
figure size, the lettering and the output format are set here: each figure is
drawn at its final printed width so that the lettering is 8-9 pt on the page
(JMB asks for Helvetica/Arial at 8-12 pt; Liberation Sans is the metric-
compatible open substitute and is embedded in the EPS as a Type 42 font).
Panel labels are lower case, as JMB requires.
"""
import argparse, json, importlib.util, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--script", default="catatonia_two_axes_reproduce.py")
ap.add_argument("--results", default="two_axes_results.json")
ap.add_argument("--width-mm", type=float, default=174.0)
ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--font", default="Liberation Sans")
ap.add_argument("--formats", default="eps,tiff,png")
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("repro", args.script)
repro = importlib.util.module_from_spec(spec)
sys.argv = [args.script]
spec.loader.exec_module(repro)

with open(args.results) as fh:
    R = json.load(fh)

NATIVE_W = 9.6                                   # inches, as in the script
scale = (args.width_mm / 25.4) / NATIVE_W        # final / native

plt.rcParams.update({
    "font.family": args.font, "mathtext.fontset": "custom",
    "mathtext.rm": args.font, "mathtext.it": args.font + ":italic",
    "mathtext.bf": args.font + ":bold",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.2, "axes.linewidth": 0.6,
    "savefig.dpi": args.dpi, "figure.dpi": 150,
    "ps.fonttype": 42, "pdf.fonttype": 42,
})

_figure = plt.figure
def figure(*a, **k):
    if "figsize" in k:
        w, h = k["figsize"]; k["figsize"] = (w * scale, h * scale)
    return _figure(*a, **k)
repro.plt.figure = figure

def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s.lower(), transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")
repro.panel_label = panel_label

fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
for k, fn in enumerate((repro.fig1, repro.fig2, repro.fig3, repro.fig4, repro.fig5), 1):
    for ext in fmts:
        fn(R, path=f"Fig{k}.{ext}")
        plt.close("all")
    print("wrote", ", ".join(f"Fig{k}.{e}" for e in fmts))
print(f"width {args.width_mm} mm (scale {scale:.2f} of the native 9.6 in); lettering 8-8.5 pt")
