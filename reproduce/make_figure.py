"""The manuscript's figure, from the committed result files.

The figure puts the two gaps side by side on one axis, per dataset and
property: what a trained model buys over a lookup that fits nothing to the
target, under a random partition, and what excluding whole libraries from
training costs that same model. It is 3.45 inches wide, which fits the A4 page
width that the journal asks figures to fit.

Rendered with a fixed creation date so a regeneration is byte-identical to the
committed PDF.

    python reproduce/make_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURE = ROOT / "paper" / "figures" / "split_vs_model.pdf"

sys.path.insert(0, str(ROOT / "reproduce"))
from make_parameters import r2, resolved  # noqa: E402

# Okabe-Ito, which separates for the common colour-vision deficiencies. The two
# series also differ in marker shape and fill, and in the supplementary panel in line
# style, so both survive greyscale.
MODEL_GAP = "#0072B2"
PARTITION_COST = "#D55E00"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((OUTPUTS / name).read_text())


def main() -> None:
    npj = load("npj/split.json")
    htem = load("htem/split.json")
    margins = load("margins/results.json")
    expt = load("expt_gap/control.json")
    aihub = margins["datasets"]["aihub"]["tasks"]
    hres = htem["targets"]["log10_resistivity"]
    hgap = htem["targets"]["direct_band_gap"]

    # (label, random lookup, random model, blocked model or None, the blocked
    # score's interval or None). The interval sets the places a printed blocked
    # score takes, as tab:main prints it (make_parameters.py, PRECISION).
    rows = [
        ("Kim & Lee 2025\nresistivity",
         npj["lookup"]["all_features"]["r2"], npj["model"]["val"]["r2"], None, None),
        ("EMPD\nresistivity", aihub["resistivity"]["lookup_r2"]["mean"],
         aihub["resistivity"]["model_r2"]["mean"],
         aihub["resistivity"]["blocked_model_r2"]["mean"],
         aihub["resistivity"]["blocked_model_r2"]),
        ("EMPD\nhardness", aihub["hardness"]["lookup_r2"]["mean"],
         aihub["hardness"]["model_r2"]["mean"],
         aihub["hardness"]["blocked_model_r2"]["mean"],
         aihub["hardness"]["blocked_model_r2"]),
        ("HTEM-DB\nresistivity", hres["random"]["lookup_r2"],
         hres["random"]["model_r2"], hres["blocked"]["model_r2"],
         hres["ci"]["blocked_model_r2"]),
        ("HTEM-DB\nband gap", hgap["random"]["lookup_r2"],
         hgap["random"]["model_r2"], hgap["blocked"]["model_r2"],
         hgap["ci"]["blocked_model_r2"]),
        ("Matbench\nband gap", expt["random"]["lookup_r2"],
         expt["random"]["model_r2"], None, None),
    ]

    figure, left = plt.subplots(figsize=(3.45, 2.95))
    x_left, x_right = -2.15, 1.08

    # Two lanes per dataset so the short bar is legible beside the long one: the
    # model's margin above, the partition's cost below.
    offset = 0.19
    for y, (_, lookup, model, blocked, blocked_ci) in enumerate(rows):
        if blocked is not None:
            left.plot([max(blocked, x_left), model], [y + offset, y + offset],
                      color=PARTITION_COST, linewidth=4.5, solid_capstyle="butt",
                      zorder=2)
            if blocked < x_left:
                # The axis cuts the bar, so an arrow at the cut and the score
                # printed beside it stand in for the square (fig:split).
                left.plot([x_left], [y + offset], marker="<", markersize=6,
                          color=PARTITION_COST, zorder=4, clip_on=False)
                left.annotate(r2(blocked, resolved(blocked_ci)).replace("$-$", "\N{MINUS SIGN}"),
                              (x_left, y + offset), xytext=(6, -2),
                              textcoords="offset points", ha="left", va="top",
                              fontsize=7, color=PARTITION_COST)
            else:
                # Open, because a filled square is the bar's own colour at the
                # bar's own width and disappears into it.
                left.plot([blocked], [y + offset], marker="s", markersize=4.5,
                          color=PARTITION_COST, markerfacecolor="white", zorder=4)
        left.plot([lookup, model], [y - offset, y - offset], color=MODEL_GAP,
                  linewidth=4.5, solid_capstyle="butt", zorder=2)
        left.plot([lookup], [y - offset], marker="o", markersize=4.5,
                  color=MODEL_GAP, markerfacecolor="white", zorder=4)
        left.plot([model], [y - offset], marker="o", markersize=4.5,
                  color=MODEL_GAP, zorder=4)

    left.axvline(0.0, color="0.4", linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    left.set_yticks(range(len(rows)))
    left.set_yticklabels([label for label, *_ in rows])
    left.set_ylim(len(rows) + 0.5, -0.7)
    left.set_xlabel("$R^2$ on test rows")
    left.set_xlim(x_left, x_right)
    handles = [plt.Line2D([], [], color=MODEL_GAP, linewidth=4.5,
                          label="lookup $\\rightarrow$ model, same partition"),
               plt.Line2D([], [], color=PARTITION_COST, linewidth=4.5,
                          label="random $\\rightarrow$ blocked, same model")]
    left.legend(handles=handles, loc="lower left", frameon=False,
                borderaxespad=0.15, handlelength=1.5, labelspacing=0.35,
                fontsize=7)

    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, metadata={"CreationDate": None})
    print(f"wrote {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
