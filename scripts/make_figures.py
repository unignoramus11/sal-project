"""Figures for the report and slides, generated only from results/ (PDF + PNG in results/figures/).

fig_probes      per probe family (seed ensemble, 95% speaker-bootstrap CI), PRiSM's random split (blue circles)
                and the speaker-disjoint split (orange squares): Recall@1 above the majority baseline and km
                below the best constant location, on the same utterances; open marker: shared pictures only.
fig_dpr_<model> doubled-phone rate by state with 95% speaker-bootstrap CIs, Haryana labelled.
Run: uv run python scripts/make_figures.py
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sal.paths import RESULTS

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e3e0"
out = RESULTS / "figures"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK})

PRETTY = {
    "dense_ecapa_rebuild": "Speaker embedding (ECAPA)",
    "phone_unigram": "TP phone unigrams", "char_unigram": "TP char unigrams",
    "phone_bigram": "TP phone bigrams", "char_bigram": "TP char bigrams",
    "char_gru": "TP-GRU, characters (PRiSM)", "phone_gru": "TP-GRU, phones", "phone_shuffle": "TP-GRU, shuffled phones",
}


def family(cfg: str) -> tuple[str, str, bool] | None:
    generic = cfg.endswith("_generic")
    c = cfg.removesuffix("_generic")
    m = re.match(r"(.+)_(prism|speaker_split)$", c)
    if not m:
        return None
    base, proto = m.groups()
    label = base
    tp = re.match(r"tp_(\w+?)_rebuild_(\w+)$", base)
    if tp:
        label = f"{PRETTY.get(tp.group(2), tp.group(2))} [{tp.group(1).upper()}]"
    elif base in PRETTY:
        label = PRETTY[base]
    elif base.startswith("rp_"):
        label = base.replace("rp_", "RP ").replace("_rebuild", "")
    elif base.startswith("prosody"):
        label = "Prosody BiGRU"
    return label, proto, generic


def fig_probes():
    """Two panels on a common reference, so the full test set and the shared-picture subset (whose location prior
    differs) are comparable: Recall@1 above the majority baseline, and km below the best constant location, both
    on the same test utterances (seed ensemble, 95% speaker-bootstrap CI)."""
    t = pd.read_csv(RESULTS / "tables" / "configs.csv")
    t[["label", "proto", "generic"]] = t["config"].apply(lambda c: pd.Series(family(c) or (None, None, None)))
    t = t.dropna(subset=["label"])
    main = t[~t["generic"].astype(bool)]
    labels = sorted(main["label"].unique(), key=lambda l: -main[main.label == l]["lift_recall1"].max())
    panels = [("lift_recall1", "lift_recall1_lo", "lift_recall1_hi", "generic_lift_recall1",
               "Recall@1 above majority (points)"),
              ("km_gain", "km_gain_lo", "km_gain_hi", "generic_km_gain", "km better than best constant")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 0.28 * len(labels) + 0.8), sharey=True)
    for ax, (col, lo, hi, gcol, xlabel) in zip(axes, panels):
        for i, lab in enumerate(labels):
            for proto, color, marker, dy in (("prism", BLUE, "o", -0.14), ("speaker_split", ORANGE, "s", 0.14)):
                r = main[(main.label == lab) & (main.proto == proto)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r[col], i + dy, xerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt=marker, ms=4, color=color,
                            ecolor=color, elinewidth=1, capsize=0, zorder=3)
                ax.plot(r[gcol], i + dy, marker, ms=4, mfc="white", mec=color, mew=1, zorder=4)
        ax.axvline(0, color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.6)
    h = [plt.Line2D([], [], marker="o", color=BLUE, ls="", ms=4, label="random split (PRiSM)"),
         plt.Line2D([], [], marker="s", color=ORANGE, ls="", ms=4, label="speaker-disjoint"),
         plt.Line2D([], [], marker="o", mfc="white", mec=INK2, ls="", ms=4, label="scored on shared pictures only")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_probes.{ext}", dpi=220)
    plt.close(fig)


def fig_dpr():
    for f in sorted((RESULTS / "dpr").glob("*_by_state.csv")):
        model = f.stem.replace("_by_state", "")
        d = pd.read_csv(f).sort_values("dpr")
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        y = np.arange(len(d))
        ax.errorbar(100 * d.dpr, y, xerr=[100 * (d.dpr - d.lo), 100 * (d.hi - d.dpr)], fmt="o", ms=4, color=BLUE,
                    elinewidth=1, capsize=0)
        ax.set_yticks(y, [f"{s} ({n})" for s, n in zip(d.state, d.speakers)])
        for tick, s in zip(ax.get_yticklabels(), d.state):
            if s == "Haryana":
                tick.set_fontweight("bold")
        ax.set_xlabel(f"Doubled-phone rate, % of consonants ({model.upper()})")
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.3)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_dpr_{model}.{ext}", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    fig_probes()
    fig_dpr()
    print(sorted(p.name for p in out.iterdir()))
