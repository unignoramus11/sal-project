"""Figures for the report and slides, generated only from results/ (PDF + PNG in results/figures/).

fig_probes      Recall@1 (seed ensemble, 95% speaker-bootstrap CI) for every probe family under PRiSM's random
                split (blue circles) and the speaker-disjoint split (orange squares); dashed line: majority
                baseline; open marker: same probe scored only on shared-picture test utterances.
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
    t = pd.read_csv(RESULTS / "tables" / "configs.csv")
    t[["label", "proto", "generic"]] = t["config"].apply(lambda c: pd.Series(family(c) or (None, None, None)))
    t = t.dropna(subset=["label"])
    main = t[~t["generic"].astype(bool)]
    labels = sorted(main["label"].unique(), key=lambda l: -main[main.label == l]["ens_recall1"].max())
    base = pd.read_csv(RESULTS / "baselines.csv")
    maj = base[(base.baseline == "majority") & (base.split == "test")].set_index("protocol")["recall1"]
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(labels) + 0.7))
    for i, lab in enumerate(labels):
        for proto, color, marker, dy in (("prism", BLUE, "o", -0.14), ("speaker_split", ORANGE, "s", 0.14)):
            r = main[(main.label == lab) & (main.proto == proto)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.ens_recall1, i + dy, xerr=[[r.ens_recall1 - r.ens_recall1_lo], [r.ens_recall1_hi - r.ens_recall1]],
                        fmt=marker, ms=4, color=color, ecolor=color, elinewidth=1, capsize=0, zorder=3)
            g = t[(t.label == lab) & (t.proto == proto) & (t["generic"].astype(bool))]
            if not g.empty:
                ax.plot(g.iloc[0].ens_recall1, i + dy, marker, ms=4, mfc="white", mec=color, mew=1, zorder=4)
            elif not np.isnan(r.get("generic_recall1", np.nan)):
                ax.plot(r.generic_recall1, i + dy, marker, ms=4, mfc="white", mec=color, mew=1, zorder=4)
    ax.axvline(maj.mean(), color=INK2, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -1.1)           # inverted, with headroom for the baseline label
    ax.text(maj.mean() + 0.6, -0.75, "majority baseline", color=INK2, fontsize=6.5, va="center")
    ax.set_xlabel("Test Recall@1 (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], marker="o", color=BLUE, ls="", ms=4, label="random split (PRiSM)"),
         plt.Line2D([], [], marker="s", color=ORANGE, ls="", ms=4, label="speaker-disjoint"),
         plt.Line2D([], [], marker="o", mfc="white", mec=INK2, ls="", ms=4, label="shared pictures only")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
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
