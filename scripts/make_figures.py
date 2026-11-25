"""Figures for the report and slides, generated only from results/ (PDF + PNG in results/figures/).

fig_probes      per probe family (seed ensemble, 95% speaker-bootstrap CI), PRiSM's random split (blue circles)
                and the speaker-disjoint split (orange squares): Recall@1 above the majority baseline and km
                below the best constant location, on the same utterances; open marker: trained and tested on
                shared pictures only, against that subset's own baselines.
fig_dpr_<model> doubled-phone rate by state with 95% speaker-bootstrap CIs, Haryana labelled.
fig_lookups    rebuilt Vaani-Hi: test Recall@1 of lookups and probes, with test speakers in training (PRiSM's split)
               and on new speakers; drawn at full page width for the report.
fig_order      Recall@1 gain from phone order (BiGRU vs shuffled BiGRU, bigram vs unigram bags) with seen and new
               speakers, on the release and the rebuild, paired 95% CIs; drawn at column width.
fig_benchmark  cross-validated benchmark: Recall@1 lift over the majority with 95% CIs per probe, all pictures
               (filled) and shared pictures only (hollow); every test speaker is new; column width.
Run: uv run python scripts/make_figures.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")   # fixed PDF creation date: same figure, same bytes
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
    "dense_ecapa_rebuild": "Speaker embedding (ECAPA)", "dense_prosody_rebuild": "Prosody statistics (pooled)",
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
    labels = sorted(main["label"].unique(), key=lambda x: -main[main.label == x]["lift_recall1"].max())
    panels = [("lift_recall1", "lift_recall1_lo", "lift_recall1_hi", "generic_lift_recall1",
               "Recall@1 above majority (points)"),
              ("km_gain", "km_gain_lo", "km_gain_hi", "generic_km_gain", "km better than best constant")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 0.28 * len(labels) + 0.8), sharey=True)
    for ax, (col, lo, hi, _, xlabel) in zip(axes, panels, strict=True):
        for i, lab in enumerate(labels):
            for proto, color, marker, dy in (("prism", BLUE, "o", -0.14), ("speaker_split", ORANGE, "s", 0.14)):
                r = main[(main.label == lab) & (main.proto == proto)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r[col], i + dy, xerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt=marker, ms=4, color=color,
                            ecolor=color, elinewidth=1, capsize=0, zorder=3)
                g = t[(t.label == lab) & (t.proto == proto) & t["generic"].astype(bool)]
                if not g.empty:                    # trained and tested on shared pictures, own baselines
                    ax.plot(g.iloc[0][col], i + dy, marker, ms=4, mfc="white", mec=color, mew=1, zorder=4)
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
         plt.Line2D([], [], marker="o", mfc="white", mec=INK2, ls="", ms=4, label="trained and tested on shared pictures")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_probes.{ext}", dpi=220, transparent=True)
    plt.close(fig)


def fig_dpr():
    for f in sorted((RESULTS / "dpr").glob("*_by_state.csv")):
        model = f.stem.replace("_by_state", "")
        d = pd.read_csv(f).sort_values("dpr")
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        y = np.arange(len(d))
        ax.errorbar(100 * d.dpr, y, xerr=[100 * (d.dpr - d.lo), 100 * (d.hi - d.dpr)], fmt="o", ms=4, color=BLUE,
                    elinewidth=1, capsize=0)
        ax.set_yticks(y, [f"{s} ({n})" for s, n in zip(d.state, d.speakers, strict=True)])
        for tick, s in zip(ax.get_yticklabels(), d.state, strict=True):
            if s == "Haryana":
                tick.set_fontweight("bold")
        ax.set_xlabel(f"Doubled-phone rate, % of consonants ({model.upper()})")
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.3)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_dpr_{model}.{ext}", dpi=220, transparent=True)
        plt.close(fig)


DARK, LIGHT, LINK = "#12352a", "#5c8f72", "#b7cbbe"   # ordinal pair, validated on white: seen / new speakers


def _clean(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _legend(ax, labels):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=5, color=c, label=lab) for c, lab in zip((DARK, LIGHT), labels, strict=True)]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=1.2, borderaxespad=0.2)


def fig_lookups():
    cfg = pd.read_csv(RESULTS / "tables" / "configs.csv").set_index("config")
    base = pd.read_csv(RESULTS / "baselines.csv")
    knn = pd.read_csv(RESULTS / "tables" / "knn_transcripts.csv").set_index(["transcripts", "protocol", "k"])
    pic = pd.read_csv(RESULTS / "rebuild_checks" / "picture_lookup.csv").set_index(["prompts", "protocol", "baseline"])

    def bl(name, proto):
        return base[(base.baseline == name) & (base.protocol == proto) & (base.prompts == "all") & (base.split == "test")].recall1.iloc[0]
    rows = [("Speaker-ID lookup", bl("speaker_lookup", "prism"), bl("speaker_lookup", "speaker_split")),
            ("Picture-ID lookup", pic.loc[("all", "prism", "picture lookup"), "recall1"],
             pic.loc[("all", "speaker_split", "picture lookup"), "recall1"]),
            ("ECAPA voice embedding", cfg.loc["dense_ecapa_rebuild_prism", "recall1_mean"],
             cfg.loc["dense_ecapa_rebuild_speaker_split", "recall1_mean"]),
            ("Transcript match (W2V2)", knn.loc[("w2v2", "prism", 1), "recall1"], knn.loc[("w2v2", "speaker_split", 1), "recall1"]),
            ("ZIPA phone bigrams", cfg.loc["tp_zipa_rebuild_phone_bigram_prism", "recall1_mean"],
             cfg.loc["tp_zipa_rebuild_phone_bigram_speaker_split", "recall1_mean"]),
            ("PRiSM's character BiGRU (W2V2)", cfg.loc["tp_w2v2_rebuild_char_gru_prism", "recall1_mean"],
             cfg.loc["tp_w2v2_rebuild_char_gru_speaker_split", "recall1_mean"]),
            ("Majority location", bl("majority", "prism"), bl("majority", "speaker_split"))]
    fig, ax = plt.subplots(figsize=(6.3, 2.15))
    y = np.arange(len(rows))[::-1]
    for yi, (_, a, b) in zip(y, rows, strict=True):
        ax.plot([min(a, b), max(a, b)], [yi, yi], color=LINK, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([a], [yi], s=26, color=DARK, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([b], [yi], s=26, color=LIGHT, zorder=3, edgecolor="white", linewidth=0.8)
        off = 1.6 + max(0.0, 3.0 - abs(a - b)) / 2
        ax.text(max(a, b) + off, yi, f"{max(a, b):.1f}", va="center", ha="left", color=INK)
        ax.text(min(a, b) - off, yi, f"{min(a, b):.1f}", va="center", ha="right", color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Test Recall@1 on rebuilt Vaani-Hi (%)")
    _clean(ax)
    _legend(ax, ("test speakers also in training (PRiSM's split)", "new speakers (speaker-disjoint split)"))
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_lookups.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_order():
    comp = pd.read_csv(RESULTS / "tables" / "comparisons.csv").set_index(["a", "b"])
    oo = pd.read_csv(RESULTS / "official" / "order_clean.csv").set_index(["protocol", "a"])

    def cp(a, b):
        r = comp.loc[(a, b)]
        return r.recall1_diff, r.recall1_lo, r.recall1_hi

    def op(P, a):
        r = oo.loc[(P, a)]
        return r["diff"], r.lo, r.hi
    rows = [("BiGRU vs shuffled\nrelease, W2V2", op("prism", "tp_w2v2_official_phone_gru"), op("speaker_split", "tp_w2v2_official_phone_gru"))]
    for m in ("w2v2", "zipa"):
        rows.append((f"BiGRU vs shuffled\nrebuild, {m.upper()}", cp(f"tp_{m}_rebuild_phone_gru_prism", f"tp_{m}_rebuild_phone_shuffle_prism"),
                     cp(f"tp_{m}_rebuild_phone_gru_speaker_split", f"tp_{m}_rebuild_phone_shuffle_speaker_split")))
    rows.append(("Bigram vs unigram\nrelease, W2V2", op("prism", "tp_w2v2_official_phone_bigram"),
                 op("speaker_split", "tp_w2v2_official_phone_bigram")))
    for m in ("w2v2", "zipa"):
        rows.append((f"Bigram vs unigram\nrebuild, {m.upper()}", cp(f"tp_{m}_rebuild_phone_bigram_prism", f"tp_{m}_rebuild_phone_unigram_prism"),
                     cp(f"tp_{m}_rebuild_phone_bigram_speaker_split", f"tp_{m}_rebuild_phone_unigram_speaker_split")))
    fig, ax = plt.subplots(figsize=(3.15, 3.2))
    y = np.arange(len(rows))[::-1] * 1.0
    for yi, (_, a, b) in zip(y, rows, strict=True):
        for (e, lo, hi), c, dy in ((a, DARK, 0.17), (b, LIGHT, -0.17)):
            ax.plot([lo, hi], [yi + dy] * 2, color=c, lw=1.4, solid_capstyle="round", zorder=2)
            ax.scatter([e], [yi + dy], s=16, color=c, zorder=3, edgecolor="white", linewidth=0.6)
            lab = f"{e:+.1f}".replace("-", "\u2212")
            if hi < 0.5:
                ax.text(lo - 0.6, yi + dy, lab, va="center", ha="right", color=INK2, fontsize=7)
            else:
                ax.text(hi + 0.6, yi + dy, lab, va="center", ha="left", color=INK2, fontsize=7)
    ax.axvline(0, color=INK2, lw=0.6)
    los = [v[1] for r in rows for v in r[1:]]
    his = [v[2] for r in rows for v in r[1:]]
    ax.set_xlim(min(los) - 4, max(his) + 4)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Recall@1 gain from phone order (points)")
    _clean(ax)
    _legend(ax, ("seen speakers", "new speakers"))
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_order.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_benchmark():
    """Cross-validated benchmark: Recall@1 lift over the majority (95% CI over speaker groups) per probe; every test
    speaker is new, so all marks use the new-speaker colour; filled = all pictures, hollow = shared pictures only."""
    f = RESULTS / "benchmark" / "scores.csv"
    if not f.exists():
        return
    sc = pd.read_csv(f).set_index(["track", "probe"])
    groups = [[("Picture-ID lookup", "picture lookup"), ("Transcript match, W2V2", "1-NN transcript (W2V2)"),
               ("Transcript match, ZIPA", "1-NN transcript (ZIPA)")],
              [("ECAPA voice", "dense_ecapa_rebuild"), ("Prosody statistics", "dense_prosody_rebuild")]]
    for rec in ("W2V2", "ZIPA"):
        groups.append([(f"{rec} {lab}", f"tp_{rec.lower()}_rebuild_{k}") for lab, k in
                       [("phone unigram", "phone_unigram"), ("phone bigram", "phone_bigram"), ("char BiGRU", "char_gru"),
                        ("phone BiGRU", "phone_gru"), ("shuffled BiGRU", "phone_shuffle")]])
    rp = sorted({k for _, k in sc.index if str(k).startswith("rp_")})
    if rp:
        names = {"attn": "attention", "gru": "BiGRU", "chunkshuffle": "chunk-shuffled", "attn_pe": "attn + pos.",
                 "transformer": "Transformer"}
        groups.append([(f"RP {k.split('_rebuild_')[0][3:].upper()} {names.get(k.split('_rebuild_')[1].split('_', 1)[1], k)}", k)
                       for k in rp])
    groups = [[(lab, k) for lab, k in g if ("all", k) in sc.index] for g in groups]
    groups = [g for g in groups if g]
    rows, ys, y, seps = [], [], 0.0, []
    for g in groups:
        for lab, k in g:
            rows.append((lab, k))
            ys.append(y)
            y -= 1
        seps.append(y + 0.5)
        y -= 0.4
    fig, ax = plt.subplots(figsize=(3.15, 0.2 * len(rows) + 0.75))
    for yi, (_, k) in zip(ys, rows, strict=True):
        for track, dy, face in (("all", 0.17, LIGHT), ("shared", -0.17, "white")):
            if (track, k) not in sc.index:
                continue
            r = sc.loc[(track, k)]
            ax.plot([r.recall1_vs_majority_lo, r.recall1_vs_majority_hi], [yi + dy] * 2, color=LIGHT, lw=1.3,
                    solid_capstyle="round", zorder=2)
            ax.scatter([r.recall1_vs_majority], [yi + dy], s=18, facecolor=face, edgecolor=LIGHT, linewidth=1.0, zorder=3)
    for sy in seps[:-1]:
        ax.axhline(sy - 0.2, color=GRID, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(min(ys) - 0.7, 0.7)
    ax.set_xlabel("Recall@1 above the majority location (points)")
    _clean(ax)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=5, markerfacecolor=fc, markeredgecolor=LIGHT, label=lab)
         for fc, lab in ((LIGHT, "all pictures"), ("white", "shared pictures only"))]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_benchmark.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_probes()
    fig_dpr()
    fig_lookups()
    fig_order()
    fig_benchmark()
    print(sorted(p.name for p in out.iterdir()))
