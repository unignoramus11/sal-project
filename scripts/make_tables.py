"""LaTeX tables for the report, generated only from results/ (booktabs, mean ± sd, best in bold).

  tab_probes.tex      test Recall@1 (mean ± sd over seeds) and seed-ensemble mean km for every probe family
                      under PRiSM's random split and the speaker-disjoint split, with the reference baselines
  tab_leakage.tex     RQ0 evidence: speaker lookup, control task, nearest-neighbour transcript matcher
  tab_order.tex       paired comparisons (speaker bootstrap): bigram vs unigram, phone vs char, BiGRU vs shuffled
  tab_mdl.tex         online codelength savings over the location prior
  tab_gemination.tex  Haryana vs New Delhi effects on the voiceless interval and closure, validity check
Output: paper/tables/*.tex
Run:    uv run python scripts/make_tables.py
"""
import re

import pandas as pd

from sal.paths import RESULTS, ROOT

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
cfg = pd.read_csv(RESULTS / "tables" / "configs.csv").set_index("config")
base = pd.read_csv(RESULTS / "baselines.csv")


def b(name, proto, col):
    r = base[(base.baseline == name) & (base.protocol == proto) & (base.prompts == "all") & (base.split == "test")]
    return float(r.iloc[0][col])


def pm(m, s, n=2):
    return f"{m:.1f} \\pm {s:.1f}" if n > 1 else f"{m:.1f}"


def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", OUT / name)


def tab_probes():
    fams = [("ECAPA speaker embedding", "dense_ecapa_rebuild"), ("Prosody statistics", "dense_prosody_rebuild")]
    for rec in ("w2v2", "zipa"):
        for lab, k in [("char unigram", "char_unigram"), ("phone unigram", "phone_unigram"),
                       ("char bigram", "char_bigram"), ("phone bigram", "phone_bigram"),
                       ("char BiGRU", "char_gru"), ("phone BiGRU", "phone_gru"), ("shuffled phone BiGRU", "phone_shuffle")]:
            fams.append((f"{rec.upper()} {lab}", f"tp_{rec}_rebuild_{k}"))
    rows = []
    for lab, key in fams:
        a, s = f"{key}_prism", f"{key}_speaker_split"
        if a not in cfg.index or s not in cfg.index:
            continue
        A, S = cfg.loc[a], cfg.loc[s]
        rows.append((lab, A.recall1_mean, A.recall1_sd, A.ens_km, S.recall1_mean, S.recall1_sd, S.ens_km,
                     int(min(A.seeds, S.seeds))))
    best_r = max(r[1] for r in rows[2:]) if len(rows) > 2 else None
    best_s = max(r[4] for r in rows[2:]) if len(rows) > 2 else None
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Random split (PRiSM)} & \multicolumn{2}{c}{Speaker-disjoint} & \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Probe & R@1 & km & R@1 & km & seeds \\", r"\midrule"]
    for name in ("majority", "speaker_lookup"):
        lab = "Majority location" if name == "majority" else "Speaker lookup (ID only)"
        lines.append(f"{lab} & {b(name, 'prism', 'recall1'):.1f} & {b(name, 'prism', 'mean_km'):.0f} & "
                     f"{b(name, 'speaker_split', 'recall1'):.1f} & {b(name, 'speaker_split', 'mean_km'):.0f} & -- \\\\")
    lines.append(r"\midrule")
    for i, (lab, ra, sa, ka, rs, ss, ks, n) in enumerate(rows):
        fa = f"$\\mathbf{{{pm(ra, sa, n)}}}$" if i >= 2 and ra == best_r else f"${pm(ra, sa, n)}$"
        fs = f"$\\mathbf{{{pm(rs, ss, n)}}}$" if i >= 2 and rs == best_s else f"${pm(rs, ss, n)}$"
        lines.append(f"{lab} & {fa} & {ka:.0f} & {fs} & {ks:.0f} & {n} \\\\")
        if i == 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_probes.tex", "\n".join(lines) + "\n")


def tab_order():
    c = pd.read_csv(RESULTS / "tables" / "comparisons.csv")
    nice = lambda x: (x.replace("tp_", "").replace("_rebuild", "").replace("_speaker_split", "")  # noqa: E731
                      .replace("_prism", "").replace("dense_", "").replace("_", " "))
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Split & Comparison ($a - b$) & $\Delta$R@1 & 95\% CI & $\Delta$km \\",
             r"\midrule"]
    for _, r in c.iterrows():
        split = "spk" if r.a.endswith("speaker_split") else "rand"
        lines.append(f"{split} & {nice(r.a)} vs {nice(r.b)} & {r.recall1_diff:+.1f} & [{r.recall1_lo:.1f}, {r.recall1_hi:.1f}] "
                     f"& {r.km_diff:+.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_order.tex", "\n".join(lines) + "\n")


def tab_mdl():
    m = pd.read_csv(RESULTS / "mdl" / "summary.csv")
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Probe & Random split & Speaker-disjoint \\", r"\midrule"]
    keys = sorted({re.sub(r"_(prism|speaker_split)$", "", c) for c in m.config})
    m = m.set_index("config")
    for k in keys:
        cells = []
        for proto in ("prism", "speaker_split"):
            c = f"{k}_{proto}"
            cells.append(f"${m.loc[c, 'gain_over_prior']:.1f} \\pm {m.loc[c, 'gain_sd']:.1f}$" if c in m.index else "--")
        lines.append(f"{k.replace('mdl_', '').replace('_rebuild', '').replace('_', ' ')} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_mdl.tex", "\n".join(lines) + "\n")


def tab_gemination():
    mm = (RESULTS / "gemination" / "mixed_model.txt").read_text()
    pat = r"^(.*?): Haryana effect on (.*?) = x([\d.]+) \(95% CI x([\d.]+) to x([\d.]+)\), p = ([\deE.+-]+); tokens (\d+)"
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Haryana / New Delhi & 95\% CI & tokens \\", r"\midrule"]
    for m in re.finditer(pat, mm, flags=re.M):
        lines.append(f"{m[1]} & $\\times${m[3]} & [{m[4]}, {m[5]}] & {int(m[7]):,} \\\\")
    v = re.search(r"validity: lexical geminate voiceless interval = x([\d.]+) of single \(95% CI x([\d.]+) to x([\d.]+)\)", mm)
    if v:
        lines += [r"\midrule", f"lexical geminate / single (New Delhi) & $\\times${v[1]} & [{v[2]}, {v[3]}] & \\\\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_gemination.tex", "\n".join(lines).replace("_", r"\_") + "\n")


if __name__ == "__main__":
    tab_probes()
    tab_order()
    tab_mdl()
    tab_gemination()
