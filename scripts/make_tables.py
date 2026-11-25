"""LaTeX tables for the report, generated only from results/ (booktabs, mean ± sd, best in bold).

  tab_probes.tex      test Recall@1 (mean ± sd over seeds) and seed-ensemble mean km for every probe family
                      under PRiSM's random split and the speaker-disjoint split, with the reference baselines
  tab_leakage.tex     RQ0 evidence: speaker lookup, control task, nearest-neighbour transcript matcher
  tab_order.tex       paired comparisons (speaker bootstrap): bigram vs unigram, phone vs char, BiGRU vs shuffled
  tab_mdl.tex         online codelength savings over the location prior
  tab_gemination.tex  Haryana vs New Delhi effects on the voiceless interval and closure, validity check
  tab_data.tex        the two datasets side by side: size, speakers, leakage under each split, duplicates
  tab_main.tex        compact rebuild results: references, voice and prosody, transcript probes, both splits
  tab_release.tex     compact results on PRiSM's release, as released and corrected
  tab_versions.tex    appendix: every data version and split, what each removes, W2V2 phone bigrams vs majority
  tab_cues.tex        appendix: what each data version and split shares between training and test
  tab_benchmark.tex   the cross-validated benchmark, both tracks: R@1, lift over the majority (95% CI), km gain
Output: paper/tables/*.tex
Run:    uv run python scripts/make_tables.py
"""
import re

import numpy as np
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


GEM_LABELS = {  # plain names for the model labels in results/gemination/mixed_model.txt
    "voiceless interval (primary)": "Voiceless interval (main model)",
    "voiceless interval, no rate control": "Voiceless interval, no speaking-rate control",
    "voiceless interval, all contexts (sensitivity: includes stops after a deleted schwa)":
        "Voiceless interval, also stops after a dropped schwa",
    "closure, tokens with a burst": "Closure, stops with a burst",
    "closure, clear bursts only (rise >= 6 dB, drop >= 10 dB)": "Closure, clear bursts only",
}


def tab_gemination():
    mm = (RESULTS / "gemination" / "mixed_model.txt").read_text()
    pat = r"^(.*?): Haryana effect on (.*?) = x([\d.]+) \(95% CI x([\d.]+) to x([\d.]+)\), p = ([\deE.+-]+); tokens (\d+)"
    lines = [r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrr@{}}", r"\toprule",
             r"Model & Haryana / New Delhi & 95\% CI & tokens \\", r"\midrule"]
    for m in re.finditer(pat, mm, flags=re.M):
        lines.append(f"{GEM_LABELS.get(m[1], m[1])} & $\\times${m[3]} & [{m[4]}, {m[5]}] & {int(m[7]):,} \\\\")
    v = re.search(r"validity: lexical geminate voiceless interval = x([\d.]+) of single \(95% CI x([\d.]+) to x([\d.]+)\)", mm)
    if v:
        lines += [r"\midrule", f"Lexical geminate vs single stop, New Delhi & $\\times${v[1]} & [{v[2]}, {v[3]}] & \\\\"]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write("tab_gemination.tex", "\n".join(lines).replace("_", r"\_").replace(">=", r"$\geq$") + "\n")


def tab_official():
    """Official release: probes under the released split, and on the corrected version under the released split
    and a pseudo-speaker-disjoint split (mean +- sd over seeds)."""
    import glob
    import json
    runs = RESULTS / "runs"

    def sm(cfg):
        v = [json.load(open(f))["metrics"]["test"]["recall1"] for f in sorted(glob.glob(str(runs / f"{cfg}_s*" / "metrics.json")))]
        return (f"${np.mean(v):.1f} \\pm {np.std(v, ddof=1):.1f}$" if len(v) > 1 else (f"${v[0]:.1f}$" if v else "--"))
    off = pd.read_csv(RESULTS / "official" / "scores_by_duplication.csv").set_index("probe")
    cl = pd.read_csv(RESULTS / "official" / "scores_clean.csv")
    nn = "nearest-neighbour transcript (1-NN)"
    rows = [("Majority", f"{off.loc['majority', 'all_recall1']:.1f}",
             f"{cl[(cl.protocol == 'prism') & (cl.probe == 'majority')].recall1.iloc[0]:.1f}",
             f"{cl[(cl.protocol == 'speaker_split') & (cl.probe == 'majority')].recall1.iloc[0]:.1f}"),
            ("1-NN transcript match", f"{off.loc[nn, 'all_recall1']:.1f}",
             f"{cl[(cl.protocol == 'prism') & (cl.probe == nn)].recall1.iloc[0]:.1f}",
             f"{cl[(cl.protocol == 'speaker_split') & (cl.probe == nn)].recall1.iloc[0]:.1f}")]
    for lab, key in [("ECAPA speaker embedding", "dense_ecapa_official"), ("W2V2 char unigram", "tp_w2v2_official_char_unigram"),
                     ("W2V2 phone unigram", "tp_w2v2_official_phone_unigram"), ("W2V2 char bigram", "tp_w2v2_official_char_bigram"),
                     ("W2V2 phone bigram", "tp_w2v2_official_phone_bigram"), ("W2V2 char BiGRU (PRiSM)", "tp_w2v2_official_char_gru"),
                     ("W2V2 phone BiGRU", "tp_w2v2_official_phone_gru"), ("W2V2 shuffled phone BiGRU", "tp_w2v2_official_phone_shuffle"),
                     ("ZIPA char unigram", "tp_zipa_official_char_unigram"), ("ZIPA phone unigram", "tp_zipa_official_phone_unigram"),
                     ("ZIPA char bigram", "tp_zipa_official_char_bigram"), ("ZIPA phone bigram", "tp_zipa_official_phone_bigram")]:
        rows.append((lab, sm(f"{key}_prism"), sm(f"{key}_prism_clean"), sm(f"{key}_speaker_split_clean")))
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r" & Released & \multicolumn{2}{c}{Corrected} \\",
             r"\cmidrule(lr){3-4}", r"Probe & released split & released split & speaker-disjoint \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_official.tex", "\n".join(lines) + "\n")


def seedstats(c):
    import glob
    import json
    v = [json.load(open(f))["metrics"]["test"]["recall1"] for f in sorted(glob.glob(str(RESULTS / "runs" / f"{c}_s*" / "metrics.json")))]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else (np.nan, 0.0, 0)


def tab_data():
    """Our rebuild against PRiSM's release."""
    sp = pd.read_parquet(ROOT / "data" / "vaani_hi_rebuild" / "splits.parquet")
    m = pd.read_parquet(ROOT / "data" / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet", columns=["district", "audio_seconds"])
    audit = (RESULTS / "official" / "audit.txt").read_text()
    rows_o = int(re.search(r"rows (\d+)", audit)[1])
    distinct_o = int(re.search(r"distinct audio clips (\d+)", audit)[1])
    pins_o, dists_o, states_o = (int(re.search(rf"{k} (\d+)", audit)[1]) for k in ("pincodes", "districts", "states"))
    dup_o = int(re.search(r"duplicated clips (\d+)", audit)[1])
    test_o = int(re.search(r"'test': (\d+)", audit)[1])
    exact_o = float(pd.read_csv(RESULTS / "official" / "leakage.csv").set_index("split").loc["test", "exact_audio_in_train"])
    dupr = pd.read_csv(RESULTS / "rebuild_checks" / "duplicates.csv").iloc[0]
    dup_r = int(dupr.recordings_repeated)
    leak = 100 * sp.loc[sp.prism == "test", "speaker"].isin(set(sp.loc[sp.prism == "train", "speaker"])).mean()
    leak_s = 100 * sp.loc[sp.speaker_split == "test", "speaker"].isin(set(sp.loc[sp.speaker_split == "train", "speaker"])).mean()
    rows = [("Rows", f"{len(sp):,}", f"{rows_o:,}"),
            ("Distinct recordings", f"{len(sp) - dup_r:,}", f"{distinct_o:,}"),
            ("Recordings in several rows", f"{dup_r}", f"{dup_o:,}"),
            ("Hours of audio", f"{m.audio_seconds.sum() / 3600:.1f}", "--"),
            ("Speakers", f"{sp.speaker.nunique():,}", "not given"),
            ("States / districts / pincodes", f"{sp.state.nunique()} / {m.district.nunique()} / {sp.pincode.nunique()}",
             f"{states_o} / {dists_o} / {pins_o}"),
            ("Test rows, PRiSM's split", f"{(sp.prism == 'test').sum():,}", f"{test_o:,}"),
            ("Test speakers also in training, PRiSM's split", f"{leak:.1f}\\%", "all (est.)"),
            ("Test speakers also in training, speaker-disjoint split", f"{leak_s:.1f}\\%", "--"),
            ("Test clips whose exact audio is in training", f"{int(dupr.test_rows_exact_audio_in_train_prism)}",
             f"{100 * exact_o:.1f}\\%")]
    lines = [r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrr@{}}", r"\toprule",
             r" & Rebuild & Release \\", r"\midrule"]
    lines += [f"{a} & {b_} & {c} \\\\" for a, b_, c in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write("tab_data.tex", "\n".join(lines) + "\n")


def tab_main():
    """Rebuild results: references, voice and prosody, transcript probes; R@1 and km under both splits."""
    knn = pd.read_csv(RESULTS / "tables" / "knn_transcripts.csv").set_index(["transcripts", "protocol", "k"])
    pic = pd.read_csv(RESULTS / "rebuild_checks" / "picture_lookup.csv").set_index(["prompts", "protocol", "baseline"])
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{PRiSM's split} & \multicolumn{2}{c}{New speakers} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Probe & R@1 & km & R@1 & km \\", r"\midrule",
             r"\multicolumn{5}{l}{\textit{References that learn nothing about dialect}} \\"]
    for lab, name in (("Majority location", "majority"), ("Speaker-ID lookup", "speaker_lookup")):
        lines.append(f"\\quad {lab} & {b(name, 'prism', 'recall1'):.1f} & {b(name, 'prism', 'mean_km'):.0f} & "
                     f"{b(name, 'speaker_split', 'recall1'):.1f} & {b(name, 'speaker_split', 'mean_km'):.0f} \\\\")
    lines.append(f"\\quad Picture-ID lookup & {pic.loc[('all', 'prism', 'picture lookup'), 'recall1']:.1f} & -- & "
                 f"{pic.loc[('all', 'speaker_split', 'picture lookup'), 'recall1']:.1f} & -- \\\\")
    k1, k2 = knn.loc[("w2v2", "prism", 1)], knn.loc[("w2v2", "speaker_split", 1)]
    lines.append(f"\\quad Transcript match (W2V2) & {k1.recall1:.1f} & {k1.mean_km:.0f} & {k2.recall1:.1f} & {k2.mean_km:.0f} \\\\")
    groups = [("Voice and prosody", [("ECAPA voice embedding", "dense_ecapa_rebuild"),
                                     ("Pooled prosody statistics", "dense_prosody_rebuild")])]
    for rec in ("W2V2", "ZIPA"):
        groups.append((f"Transcript probes, {rec}", [(lab, f"tp_{rec.lower()}_rebuild_{k}") for lab, k in
                       [("phone unigram", "phone_unigram"), ("phone bigram", "phone_bigram"),
                        ("character BiGRU (PRiSM's)", "char_gru"), ("phone BiGRU", "phone_gru"),
                        ("shuffled phone BiGRU", "phone_shuffle")]]))
    for title, items in groups:
        lines += [r"\midrule", f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for lab, key in items:
            A, S = cfg.loc[f"{key}_prism"], cfg.loc[f"{key}_speaker_split"]
            lines.append(f"\\quad {lab} & ${pm(A.recall1_mean, A.recall1_sd)}$ & {A.ens_km:.0f} & "
                         f"${pm(S.recall1_mean, S.recall1_sd)}$ & {S.ens_km:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main.tex", "\n".join(lines) + "\n")


def tab_release():
    """PRiSM's release: as released, corrected rows under the released split, corrected rows with new speakers."""
    off = pd.read_csv(RESULTS / "official" / "scores_by_duplication.csv").set_index("probe")
    cl = pd.read_csv(RESULTS / "official" / "scores_clean.csv")
    nn = "nearest-neighbour transcript (1-NN)"

    def c(proto, probe):
        return cl[(cl.protocol == proto) & (cl.probe == probe)].recall1.iloc[0]
    rows = [("Majority location", f"{off.loc['majority', 'all_recall1']:.1f}", f"{c('prism', 'majority'):.1f}",
             f"{c('speaker_split', 'majority'):.1f}"),
            ("Transcript match", f"{off.loc[nn, 'all_recall1']:.1f}", f"{c('prism', nn):.1f}", f"{c('speaker_split', nn):.1f}")]
    for lab, key in [("ECAPA voice", "dense_ecapa_official"), ("W2V2 phone bigram", "tp_w2v2_official_phone_bigram"),
                     ("ZIPA phone bigram", "tp_zipa_official_phone_bigram"),
                     ("PRiSM's char.\\ BiGRU", "tp_w2v2_official_char_gru"),
                     ("Phone BiGRU", "tp_w2v2_official_phone_gru"),
                     ("Shuffled phone BiGRU", "tp_w2v2_official_phone_shuffle")]:
        cells = []
        for cfg_ in (f"{key}_prism", f"{key}_prism_clean", f"{key}_speaker_split_clean"):
            m_, s_, n_ = seedstats(cfg_)
            cells.append(f"${pm(m_, s_)}$" if n_ else "--")
        rows.append((lab, *cells))
    lines = [r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrr@{}}", r"\toprule",
             r"Probe & Released & Corrected & New speakers \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write("tab_release.tex", "\n".join(lines) + "\n")


def tab_versions():
    """Every data version and split, what each removes, and one probe (W2V2 phone bigrams) against the majority
    on the same test utterances and labels."""
    off = pd.read_csv(RESULTS / "official" / "scores_by_duplication.csv").set_index("probe")
    cl = pd.read_csv(RESULTS / "official" / "scores_clean.csv")
    ctrl = pd.read_csv(RESULTS / "tables" / "control.csv").set_index("probe")

    def maj_clean(proto):
        return cl[(cl.protocol == proto) & (cl.probe == "majority")].recall1.iloc[0]

    def seeds(c):
        m_, s_, n_ = seedstats(c)
        return f"${pm(m_, s_)}$" if n_ else "--"

    def rb(c):
        return f"${pm(cfg.loc[c, 'recall1_mean'], cfg.loc[c, 'recall1_sd'])}$"

    def ct(c):
        return f"${pm(ctrl.loc[c, 'control_recall1'], ctrl.loc[c, 'control_sd'])}$", f"{ctrl.loc[c, 'control_majority']:.1f}"

    bg = "tp_w2v2_rebuild_phone_bigram"
    ob = "tp_w2v2_official_phone_bigram"
    rows = [
        ("PRiSM's release", "Nothing (the data as published)",
         seeds(f"{ob}_prism"), f"{off.loc['majority', 'all_recall1']:.1f}", "--", "--"),
        ("Corrected release", "Duplicate audio, conflicting labels, Shimla's coordinates. New speakers are "
         "estimated by clustering voice embeddings",
         seeds(f"{ob}_prism_clean"), f"{maj_clean('prism'):.1f}",
         seeds(f"{ob}_speaker_split_clean"), f"{maj_clean('speaker_split'):.1f}"),
        ("Our rebuild", "The shard bug; speakers are known from Vaani's speaker codes",
         rb(f"{bg}_prism"), f"{b('majority', 'prism', 'recall1'):.1f}",
         rb(f"{bg}_speaker_split"), f"{b('majority', 'speaker_split', 'recall1'):.1f}"),
        ("Rebuild, shared pictures only", "Topic (shared pictures are not tied to one district)",
         rb(f"{bg}_prism_generic"), f"{cfg.loc[f'{bg}_prism_generic', 'maj_recall1']:.1f}",
         rb(f"{bg}_speaker_split_generic"), f"{cfg.loc[f'{bg}_speaker_split_generic', 'maj_recall1']:.1f}"),
        ("Rebuild, control labels", "Dialect (each speaker gets another speaker's location)",
         *ct(f"{bg}_prism"), *ct(f"{bg}_speaker_split")),
        ("Benchmark (November)", "Look-alike speaker codes (merged into one speaker), reliance on one test split "
         "(5-fold cross-validation), misplaced pincodes", r"\multicolumn{4}{c}{not run yet}"),
    ]
    lines = [r"\begin{tabularx}{\linewidth}{@{}p{3.6cm}>{\raggedright\arraybackslash}Xrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{PRiSM's split} & \multicolumn{2}{c}{New speakers} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
             r"Data & What it removes & Probe & Majority & Probe & Majority \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write("tab_versions.tex", "\n".join(lines) + "\n")


def tab_cues():
    """Appendix: for every data version and split, how much non-dialect information training and test share."""
    t = pd.read_csv(RESULTS / "tables" / "shared_cues.csv")
    split = {"prism": "PRiSM's", "speaker_split": "New speakers", "folds": "5 folds"}

    def pc(x):
        return "--" if pd.isna(x) else f"{x:.1f}"

    lines = [r"\begin{tabular}{@{}llrrrrrc@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Same speaker} & \multicolumn{2}{c}{Picture lookup} & Labels \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Data & Split & Same audio & Code & Code or look-alike & R@1 & Lift & real \\", r"\midrule"]
    prev = None
    for r in t.itertuples():
        if prev is not None and r.data.split(",")[0] != prev:
            lines.append(r"\midrule")
        prev = r.data.split(",")[0]
        if not r.built:
            lines.append(f"{r.data} & {split[r.split]} & \\multicolumn{{6}}{{c}}{{not built (no speaker labels)}} \\\\")
            continue
        if r.data.endswith("release"):
            code = f"{r.est_speaker:.0f} (est.)" if r.est_speaker >= 10 else f"{r.est_speaker:.1f} (est.)"
            cells = [pc(r.same_audio), code, "--", r"\multicolumn{2}{c}{no picture IDs}"]
        else:
            lift = f"{r.topic_lift:+.1f}".replace("-", "$-$")
            cells = [pc(r.same_audio), pc(r.same_code), pc(r.same_group), f"{r.lookup:.1f}", lift]
        lines.append(" & ".join([r.data, split[r.split], *cells, r.dialect]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_cues.tex", "\n".join(lines) + "\n")


def tab_benchmark():
    """The cross-validated benchmark (results/benchmark/scores.csv): every speaker group tested once, look-alike speaker
    codes merged, corrected coordinates; all pictures and shared pictures only. Rows without runs yet say pending."""
    f = RESULTS / "benchmark" / "scores.csv"
    if not f.exists():
        return
    sc = pd.read_csv(f).set_index(["track", "probe"])

    def cells(track, key):
        if (track, key) not in sc.index:
            return [r"\multicolumn{3}{c}{pending}"]
        r = sc.loc[(track, key)]
        lift = ("--" if key == "majority" else
                f"{r.recall1_vs_majority:+.1f} [{r.recall1_vs_majority_lo:.1f}, {r.recall1_vs_majority_hi:.1f}]".replace("-", "$-$"))
        return [f"{r.recall1:.1f}", lift, f"{r.km_vs_constant:+.0f}".replace("-", "$-$")]

    groups = [("References that learn nothing about dialect", [
                  ("Majority location", "majority"), ("Picture-ID lookup", "picture lookup"),
                  ("Transcript match (W2V2)", "1-NN transcript (W2V2)"), ("Transcript match (ZIPA)", "1-NN transcript (ZIPA)")]),
              ("Voice and prosody", [("ECAPA voice embedding", "dense_ecapa_rebuild"),
                                     ("Pooled prosody statistics", "dense_prosody_rebuild")])]
    for rec in ("W2V2", "ZIPA"):
        groups.append((f"Transcript probes, {rec}", [(lab, f"tp_{rec.lower()}_rebuild_{k}") for lab, k in
                       [("phone unigram", "phone_unigram"), ("phone bigram", "phone_bigram"),
                        ("character BiGRU (PRiSM's)", "char_gru"), ("phone BiGRU", "phone_gru"),
                        ("shuffled phone BiGRU", "phone_shuffle")]]))
    rp = sorted({k for _, k in sc.index if str(k).startswith("rp_")})
    if rp:
        names = {"attn": "attention (PRiSM's)", "gru": "BiGRU", "chunkshuffle": "BiGRU on shuffled 200 ms chunks",
                 "attn_pe": "attention with positions", "transformer": "Transformer"}
        groups.append(("Representation probes", [(f"{k.split('_rebuild_')[0][3:].upper()} {k.split('_rebuild_')[1].split('_', 1)[0]} "
                                                    f"{names.get(k.split('_rebuild_')[1].split('_', 1)[1], k)}", k) for k in rp]))
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{All pictures} & \multicolumn{3}{c}{Shared pictures} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Probe & R@1 & vs majority & km gain & R@1 & vs majority & km gain \\"]
    for title, items in groups:
        lines += [r"\midrule", f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for lab, key in items:
            lines.append(f"\\quad {lab} & " + " & ".join(cells("all", key) + cells("shared", key)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_benchmark.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    tab_benchmark()
    tab_cues()
    tab_versions()
    tab_data()
    tab_main()
    tab_release()
    tab_probes()
    tab_order()
    tab_mdl()
    tab_gemination()
    tab_official()
