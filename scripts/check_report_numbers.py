"""Cross-check of every number quoted in the report's prose (paper/report.tex) against results/.

Each entry maps a phrase exactly as the report prints it to the same phrase rebuilt from the results files at the
printed precision. A phrase that is missing from the report, or that differs from its recomputed version, is listed
and the script exits with status 1. Tables and figures need no entry: scripts/make_tables.py and
scripts/make_figures.py generate them. When the prose changes, update the matching entry here.

The final report (--tex paper/final.tex) is checked against the same claims, less those it no longer makes
(FINAL_DROP), plus its own (FINAL_EXTRA); it also fails while any \\pending{...} draft marker is left in it.

Run:    uv run python scripts/check_report_numbers.py [--tex paper/final.tex]   (also run by `make report`, `make final`)
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sal.paths import DATA, OUTPUTS, RESULTS, ROOT

R = str(RESULTS) + "/"
ap = argparse.ArgumentParser()
ap.add_argument("--tex", default=str(ROOT / "paper" / "report.tex"))
args = ap.parse_args()
FINAL = args.tex.endswith("final.tex")
tex = Path(args.tex).read_text()
flat = re.sub(r"\s+", " ", tex)

cfg = pd.read_csv(R + "tables/configs.csv").set_index("config")
base = pd.read_csv(R + "baselines.csv")
comp = pd.read_csv(R + "tables/comparisons.csv").set_index(["a", "b"])
knn = pd.read_csv(R + "tables/knn_transcripts.csv").set_index(["transcripts", "protocol", "k"])
pic = pd.read_csv(R + "rebuild_checks/picture_lookup.csv").set_index(["prompts", "protocol", "baseline"])
look = pd.read_csv(R + "rebuild_checks/lookalike.csv").set_index("probe")
oo = pd.read_csv(R + "official/order_clean.csv").set_index(["protocol", "a"])
ocl = pd.read_csv(R + "official/scores_clean.csv")
odup = pd.read_csv(R + "official/scores_by_duplication.csv").set_index("probe")
pc = pd.read_csv(R + "tables/prosody_checks.csv").set_index(["subset", "comparison"])
mdl = pd.read_csv(R + "mdl/summary.csv").set_index("config")
dbl = pd.read_csv(R + "gemination/model_doubling.csv").set_index(["model", "region"])
ctrl = pd.read_csv(R + "tables/control.csv").set_index("probe")
cues = pd.read_csv(R + "tables/shared_cues.csv").set_index(["data", "split"])
mm = open(R + "gemination/mixed_model.txt").read()
rb = open(R + "gemination/robustness.txt").read()
audit = open(R + "official/audit.txt").read()
cal = open(R + "official/speaker_calibration.txt").read()
sp = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")
pr = pd.read_parquet(DATA / "vaani_hi_rebuild" / "prompts.parquet")
meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet", columns=["audio_seconds", "district"])


def bl(name, proto):
    return base[(base.baseline == name) & (base.protocol == proto) & (base.prompts == "all") & (base.split == "test")].iloc[0]


def seed(c):
    v = [json.load(open(f))["metrics"]["test"]["recall1"] for f in sorted(glob.glob(R + f"runs/{c}_s*/metrics.json"))]
    return np.mean(v), np.std(v, ddof=1)


def f1(x):
    return f"{x:.1f}"


def ci(e, lo, hi):
    """Estimate with its interval, minus signs as the report sets them."""
    return f"{e:.1f} [{lo:.1f}, {hi:.1f}]".replace("-", "$-$")


def o(proto, a):
    return oo.loc[(proto, a)]


def cp(a, b):
    return comp.loc[(a, b)]


leak = 100 * sp.loc[sp.prism == "test", "speaker"].isin(set(sp.loc[sp.prism == "train", "speaker"])).mean()
spk = sp.groupby("speaker").size()
grp = sp["speaker"].map(pd.read_csv(R + "benchmark/speaker_groups.csv", index_col=0)["group"])
te_s = sp[sp.speaker_split == "test"]
lookalike_spk = te_s.loc[grp[te_s.index].isin(set(grp[sp.speaker_split == "train"])), "speaker"].nunique()
rows_o, distinct_o = int(re.search(r"rows (\d+)", audit)[1]), int(re.search(r"distinct audio clips (\d+)", audit)[1])
dup_o = int(re.search(r"duplicated clips (\d+)", audit)[1])
new_spk = ocl[(ocl.protocol == "speaker_split") & (ocl.probe != "majority")].recall1
claims = {
 # abstract / intro
 "45.5 Recall@1": f1(seed("tp_w2v2_official_char_gru_prism")[0]) + " Recall@1",
 "speakers apart scores 55.1": "speakers apart scores " + f1(seed("dense_ecapa_official_prism")[0]),
 "speaker scores 98.0": "speaker scores " + f1(bl("speaker_lookup", "prism").recall1),
 "was described scores 61.8": "was described scores " + f1(pic.loc[("all", "prism", "picture lookup"), "recall1"]),
 # data
 "obtained 22,551 utterances (40.3 hours)": f"obtained {len(sp):,} utterances ({meta.audio_seconds.sum()/3600:.1f} hours)",
 "26 pincodes in 9 states": f"{sp.pincode.nunique()} pincodes in {sp.state.nunique()} states",
 "the rebuild has 1,570 speakers, with a median of 5 and at most 257": f"the rebuild has {spk.size:,} speakers, with a median of {spk.median():.0f} and at most {spk.max()}",
 "26,385 rows but only 21,481 distinct": f"{rows_o:,} rows but only {distinct_o:,} distinct",
 "4,559 recordings appear in two to four rows": f"{dup_o:,} recordings appear in two to four rows",
 "28.0\\% of test rows": f"{100*pd.read_csv(R+'official/leakage.csv').set_index('split').loc['test','exact_audio_in_train']:.1f}\\% of test rows",
 "844\\,km from its post offices": f"{pd.read_csv(R+'official/coordinates_audit.csv').iloc[0].km_from_valid_median:.0f}\\,km from its post offices",
 "97.9\\% of test utterances": f"{leak:.1f}\\% of test utterances",
 "68\\% of our utterances": f"{100*(pr.prompt_type=='SPECIFIC').mean():.0f}\\% of our utterances",
 # RQ0
 "reaches 98.0 Recall@1, against 10.5": f"reaches {f1(bl('speaker_lookup','prism').recall1)} Recall@1, against {f1(bl('majority','prism').recall1)}",
 "embedding reaches 41.6": "embedding reaches " + f1(cfg.loc["dense_ecapa_rebuild_prism", "recall1_mean"]),
 "ECAPA still scores 33.2 against a control majority of 14.0": f"ECAPA still scores {f1(ctrl.loc['dense_ecapa_rebuild_prism','control_recall1'])} against a control majority of {f1(ctrl.loc['dense_ecapa_rebuild_prism','control_majority'])}",
 "ECAPA scores 10.1, the majority level": f"ECAPA scores {f1(cfg.loc['dense_ecapa_rebuild_speaker_split','recall1_mean'])}, the majority level",
 "matcher reaches 39.6 under": "matcher reaches " + f1(knn.loc[("w2v2","prism",1),"recall1"]) + " under",
 "for 25\\% of test utterances": f"for {100*knn.loc[('w2v2','prism',1),'nn_same_speaker']:.0f}\\% of test utterances",
 "On new speakers it scores 18.2": "On new speakers it scores " + f1(knn.loc[("w2v2","speaker_split",1),"recall1"]),
 "picture ID scores 61.8 [58.9, 64.7]": "picture ID scores " + ci(*pic.loc[("all","prism","picture lookup"),["recall1","lo","hi"]]),
 "and 27.5 [20.7, 36.5] on new speakers": "and " + ci(*pic.loc[("all","speaker_split","picture lookup"),["recall1","lo","hi"]]) + " on new speakers",
 "lookup scores 10.1, below that subset's majority of 21.3": f"lookup scores {f1(pic.loc[('shared','speaker_split','picture lookup'),'recall1'])}, below that subset's majority of {f1(pic.loc[('shared','speaker_split','majority'),'recall1'])}",
 "by 13.7 [8.2, 19.1] points for W2V2 and 12.5 [7.6, 17.5] for ZIPA": "by {} points for W2V2 and {} for ZIPA".format(*[ci(*cfg.loc[f"tp_{m}_rebuild_phone_bigram_prism_generic", ["lift_recall1", "lift_recall1_lo", "lift_recall1_hi"]]) for m in ("w2v2", "zipa")]),
 "In 34.8\\% of its test utterances": "In {:.1f}\\% of its test utterances".format(cues.loc[("Rebuild", "speaker_split"), "same_group"]),
 "control labels (43.5 against a majority of 14.0)": "control labels ({:.1f} against a majority of {:.1f})".format(*cues.loc[("Rebuild, control labels", "prism"), ["lookup", "majority"]]),
 "at 21.3 Recall@1, 10.6 [6.2, 15.9] points": f"at {f1(cfg.loc['tp_zipa_rebuild_phone_bigram_speaker_split','recall1_mean'])} Recall@1, " + ci(*cfg.loc['tp_zipa_rebuild_phone_bigram_speaker_split',['lift_recall1','lift_recall1_lo','lift_recall1_hi']]) + " points",
 # release
 "45.5\\,$\\pm$\\,1.1": "{:.1f}\\,$\\pm$\\,{:.1f}".format(*seed("tp_w2v2_official_char_gru_prism")),
 "ECAPA voice embedding reaches 55.1 and the transcript matcher 58.9": f"ECAPA voice embedding reaches {f1(seed('dense_ecapa_official_prism')[0])} and the transcript matcher {f1(odup.loc['nearest-neighbour transcript (1-NN)','all_recall1'])}",
 "(19,731 rows)": f"({len(pd.read_parquet(R + 'official/splits_clean.parquet')):,} rows)",
 "into 375 pseudo-speakers": "into " + re.search(r"official: (\d+) pseudo-speakers", cal)[1] + " pseudo-speakers",
 "leaves 2.1\\% of test speakers": "leaves " + re.search(r"chosen threshold [\d.]+ \(rebuild true test leakage ([\d.]+)%", cal)[1] + "\\% of test speakers",
 "between 3.4 and 9.6": f"between {new_spk.min():.1f} and {new_spk.max():.1f}",
 "majority of 6.8": "majority of " + f1(ocl[(ocl.protocol=='speaker_split')&(ocl.probe=='majority')].recall1.iloc[0]),
 # RQ1
 "by 18.6 [15.5, 22.3] points": "by " + ci(*o("prism","tp_w2v2_official_phone_gru")[["diff","lo","hi"]]) + " points",
 "by $-$0.3 [$-$3.5, 2.4]": "by " + ci(*o("speaker_split","tp_w2v2_official_phone_gru")[["diff","lo","hi"]]),
 "(3.9 for W2V2, 4.3 for ZIPA)": f"({f1(cp('tp_w2v2_rebuild_phone_gru_prism','tp_w2v2_rebuild_phone_shuffle_prism').recall1_diff)} for W2V2, {f1(cp('tp_zipa_rebuild_phone_gru_prism','tp_zipa_rebuild_phone_shuffle_prism').recall1_diff)} for ZIPA)",
 "($-$3.5 [$-$6.5, $-$1.0] and 0.4 [$-$1.9, 2.9])": "(" + ci(*cp('tp_w2v2_rebuild_phone_gru_speaker_split','tp_w2v2_rebuild_phone_shuffle_speaker_split')[['recall1_diff','recall1_lo','recall1_hi']]) + " and " + ci(*cp('tp_zipa_rebuild_phone_gru_speaker_split','tp_zipa_rebuild_phone_shuffle_speaker_split')[['recall1_diff','recall1_lo','recall1_hi']]) + ")",
 "by 4.8 [2.6, 6.9] points for W2V2 and 7.8": "by " + ci(*cp('tp_w2v2_rebuild_phone_bigram_speaker_split','tp_w2v2_rebuild_phone_unigram_speaker_split')[['recall1_diff','recall1_lo','recall1_hi']]) + " points for W2V2 and " + f1(cp('tp_zipa_rebuild_phone_bigram_speaker_split','tp_zipa_rebuild_phone_unigram_speaker_split').recall1_diff),
 "by 3.5 [2.1, 5.1] for W2V2": "by " + ci(*cp('tp_w2v2_rebuild_phone_bigram_speaker_split','tp_w2v2_rebuild_char_bigram_speaker_split')[['recall1_diff','recall1_lo','recall1_hi']]) + " for W2V2",
 "gain is 1.8 [$-$0.9, 4.7]": "gain is " + ci(*o("speaker_split","tp_w2v2_official_phone_bigram")[["diff","lo","hi"]]),
 "only 59 test pseudo-speakers": f"only {ocl[ocl.protocol=='speaker_split'].test_pseudo_speakers.iloc[0]} test pseudo-speakers",
 "save 7.9\\,kbits": f"save {mdl.loc['mdl_zipa_rebuild_phone_bigram_speaker_split','gain_over_prior']:.1f}\\,kbits",
 "against 21.0\\,kbits": f"against {mdl.loc['mdl_zipa_rebuild_phone_bigram_prism','gain_over_prior']:.1f}\\,kbits",
 # RQ2
 "We measured 26,033 single voiceless stops between vowels, from 3,003 speakers": "We measured {:,} single voiceless stops between vowels, from {:,} speakers".format(*map(int, re.search(r"tokens (\d+), speakers (\d+)", mm).groups())),
 "$\\times$1.14 [1.11, 1.18] as long": "$\\times${:.2f} [{:.2f}, {:.2f}] as long".format(*map(float, re.search(r"primary\).*?x([\d.]+) \(95% CI x([\d.]+) to x([\d.]+)", mm).groups())),
 "closure $\\times$1.26 [1.21, 1.31]": "closure $\\times${:.2f} [{:.2f}, {:.2f}]".format(*map(float, re.search(r"closure, tokens with a burst.*?x([\d.]+) \(95% CI x([\d.]+) to x([\d.]+)", mm).groups())),
 "($\\times$1.13 to $\\times$1.15)": "($\\times${:.2f} to $\\times${:.2f})".format(min(map(float, re.findall(r"district \w+ vs NewDelhi: x([\d.]+)", rb))), max(map(float, re.findall(r"district \w+ vs NewDelhi: x([\d.]+)", rb)))),
 "in 45 of 57": "in " + "{} of {}".format(*re.search(r"Haryana longer in (\d+) of (\d+) words", rb).groups()),
 "model ($\\times$1.14)": "model ($\\times${:.2f})".format(float(re.search(r"speaker level:.*?x([\d.]+)", rb)[1])),
 "$\\times$1.59 [1.51, 1.67]": "$\\times${} [{}, {}]".format(*re.search(r"validity: lexical geminate voiceless interval = x([\d.]+) of single \(95% CI x([\d.]+) to x([\d.]+)\)", mm).groups()),
 "for only 1.0\\% of the Haryana stops and ZIPA for 1.9\\%": f"for only {100*dbl.loc[('w2v2','Haryana'),'doubled_share']:.1f}\\% of the Haryana stops and ZIPA for {100*dbl.loc[('zipa','Haryana'),'doubled_share']:.1f}\\%",
 "(AUC 0.73": f"(AUC {dbl.loc[('w2v2','Haryana'),'auc_voi_predicts_doubling']:.2f}",
 "are not (0.46)": f"are not ({dbl.loc[('zipa','Haryana'),'auc_voi_predicts_doubling']:.2f})",
 "(1.00\\% against 1.17\\%, $p = 0.012$)": "({:.2f}\\% against {:.2f}\\%, $p = {:.3f}$)".format(*(lambda r: (100*r.haryana, 100*r.rest, r.p))(pd.read_csv(R+'dpr/w2v2_haryana_vs_rest.csv').iloc[0])),
 "(1.63\\% against 1.82\\%, $p = 0.30$)": "({:.2f}\\% against {:.2f}\\%, $p = {:.2f}$)".format(*(lambda r: (100*r.haryana, 100*r.rest, r.p))(pd.read_csv(R+'dpr/zipa_haryana_vs_rest.csv').iloc[0])),
 # RQ3
 "(4.8 Recall@1 on new speakers)": f"({f1(cfg.loc['dense_prosody_rebuild_speaker_split','recall1_mean'])} Recall@1 on new speakers)",
 "scores 11.3 on new speakers against a majority of 10.0, a difference of 1.3 [$-$5.1, 7.8]": f"scores {json.load(open(R+'runs/prosody_gru_speaker_split_s42/metrics.json'))['metrics']['test']['recall1']:.1f} on new speakers against a majority of {f1(bl('majority','speaker_split').recall1)}, a difference of " + ci(*pc.loc[('all','prosody - majority, Recall@1'),['diff','lo','hi']]),
 "from 441 to 426\\,km ($-$14.9 [$-$21.6, $-$8.1])": "from {:.0f} to {:.0f}\\,km (".format(*pd.read_csv(R+'tables/fusion_tp_zipa_rebuild_phone_gru_speaker_split__prosody_gru_speaker_split.csv').iloc[0][['a_mean_km','fused_mean_km']]) + ci(*pc.loc[('all','fusion - transcript, km'),['diff','lo','hi']]) + ")",
 "($-$1.2 [$-$3.4, 0.8])": "(" + ci(*pc.loc[('all','fusion - transcript, Recall@1'),['diff','lo','hi']]) + ")",
 # checks
 "merges 1,570 codes into 1,327 groups": "merges {:,} codes into {:,} groups".format(*map(int, re.search(r"(\d+) codes -> (\d+) groups", open(R+'benchmark/report.txt').read()).groups())),
 "49 of 223 test speakers": f"{lookalike_spk} of {te_s.speaker.nunique()} test speakers",
 "$+$9.1 on their utterances and $-$4.2 [$-$12.1, 2.2]": "$+${:.1f} on their utterances and ".format(look.loc['dense_ecapa_rebuild_speaker_split','lookalike_lift']) + ci(*look.loc['dense_ecapa_rebuild_speaker_split',['strict_lift','strict_lift_lo','strict_lift_hi']]),
 "phone bigrams $+$11.6 [5.2, 17.6]": "phone bigrams $+$" + ci(*look.loc['tp_zipa_rebuild_phone_bigram_speaker_split',['strict_lift','strict_lift_lo','strict_lift_hi']]),
 "from 132 to 48\\,km [$-$14, 123]": "from {:.0f} to {:.0f}\\,km [{:.0f}, {:.0f}]".format(pc.loc[('all','best constant - prosody, km'),'diff'], *pc.loc[('strict','best constant - prosody, km'),['diff','lo','hi']]).replace("-", "$-$"),
 "stays at $-$12.9\\,km [$-$20.0, $-$5.1]": "stays at " + "{:.1f}\\,km [{:.1f}, {:.1f}]".format(*pc.loc[('strict','fusion - transcript, km'),['diff','lo','hi']]).replace("-", "$-$"),
 "454552 (Dhar) by 266\\,km and 834008 (Ranchi) by 61\\,km": "454552 (Dhar) by {:.0f}\\,km and 834008 (Ranchi) by {:.0f}\\,km".format(*pd.read_csv(R+'rebuild_checks/coordinates.csv').query("data=='rebuild'").set_index('pincode').loc[[454552, 834008],'km_from_cluster']),
 # status
 "transcribed 72,899 utterances": "transcribed {:,} utterances".format(
     sum(pq.ParquetFile(OUTPUTS / "transcripts" / f"w2v2_{x}.parquet").metadata.num_rows for x in ("rebuild", "official", "districts"))),
 "New Delhi, 23,963 utterances": "New Delhi, {:,} utterances".format(pq.ParquetFile(OUTPUTS / "transcripts" / "w2v2_districts.parquet").metadata.num_rows),
 "three of them have latitude and longitude": "{} of them have latitude and longitude".format(
     {"3": "three"}.get(re.search(r"(\d+) of \d+ offices have invalid", audit)[1], "?")),
 "trained 392 probe runs": f"trained {sum('_bench' not in f for f in glob.glob(R+'runs/*/metrics.json'))} probe runs",
 "measured 26,033 stops": "measured {:,} stops".format(int(re.search(r"tokens (\d+)", mm)[1])),
 "(17.1\\% against 16.4--20.0\\% for W2V2, 20.0\\% against 18.2--22.4\\% for ZIPA)": "({}\\% against {}--{}\\% for W2V2, {}\\% against {}--{}\\% for ZIPA)".format(*[x for m in ('w2v2','zipa') for x in (lambda d: (f"{100*d.set_index('state').loc['Haryana','closed_share']:.1f}", f"{100*d[d.state!='Haryana'].closed_share.min():.1f}", f"{100*d[d.state!='Haryana'].closed_share.max():.1f}"))(pd.read_csv(R+f'nidhi/{m}_syllables_by_state.csv'))]),
}
if FINAL:
    bench = pd.read_csv(R + "benchmark/scores.csv").set_index(["track", "probe"])
    bpair = pd.read_csv(R + "benchmark/paired.csv").set_index(["track", "a", "b"])
    brep = open(R + "benchmark/report.txt").read()
    fold_tests = [int(x) for x in re.findall(r"all_f\d: train \d+, val \d+, test (\d+)", brep)]
    sh = re.search(r"shared: (\d+) utterances, \d+ speaker codes, (\d+) groups, (\d+) pincodes", brep)

    def bl_(track, probe):
        r = bench.loc[(track, probe)]
        return r.recall1_vs_majority, r.recall1_vs_majority_lo, r.recall1_vs_majority_hi

    def below(track, probe):
        e, lo, hi = bl_(track, probe)
        return ci(-e, -hi, -lo)

    def bp(track, a, b):
        r = bpair.loc[(track, a, b)]
        return ci(r.recall1_diff, r.recall1_lo, r.recall1_hi)

    shared_rows = [p for t, p in bench.index if t == "shared" and p not in ("majority", "best constant")]
    tp_shared = [p for p in shared_rows if p.startswith("tp_")]
    FINAL_DROP = {
        # the progress report's status section
        "trained 392 probe runs", "transcribed 72,899 utterances", "measured 26,033 stops",
        # the abstract: restore these when the final abstract is written
        "45.5 Recall@1", "speakers apart scores 55.1", "speaker scores 98.0", "was described scores 61.8",
        # superseded by the benchmark (look-alike speakers merged, shared-picture track)
        "by 13.7 [8.2, 19.1] points for W2V2 and 12.5 [7.6, 17.5] for ZIPA", "at 21.3 Recall@1, 10.6 [6.2, 15.9] points",
        "49 of 223 test speakers", "$+$9.1 on their utterances and $-$4.2 [$-$12.1, 2.2]", "phone bigrams $+$11.6 [5.2, 17.6]",
    }
    FINAL_EXTRA = {
     "each of the 22,548 utterances": "each of the {:,} utterances".format(int(re.search(r"all: (\d+) utterances", brep)[1])),
     "(7,157 utterances from 650 speaker groups in 24 pincodes)": f"({int(sh[1]):,} utterances from {sh[2]} speaker groups in {sh[3]} pincodes)",
     "(3,801 to 6,022 test utterances)": f"({min(fold_tests):,} to {max(fold_tests):,} test utterances)",
     "This merges 1,570 codes into 1,327 groups": "This merges {:,} codes into {:,} groups".format(*map(int, re.search(r"(\d+) codes -> (\d+) groups", brep).groups())),
     "In 34.8\\% of the speaker-disjoint split's": "In {:.1f}\\% of the speaker-disjoint split's".format(cues.loc[("Rebuild", "speaker_split"), "same_group"]),
     "majority, by 2.8 [0.4, 5.6] points": "majority, by " + below("all", "dense_ecapa_rebuild") + " points",
     "the picture-ID lookup is 18.4 [15.1, 22.4] points": "the picture-ID lookup is " + ci(*bl_("all", "picture lookup")) + " points",
     "ZIPA phone bigrams, is 6.0 [3.8, 8.1] points": "ZIPA phone bigrams, is " + ci(*bl_("all", "tp_zipa_rebuild_phone_bigram")) + " points",
     "by 2.4 [1.3, 3.5] points for W2V2 and 5.1 [3.9, 6.4] for ZIPA": "by {} points for W2V2 and {} for ZIPA".format(
         bp("all", "tp_w2v2_rebuild_phone_bigram", "tp_w2v2_rebuild_phone_unigram"), bp("all", "tp_zipa_rebuild_phone_bigram", "tp_zipa_rebuild_phone_unigram")),
     "by 0.2 [$-$1.3, 1.7] and 0.3 [$-$1.6, 2.5] on shared pictures": "by {} and {} on shared pictures".format(
         bp("shared", "tp_w2v2_rebuild_phone_bigram", "tp_w2v2_rebuild_phone_unigram"), bp("shared", "tp_zipa_rebuild_phone_bigram", "tp_zipa_rebuild_phone_unigram")),
     "pooled statistics are 6.5 [4.0, 9.3] points below": "pooled statistics are " + below("all", "dense_prosody_rebuild") + " points below",
     # claims about every row, not one number
     "every lookup and every probe run so far is at or below the majority":
         "every lookup and every probe run so far is at or below the majority" if all(bl_("shared", p)[0] <= 0.05 for p in shared_rows) else "FALSE: some shared-picture lift is above zero",
     "(every interval includes zero)":
         "(every interval includes zero)" if all(bench.loc[("shared", p), "km_vs_constant_lo"] < 0 < bench.loc[("shared", p), "km_vs_constant_hi"] for p in tp_shared) else "FALSE: some km interval excludes zero",
    }
    claims = {k: v for k, v in claims.items() if k not in FINAL_DROP} | FINAL_EXTRA

bad = 0
for printed, computed in claims.items():
    inflat = printed in flat
    ok = inflat and printed == computed
    if not ok:
        bad += 1
        print(("MISMATCH" if inflat else "NOT IN TEXT"), "|", printed, "| computed:", computed)
pending = len(re.findall(r"\\pending\{", tex)) - tex.count("\\newcommand{\\pending}")
if pending:
    print(f"{pending} \\pending markers left in {Path(args.tex).name}")
print(f"{len(claims) - bad} of {len(claims)} report numbers match results/")
sys.exit(1 if bad or pending else 0)
