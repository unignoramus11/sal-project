"""Gemination analysis (RQ2): acoustics of intervocalic voiceless stops, Haryana vs New Delhi, and whether
the phone recognizers' doubling tracks the measured closure.

Inputs:  data/vaani_districts/closures.parquet (scripts/measure_closure.py)
         outputs/transcripts/{w2v2,zipa}_districts.parquet (frame-level CTC argmax ids, scripts/transcribe.py)
Tokens: main analyses use stops between vowels in speech (sal.gemination: after a vowel sign or a word-initial
consonant); stops after a word-medial consonant (whose schwa Hindi usually deletes) enter one sensitivity model.
Outputs: results/gemination/
  acoustic_summary.csv     per region (and per stop, per district): accepted tokens, speakers, closure, VOT,
                           following vowel, closure/vowel ratio (medians and speaker-bootstrap CIs of means)
  rejection.csv            share of tokens rejected per region and reason (a measurement-bias check)
  mixed_model.txt          log(voiceless interval) ~ Haryana + log(vowel) + C(word), random intercept per speaker
                           (the word fixes the stop, so aspiration is absorbed by C(word)); the same for closure on
                           tokens with a burst and with clear bursts; Lee bounds for unequal rejection by region
  geminate_vs_single.csv   validity check: closures of lexical geminates vs single stops by region (when
                           data/vaani_districts/closures_geminate.parquet exists; tools/geminate_pass.sh)
  model_doubling.csv       per recognizer: share of tokens doubled in the CTC output by region; closure for
                           doubled vs single tokens; AUC of closure for predicting doubling
Run: uv run python scripts/analyse_gemination.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sal import phones as P
from sal.gemination import token_contexts
from sal.paths import DATA, OUTPUTS, RESULTS
from sal.stats import cluster_bootstrap

out = RESULTS / "gemination"
out.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA / "vaani_districts" / "closures.parquet")
df["region"] = np.where(df["state"] == "Haryana", "Haryana", "Delhi")
df["aspirated"] = df["stop"].isin(list("खछठथ"))
df["context"] = token_contexts(df["word"], df["stop"])     # sal.gemination: is the stop between vowels in speech?
ok_all = df[df["reject"] == ""].copy()
ok_all["ratio"] = ok_all["voi_ms"] / ok_all["vowel_ms"]
ok = ok_all[ok_all["context"].isin(["vowel", "initial_schwa"])].copy()   # main analysis
print(df["context"].value_counts().to_string())

# --- rejection by region
rej = df.groupby(["region", "context"])["reject"].apply(lambda s: (s != "").mean()).rename("rejected_share").to_frame()
rej = rej.join(df[df.reject != ""].groupby(["region", "context", "reject"]).size().unstack(fill_value=0))
rej.to_csv(out / "rejection.csv")


# --- acoustic summary
def summ(g):
    r = {"tokens": len(g), "speakers": g["speaker"].nunique()}
    for col in ("voi_ms", "closure_ms", "vot_ms", "vowel_ms", "ratio"):
        v = g[col].notna().to_numpy()
        ci = cluster_bootstrap(g[col].to_numpy()[v], g["speaker"].to_numpy()[v], n=1000)
        r.update({f"{col}_median": g[col].median(), f"{col}_mean": ci["estimate"], f"{col}_lo": ci["lo"],
                  f"{col}_hi": ci["hi"]})
    return pd.Series(r)


parts = []
for level, keys in [("region", ["region"]), ("region_aspiration", ["region", "aspirated"]),
                    ("district", ["district"]), ("region_stop", ["region", "stop"]),
                    ("region_context", ["region", "context"])]:
    src = ok_all if level == "region_context" else ok
    g = src.groupby(keys).apply(summ).reset_index()
    g.insert(0, "group", g[keys].astype(str).agg(" / ".join, axis=1))
    g.insert(0, "level", level)
    parts.append(g.drop(columns=keys))
summary = pd.concat(parts, ignore_index=True)
summary.to_csv(out / "acoustic_summary.csv")

# --- mixed models with speaker random intercepts
for d_ in (ok, ok_all):
    d_["haryana"] = (d_["region"] == "Haryana").astype(int)
    d_["log_voi"] = np.log(d_["voi_ms"])
    d_["log_closure"] = np.log(d_["closure_ms"])
    d_["log_vowel"] = np.log(d_["vowel_ms"].clip(lower=20))


def frequent(d_):
    common = d_["word"].value_counts()
    return d_[d_["word"].isin(common[common >= 20].index)]


lines = []
burst = ok[ok["closure_ms"].notna() & (ok["closure_ms"] > 0)]
for name, measure_, formula, data in [
        ("voiceless interval (primary)", "voiceless interval", "log_voi ~ haryana + log_vowel + C(word)", ok),
        ("voiceless interval, no rate control", "voiceless interval", "log_voi ~ haryana + C(word)", ok),
        ("voiceless interval, all contexts (sensitivity: includes stops after a deleted schwa)", "voiceless interval",
         "log_voi ~ haryana + log_vowel + C(word)", ok_all),
        ("closure, tokens with a burst", "closure", "log_closure ~ haryana + log_vowel + C(word)", burst),
        ("closure, clear bursts only (rise >= 6 dB, drop >= 10 dB)", "closure",
         "log_closure ~ haryana + log_vowel + C(word)", burst[burst["clear_burst"]])]:
    ok_m = frequent(data)
    fit = smf.mixedlm(formula, ok_m, groups=ok_m["speaker"]).fit(method="lbfgs")
    b, (lo, hi), p = fit.params["haryana"], fit.conf_int().loc["haryana"], fit.pvalues["haryana"]
    lines.append(f"{name}: Haryana effect on {measure_} = x{np.exp(b):.3f} (95% CI x{np.exp(lo):.3f} to x{np.exp(hi):.3f}),"
                 f" p = {p:.3g}; tokens {len(ok_m)}, speakers {ok_m['speaker'].nunique()}, words {ok_m['word'].nunique()}")
    lines.append(fit.summary().tables[1].loc[["Intercept", "haryana", "log_vowel"] if "log_vowel" in formula
                                            else ["Intercept", "haryana"]].to_string())

# --- unequal rejection: Lee bounds on the raw Haryana - Delhi difference in mean log voiceless interval
main = df[df["context"].isin(["vowel", "initial_schwa"])]
acc = main.groupby("region")["reject"].apply(lambda s: (s == "").mean())
lv = {r: np.sort(np.log(ok.loc[ok["region"] == r, "voi_ms"].to_numpy())) for r in ("Haryana", "Delhi")}
hi_r, lo_r = (("Haryana", "Delhi") if acc["Haryana"] > acc["Delhi"] else ("Delhi", "Haryana"))
q = (acc[hi_r] - acc[lo_r]) / acc[hi_r]            # share of the better-measured region's tokens to trim
k = int(round(q * len(lv[hi_r])))
trim_low, trim_high = lv[hi_r][k:], lv[hi_r][: len(lv[hi_r]) - k]
sign = 1 if hi_r == "Haryana" else -1
bounds = sorted([sign * (trim_low.mean() - lv[lo_r].mean()), sign * (trim_high.mean() - lv[lo_r].mean())])
raw = lv["Haryana"].mean() - lv["Delhi"].mean()
lines.append(f"acceptance (main contexts): Haryana {acc['Haryana']:.3f}, Delhi {acc['Delhi']:.3f}; raw Haryana - Delhi "
             f"mean log voiceless interval {raw:+.3f} (x{np.exp(raw):.3f}); Lee bounds x{np.exp(bounds[0]):.3f} to "
             f"x{np.exp(bounds[1]):.3f} (trimming {100 * q:.1f}% of {hi_r} tokens)")
(out / "mixed_model.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))


# --- validity: lexical geminates (अच्छा, बच्चे, ...) must have longer closures than single stops
gpath = DATA / "vaani_districts" / "closures_geminate.parquet"
if gpath.exists():
    gm = pd.read_parquet(gpath)
    gm = gm[gm["reject"] == ""].copy()
    gm["region"] = np.where(gm["state"] == "Haryana", "Haryana", "Delhi")
    both = pd.concat([ok.assign(lexical_geminate=0), gm.assign(lexical_geminate=1)], ignore_index=True)
    both["log_voi"] = np.log(both["voi_ms"])
    both["haryana"] = (both["region"] == "Haryana").astype(int)
    vrows = []
    for (region, gem), g in both.groupby(["region", "lexical_geminate"]):
        ci = cluster_bootstrap(g["voi_ms"].to_numpy(), g["speaker"].to_numpy(), n=1000)
        vrows.append({"region": region, "lexical_geminate": gem, "tokens": len(g), "speakers": g["speaker"].nunique(),
                      "voi_median": g["voi_ms"].median(), "voi_mean": ci["estimate"],
                      "lo": ci["lo"], "hi": ci["hi"]})
    pd.DataFrame(vrows).to_csv(out / "geminate_vs_single.csv", index=False)
    fit = smf.mixedlm("log_voi ~ lexical_geminate * haryana", both, groups=both["speaker"]).fit(method="lbfgs")
    b, (lo, hi) = fit.params["lexical_geminate"], fit.conf_int().loc["lexical_geminate"]
    msg = (f"validity: lexical geminate voiceless interval = x{np.exp(b):.2f} of single (95% CI x{np.exp(lo):.2f} to "
           f"x{np.exp(hi):.2f}) in New Delhi; interaction with Haryana {fit.params['lexical_geminate:haryana']:+.3f} "
           f"(p = {fit.pvalues['lexical_geminate:haryana']:.3g}); tokens {len(both)}")
    with open(out / "mixed_model.txt", "a") as f:
        f.write(msg + "\n" + fit.summary().tables[1].to_string() + "\n")
    print(msg)
    print(pd.DataFrame(vrows).round(1).to_string(index=False))

# --- model doubling inside each token's window
def vocab(model):
    if model == "w2v2":
        v = json.load(open(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/*/vocab.json"))[0]))
        v = next(iter(v.values())) if isinstance(next(iter(v.values())), dict) else v
        id2 = {i: t for t, i in v.items()}
        blank = v["<pad>"]
    else:
        lines_ = open(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--anyspeech--zipa-large-crctc-ns-800k/snapshots/*/tokens.txt"))[0]).read().split("\n")
        id2 = {int(i): t for t, i in (l.rsplit(" ", 1) for l in lines_ if l.strip())}
        blank = 0
    return id2, blank


def window_string(frame_ids, id2, blank, t0, t1, rate=50):
    a, b = max(0, int(t0 * rate)), min(len(frame_ids), int(np.ceil(t1 * rate)))
    seg, prev = [], None
    for f in frame_ids[a:b]:
        if f != blank and f != prev:
            tok = id2.get(int(f), "")
            if not tok.startswith("<") and tok not in ("▁", "|"):
                seg.append(tok)
        prev = f
    return "".join(seg)


rows = []
for model in ("w2v2", "zipa"):
    path = OUTPUTS / "transcripts" / f"{model}_districts.parquet"
    if not path.exists():
        print(f"{model}: district transcripts not available yet")
        continue
    id2, blank = vocab(model)
    tr = pd.read_parquet(path, columns=["audio_path", "frame_ids"]).set_index("audio_path")["frame_ids"]
    s = ok[ok["audio_path"].isin(tr.index)].copy()
    s["window_phones"] = [window_string(tr[p], id2, blank, a - 0.04, b + 0.04)
                          for p, a, b in zip(s["audio_path"], s["stop_start"], s["stop_end"])]
    s["doubled"] = [P.dpr(w)[0] > 0 for w in s["window_phones"]]
    for region, g in s.groupby("region"):
        ci = cluster_bootstrap(g["doubled"].astype(float).to_numpy(), g["speaker"].to_numpy(), n=1000)
        d, n_ = g[g.doubled], g[~g.doubled]
        # AUC: probability that a doubled token has a longer voiceless interval than a single one
        if len(d) and len(n_):
            ranks = pd.concat([d["voi_ms"], n_["voi_ms"]]).rank()
            auc = (ranks.iloc[: len(d)].sum() - len(d) * (len(d) + 1) / 2) / (len(d) * len(n_))
        else:
            auc = np.nan
        rows.append({"model": model, "region": region, "tokens": len(g), "doubled_share": ci["estimate"],
                     "lo": ci["lo"], "hi": ci["hi"], "voi_doubled_median": d["voi_ms"].median(),
                     "voi_single_median": n_["voi_ms"].median(), "auc_voi_predicts_doubling": auc})
    s[["token_id", "audio_path", "word", "stop", "region", "voi_ms", "closure_ms", "window_phones", "doubled"]].to_parquet(
        out / f"model_doubling_tokens_{model}.parquet", index=False)
if rows:
    pd.DataFrame(rows).to_csv(out / "model_doubling.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
print(summary[summary.level == "region"].round(2).to_string(index=False))
print(rej.round(3).to_string())
