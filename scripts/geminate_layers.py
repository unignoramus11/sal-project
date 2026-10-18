"""Where do the encoders represent consonant length? Layer-wise geminate probe (RQ2).

Two modes.

--export-tokens   builds results/gemination/layer_probe_tokens.parquet from the closure measurements: accepted
                  single voiceless stops between vowels in speech (data/vaani_districts/closures.parquet, main
                  contexts of sal.gemination) and accepted lexical geminates (closures_geminate.parquet), each
                  with its measured span (closure start to voice onset). Small and versioned, so Ada or the PC
                  can run scripts/token_frames.py without the Mac's data.

--model M         reads outputs/features/geminate_M/ (scripts/token_frames.py) and, for every layer, trains an
                  L2 logistic regression (standardised features, balanced class weights) for geminate vs single
                  with 5-fold cross-validation grouped by speaker. Reported per layer:
                    auc            out-of-fold ROC AUC over all tokens, with a 95% speaker-bootstrap CI
                    auc_within     mean AUC within place of articulation (velar, palatal, retroflex, dental,
                                   labial; places with >= 20 tokens of each class), so that the probe cannot
                                   score by recognising which consonant it is (geminates and singles differ in
                                   stop mix)
                    haryana_logit  dialect test on single stops only: out-of-fold logit P(geminate) ~ Haryana +
                                   word + speaker random intercept; > 0 means Haryana single stops look more
                                   geminate-like to the probe than New Delhi ones (95% CI, p)
                    haryana_given_length  the same, holding the token's measured voiceless interval constant
                                   (does the encoder carry more than the measured lengthening?)
Controls: --groups word (word-disjoint folds: the probe cannot score by recognising words) and --contrast
aspiration (aspirated vs plain single stops: a Haryana shift along an unrelated direction would mean a general
regional offset rather than lengthening).
Output: results/gemination/geminate_layers_<model>[_<contrast>_<groups>].csv and
        results/figures/fig_geminate_layers_<model>.{pdf,png}
Run:    uv run python scripts/geminate_layers.py --export-tokens
        uv run python scripts/geminate_layers.py --model w2v2
"""
import argparse
import json

import numpy as np
import pandas as pd

from sal.gemination import token_contexts
from sal.paths import DATA, OUTPUTS, RESULTS

PLACE = {"क": "velar", "ख": "velar", "च": "palatal", "छ": "palatal", "ट": "retroflex", "ठ": "retroflex",
         "त": "dental", "थ": "dental", "प": "labial"}
TOKENS = RESULTS / "gemination" / "layer_probe_tokens.parquet"


def export_tokens():
    cols = ["token_id", "audio_path", "speaker", "state", "district", "word", "stop", "closure_start",
            "voice_onset", "closure_ms", "reject"]
    s = pd.read_parquet(DATA / "vaani_districts" / "closures.parquet")
    s["context"] = token_contexts(s["word"], s["stop"])
    s = s[(s["reject"] == "") & s["context"].isin(["vowel", "initial_schwa"])][cols].assign(geminate=False)
    g = pd.read_parquet(DATA / "vaani_districts" / "closures_geminate.parquet")
    g = g[g["reject"] == ""][cols].assign(geminate=True)
    t = pd.concat([s, g], ignore_index=True)
    t["token_key"] = np.where(t["geminate"], "g", "s") + t["token_id"].astype(str)
    t["region"] = np.where(t["state"] == "Haryana", "Haryana", "Delhi")
    t["place"] = t["stop"].map(PLACE)
    t = t.rename(columns={"closure_start": "span_start", "voice_onset": "span_end"}).drop(columns=["reject"])
    t = t[t["span_end"] > t["span_start"]]
    t.to_parquet(TOKENS, index=False)
    print(t.groupby(["geminate", "region", "place"]).size().unstack(fill_value=0).to_string())
    print(f"wrote {TOKENS}: {len(t)} tokens")


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def speaker_ci(y, p, spk, n=300, seed=0):
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(y))).groupby(spk).indices
    keys = list(groups)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        vals.append(auc(y[idx], p[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])


def probe_layers(model: str, contrast: str = "geminate", groups: str = "speaker"):
    import statsmodels.formula.api as smf
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = OUTPUTS / "features" / f"geminate_{model}"
    X_all = np.load(d / "tokens.npy", mmap_mode="r")
    tok = pd.read_parquet(d / "tokens.parquet")
    meta = json.loads((d / "meta.json").read_text())
    tok["log_span"] = np.log(1000 * (tok["span_end"] - tok["span_start"]).clip(lower=0.005))
    if contrast == "aspiration":                 # control direction: aspirated vs plain, single stops only
        keep = ~tok["geminate"].to_numpy()
        X_all, tok = X_all[np.flatnonzero(keep)], tok[keep].reset_index(drop=True)
        tok["target"] = tok["stop"].isin(list("खछठथ"))
    else:
        tok["target"] = tok["geminate"]
    y = tok["target"].to_numpy().astype(int)
    spk = tok["speaker"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, tok[groups].to_numpy()))
    rows = []
    for layer, (name, w) in enumerate(zip(meta["layers"], meta["widths"], strict=True)):
        X = np.asarray(X_all[:, layer, :w], dtype=np.float32)
        p = np.zeros(len(y))
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        lo, hi = speaker_ci(y, p, spk)
        within = []
        for _place, g in tok.groupby("place"):
            yy = y[g.index]
            if min(yy.sum(), len(yy) - yy.sum()) >= 20:
                within.append(auc(yy, p[g.index]))
        sel = ~tok["geminate"].to_numpy()
        pc = p[sel].clip(1e-4, 1 - 1e-4)
        single = tok[sel].assign(logit=np.log(pc) - np.log1p(-pc), haryana=lambda t: (t["region"] == "Haryana").astype(int))
        common = single["word"].value_counts()
        single = single[single["word"].isin(common[common >= 20].index)]
        fit = fit_mixed(smf, "logit ~ haryana + C(word)", single)
        ci = fit.conf_int().loc["haryana"]
        fit2 = fit_mixed(smf, "logit ~ haryana + log_span + C(word)", single)   # holding measured length constant
        ci2 = fit2.conf_int().loc["haryana"]
        rows.append({"layer": layer, "name": name, "auc": auc(y, p), "auc_lo": lo, "auc_hi": hi,
                     "auc_within": float(np.nanmean(within)) if within else np.nan, "places": len(within),
                     "haryana_logit": fit.params["haryana"], "haryana_lo": ci[0], "haryana_hi": ci[1],
                     "haryana_p": fit.pvalues["haryana"], "haryana_given_length": fit2.params["haryana"],
                     "haryana_given_length_lo": ci2[0], "haryana_given_length_hi": ci2[1],
                     "converged": bool(fit.converged and fit2.converged), "tokens": len(y), "positives": int(y.sum())})
        print(f"{model} layer {layer:2d} {name:9s} AUC {rows[-1]['auc']:.3f} [{lo:.3f}, {hi:.3f}] within-place "
              f"{rows[-1]['auc_within']:.3f}  Haryana logit {fit.params['haryana']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]",
              flush=True)
    out = pd.DataFrame(rows)
    tag = model if (contrast, groups) == ("geminate", "speaker") else f"{model}_{contrast}_{groups}"
    out.to_csv(RESULTS / "gemination" / f"geminate_layers_{tag}.csv", index=False)
    if tag == model:
        figure(out, model)


def fit_mixed(smf, formula, data):
    """Mixed model with a speaker random intercept; falls back to Powell if L-BFGS does not converge."""
    fit = smf.mixedlm(formula, data, groups=data["speaker"]).fit(method="lbfgs")
    if not fit.converged:
        fit = smf.mixedlm(formula, data, groups=data["speaker"]).fit(method="powell", maxiter=3000)
    return fit


def figure(t, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink2, grid = "#2a78d6", "#eb6834", "#52514e", "#e3e3e0"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 8})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.3))
    a1.fill_between(t.layer, t.auc_lo, t.auc_hi, color=blue, alpha=0.2, lw=0)
    a1.plot(t.layer, t.auc, "-o", ms=3, color=blue, label="all tokens")
    a1.plot(t.layer, t.auc_within, "--s", ms=3, color=orange, label="within place of articulation")
    a1.set_ylabel("ROC AUC, geminate vs single")
    a1.legend(frameon=False, fontsize=6.5)
    a2.errorbar(t.layer, t.haryana_logit, yerr=[t.haryana_logit - t.haryana_lo, t.haryana_hi - t.haryana_logit],
                fmt="o", ms=3, color=blue, elinewidth=1, capsize=0)
    a2.axhline(0, color=ink2, lw=0.8, ls="--")
    a2.set_ylabel("Haryana - Delhi, logit P(geminate)\nfor single stops")
    for a in (a1, a2):
        a.set_xlabel(f"layer ({model.upper()})")
        a.yaxis.grid(True, color=grid, lw=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / "figures" / f"fig_geminate_layers_{model}.{ext}", dpi=220, transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-tokens", action="store_true")
    ap.add_argument("--model", choices=["w2v2", "whisper", "zipa"])
    ap.add_argument("--contrast", choices=["geminate", "aspiration"], default="geminate",
                    help="aspiration: control direction (aspirated vs plain single stops)")
    ap.add_argument("--groups", choices=["speaker", "word"], default="speaker",
                    help="cross-validation groups; word: test words are never seen in training")
    args = ap.parse_args()
    if args.export_tokens:
        export_tokens()
    if args.model:
        probe_layers(args.model, args.contrast, args.groups)


if __name__ == "__main__":
    main()
