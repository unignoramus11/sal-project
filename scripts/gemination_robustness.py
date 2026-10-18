"""Robustness of the Haryana lengthening (RQ2), beyond the pooled mixed model of analyse_gemination.py.

  by_district     each Haryana district against New Delhi: the same mixed model (log voiceless interval ~
                  district + log following vowel + word + speaker intercept)
  by_word         per word with >= 10 tokens in each region: Haryana - New Delhi difference in mean log voiceless
                  interval; the share of words with a positive difference and a sign test
  speaker_level   one value per speaker (mean of word-centred log voiceless interval, i.e. after removing each
                  word's mean), Haryana vs New Delhi speakers, 95% bootstrap CI over speakers; free of the mixed
                  model's distributional assumptions
  meta            per-word Haryana - New Delhi effects (speaker means, bootstrap SEs) pooled by a random-effects
                  meta-analysis, so the effect generalises over words rather than over these words; tau and I2
                  describe how much it varies between words
Output: results/gemination/robustness.txt, robustness_by_word.csv, results/figures/fig_gemination.{pdf,png}
Run:    uv run python scripts/gemination_robustness.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest

from sal.gemination import token_contexts
from sal.paths import DATA, RESULTS

out = RESULTS / "gemination"
df = pd.read_parquet(DATA / "vaani_districts" / "closures.parquet")
df["region"] = np.where(df["state"] == "Haryana", "Haryana", "Delhi")
df["context"] = token_contexts(df["word"], df["stop"])
ok = df[(df["reject"] == "") & df["context"].isin(["vowel", "initial_schwa"])].copy()
ok["log_voi"] = np.log(ok["voi_ms"])
ok["log_vowel"] = np.log(ok["vowel_ms"].clip(lower=20))
ok["haryana"] = (ok["region"] == "Haryana").astype(int)
common = ok["word"].value_counts()
ok = ok[ok["word"].isin(common[common >= 20].index)].copy()
lines = [f"tokens {len(ok)}, speakers {ok['speaker'].nunique()}, words {ok['word'].nunique()}"]

# --- by district (New Delhi as reference)
ok["district_c"] = pd.Categorical(ok["district"], categories=["NewDelhi"] + sorted(set(ok["district"]) - {"NewDelhi"}))
fit = smf.mixedlm("log_voi ~ C(district_c) + log_vowel + C(word)", ok, groups=ok["speaker"]).fit(method="lbfgs")
ci = fit.conf_int()
for k in fit.params.index:
    if k.startswith("C(district_c)"):
        name = k.split("T.")[-1].rstrip("]")
        lines.append(f"district {name} vs NewDelhi: x{np.exp(fit.params[k]):.3f} (95% CI x{np.exp(ci.loc[k, 0]):.3f} to "
                     f"x{np.exp(ci.loc[k, 1]):.3f}), p = {fit.pvalues[k]:.2g}, tokens {int((ok['district'] == name).sum())}")

# --- by word
rows = []
for w, g in ok.groupby("word"):
    h, d = g[g.region == "Haryana"]["log_voi"], g[g.region == "Delhi"]["log_voi"]
    if len(h) >= 10 and len(d) >= 10:
        rows.append({"word": w, "stop": g["stop"].iloc[0], "haryana_tokens": len(h), "delhi_tokens": len(d),
                     "ratio": float(np.exp(h.mean() - d.mean()))})
bw = pd.DataFrame(rows).sort_values("ratio")
bw.to_csv(out / "robustness_by_word.csv", index=False)
npos = int((bw["ratio"] > 1).sum())
lines.append(f"by word: Haryana longer in {npos} of {len(bw)} words (sign test p = {binomtest(npos, len(bw)).pvalue:.2g}); "
             f"median ratio x{bw['ratio'].median():.3f}")

# --- speaker level (word-centred)
ok["centred"] = ok["log_voi"] - ok.groupby("word")["log_voi"].transform("mean")
spk = ok.groupby("speaker").agg(region=("region", "first"), v=("centred", "mean"), n=("centred", "size"))
rng = np.random.default_rng(0)
hv, dv = spk[spk.region == "Haryana"]["v"].to_numpy(), spk[spk.region == "Delhi"]["v"].to_numpy()
boot = [rng.choice(hv, len(hv)).mean() - rng.choice(dv, len(dv)).mean() for _ in range(5000)]
diff = hv.mean() - dv.mean()
lo, hi = np.percentile(boot, [2.5, 97.5])
lines.append(f"speaker level: {len(hv)} Haryana vs {len(dv)} New Delhi speakers, word-centred log voiceless interval "
             f"difference x{np.exp(diff):.3f} (95% bootstrap CI x{np.exp(lo):.3f} to x{np.exp(hi):.3f})")

# --- effect generalised over words: per-word effects pooled by random-effects meta-analysis (DerSimonian-Laird);
# each word's SE from a speaker bootstrap of its Haryana - New Delhi difference in mean log interval
est, var = [], []
for _w, g in ok.groupby("word"):
    h, d = g[g.region == "Haryana"], g[g.region == "Delhi"]
    if h["speaker"].nunique() < 10 or d["speaker"].nunique() < 10:
        continue
    hs, ds = h.groupby("speaker")["log_voi"].mean().to_numpy(), d.groupby("speaker")["log_voi"].mean().to_numpy()
    bt = [rng.choice(hs, len(hs)).mean() - rng.choice(ds, len(ds)).mean() for _ in range(500)]
    est.append(hs.mean() - ds.mean())
    var.append(np.var(bt, ddof=1))
est, var = np.array(est), np.array(var)
wf = 1 / var
fixed = (wf * est).sum() / wf.sum()
q = (wf * (est - fixed) ** 2).sum()
tau2 = max(0.0, (q - (len(est) - 1)) / (wf.sum() - (wf ** 2).sum() / wf.sum()))
wr = 1 / (var + tau2)
pooled, se = (wr * est).sum() / wr.sum(), np.sqrt(1 / wr.sum())
i2 = max(0.0, (q - (len(est) - 1)) / q) if q > 0 else 0.0
lines.append(f"random-effects meta-analysis over {len(est)} words (speaker means, >= 10 speakers per region): "
             f"x{np.exp(pooled):.3f} (95% CI x{np.exp(pooled - 1.96 * se):.3f} to x{np.exp(pooled + 1.96 * se):.3f}); "
             f"between-word tau = {np.sqrt(tau2):.3f} on the log scale, I2 = {100 * i2:.0f}%")

(out / "robustness.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# --- figure: per-word ratios (left) and speaker-level distributions (right)
blue, orange, ink2, grid = "#2a78d6", "#eb6834", "#52514e", "#e3e3e0"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"], "font.size": 8})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
a1.scatter(range(len(bw)), bw["ratio"], s=10, color=blue, zorder=3)
a1.axhline(1, color=ink2, lw=0.8, ls="--")
a1.set_xlabel(f"word ({len(bw)} with >= 10 tokens per region, sorted)")
a1.set_ylabel("Haryana / New Delhi\nvoiceless interval")
a1.set_xticks([])
bins = np.linspace(-0.8, 0.8, 33)
a2.hist(dv, bins=bins, density=True, alpha=0.6, color=orange, label=f"New Delhi ({len(dv)})")
a2.hist(hv, bins=bins, density=True, alpha=0.6, color=blue, label=f"Haryana ({len(hv)})")
a2.set_xlabel("speaker mean, word-centred log interval")
a2.legend(frameon=False, fontsize=6.5)
for a in (a1, a2):
    a.yaxis.grid(True, color=grid, lw=0.5)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(RESULTS / "figures" / f"fig_gemination.{ext}", dpi=220, transparent=True)
