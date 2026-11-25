"""RQ2 channel check: is the Haryana lengthening of voiceless stops a measurement artifact of recording conditions?

On the benchmark, recording-channel features alone locate new speakers above chance (scripts/benchmark_checks.py), so
Haryana and New Delhi recordings may differ in noise and level. The closure landmarks are found automatically
(voicing offset, burst), and noisier audio could move them. This script measures each district utterance's channel
(sal.channel: noise floor, speech level, their difference as a signal-to-noise proxy) and asks whether the Haryana
effect survives it:
  1. channel by region (utterances that contribute tokens; medians and quartiles);
  2. the primary mixed model of analyse_gemination.py, refitted unchanged (it must reproduce the reported effect),
     then with the utterance's SNR and noise floor (z-scored) as covariates;
  3. the Haryana effect within each third and each fifth of the SNR distribution (with token counts per region),
     and a model with a Haryana x SNR interaction;
  4. how the voiceless interval changes with SNR within each region on its own (the same dialect at different
     noise levels: a slope there is the measurement responding to noise);
  5. the validity check (lexical geminates against single stops, as in analyse_gemination.py) within each SNR
     third: if geminates still measure much longer than single stops in clean audio, the method sees length there.
A lengthening that comes from the measurement would shrink with the covariates and change across the SNR bands.

Output: results/gemination/channel_check.txt, channel_by_region.csv, channel_bands.csv (one row per SNR band or
        model: Haryana effect with CI and tokens per region; geminate/single ratios per region for the thirds)
        (cache: data/vaani_districts/channel.parquet)
Run:    uv run python scripts/gemination_channel.py
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

from sal.channel import NAMES, features, read_wav
from sal.gemination import token_contexts
from sal.paths import DATA, RESULTS

AUDIO = DATA / "vaani_districts" / "candidates_audio.parquet"
CACHE = DATA / "vaani_districts" / "channel.parquet"
OUT = RESULTS / "gemination"
KEEP = ["noise_floor_db", "speech_db", "snr_db", "clip_share"]


def batch_features(paths_audio):
    paths, audio = paths_audio
    x = np.stack([features(read_wav(a)) for a in audio])
    return pd.DataFrame(x[:, [NAMES.index(k) for k in KEEP]], columns=KEEP).assign(audio_path=paths)


def channel_table() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    parts, pending = [], []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for b in pq.ParquetFile(AUDIO).iter_batches(batch_size=400, columns=["audio_path", "audio"]):
            d = b.to_pydict()
            pending.append(ex.submit(batch_features, (d["audio_path"], d["audio"])))
            if len(pending) >= 16:                     # bounded: the file is 4.2 GB
                parts += [f.result() for f in pending]
                pending = []
        parts += [f.result() for f in pending]
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(CACHE, index=False)
    return t


BANDS = []          # rows of channel_bands.csv


def fit(formula, data, term="haryana", band=None):
    f = smf.mixedlm(formula, data, groups=data["speaker"]).fit(method="lbfgs")
    b, (lo, hi), p = f.params[term], f.conf_int().loc[term], f.pvalues[term]
    if band is not None:
        n = data["region"].value_counts()
        BANDS.append({**band, "effect": np.exp(b), "lo": np.exp(lo), "hi": np.exp(hi), "p": p, "tokens": len(data),
                      "tokens_haryana": int(n.get("Haryana", 0)), "tokens_delhi": int(n.get("Delhi", 0)),
                      "speakers": data["speaker"].nunique(), "snr_min": data["snr_db"].min(), "snr_max": data["snr_db"].max()})
    return f"x{np.exp(b):.3f} (95% CI x{np.exp(lo):.3f} to x{np.exp(hi):.3f}), p = {p:.3g}; tokens {len(data)}, " \
           f"speakers {data['speaker'].nunique()}"


def main():
    ch = channel_table()
    df = pd.read_parquet(DATA / "vaani_districts" / "closures.parquet")
    df["region"] = np.where(df["state"] == "Haryana", "Haryana", "Delhi")
    df["context"] = token_contexts(df["word"], df["stop"])
    ok = df[(df["reject"] == "") & df["context"].isin(["vowel", "initial_schwa"])].copy()
    common = ok["word"].value_counts()
    ok = ok[ok["word"].isin(common[common >= 20].index)].copy()        # as analyse_gemination.frequent
    ok["haryana"] = (ok["region"] == "Haryana").astype(int)
    ok["log_voi"] = np.log(ok["voi_ms"])
    ok["log_vowel"] = np.log(ok["vowel_ms"].clip(lower=20))
    ok = ok.merge(ch, on="audio_path", how="left")
    missing = ok["snr_db"].isna().sum()
    assert missing == 0, f"{missing} tokens without channel features"
    for c in ("snr_db", "noise_floor_db"):
        ok[c + "_z"] = (ok[c] - ok[c].mean()) / ok[c].std()

    utt = ok.drop_duplicates("audio_path")
    by_region = utt.groupby("region")[KEEP].describe().T.unstack()
    by_region.to_csv(OUT / "channel_by_region.csv")
    lines = ["channel of the utterances that contribute tokens (median [quartiles])"]
    for c in ("snr_db", "noise_floor_db", "speech_db"):
        lines.append("  " + c + ": " + ", ".join(
            f"{r} {utt.loc[utt.region == r, c].median():.1f} [{utt.loc[utt.region == r, c].quantile(.25):.1f}, "
            f"{utt.loc[utt.region == r, c].quantile(.75):.1f}]" for r in ("Haryana", "Delhi")))
    lines.append(f"  utterances: Haryana {int((utt.region == 'Haryana').sum())}, Delhi {int((utt.region == 'Delhi').sum())}")
    base = "log_voi ~ haryana + log_vowel + C(word)"
    lines.append("\nHaryana effect on the voiceless interval")
    lines.append("  primary model (must reproduce mixed_model.txt): " + fit(base, ok, band={"kind": "model", "band": "primary"}))
    lines.append("  + SNR and noise floor: " + fit(base + " + snr_db_z + noise_floor_db_z", ok,
                                                   band={"kind": "model", "band": "snr covariates"}))
    ok["snr_third"] = pd.qcut(ok["snr_db"], 3, labels=["low", "middle", "high"])
    for t in ("low", "middle", "high"):
        s = ok[ok["snr_third"] == t]
        lines.append(f"  SNR {t} third ({s['snr_db'].min():.0f} to {s['snr_db'].max():.0f} dB): "
                     + fit(base, s, band={"kind": "third", "band": t}))
    ok["snr_fifth"] = pd.qcut(ok["snr_db"], 5, labels=False)
    for q in range(5):
        s = ok[ok["snr_fifth"] == q]
        n = s["region"].value_counts()
        lines.append(f"  SNR fifth {q + 1} ({s['snr_db'].min():.0f} to {s['snr_db'].max():.0f} dB; tokens Haryana "
                     f"{n.get('Haryana', 0)}, Delhi {n.get('Delhi', 0)}): " + fit(base, s, band={"kind": "fifth", "band": q + 1}))
    f = smf.mixedlm(base + " + snr_db_z + haryana:snr_db_z", ok, groups=ok["speaker"]).fit(method="lbfgs")
    lines.append(f"  interaction model: Haryana x{np.exp(f.params['haryana']):.3f} at the mean SNR; Haryana x SNR "
                 f"{f.params['haryana:snr_db_z']:+.3f} per SD (p = {f.pvalues['haryana:snr_db_z']:.3g}); SNR "
                 f"{f.params['snr_db_z']:+.3f} per SD in Delhi (p = {f.pvalues['snr_db_z']:.3g})")

    lines.append("\nvoiceless interval against SNR within each region (log scale, per SD of SNR)")
    for r in ("Haryana", "Delhi"):
        lines.append(f"  {r}: " + fit("log_voi ~ snr_db_z + log_vowel + C(word)", ok[ok["region"] == r], term="snr_db_z",
                                      band={"kind": "slope", "band": r}).replace("x", "slope x", 1))

    gm = pd.read_parquet(DATA / "vaani_districts" / "closures_geminate.parquet")
    gm = gm[gm["reject"] == ""].merge(ch, on="audio_path", how="inner")
    gm["region"] = np.where(gm["state"] == "Haryana", "Haryana", "Delhi")
    both = pd.concat([ok.drop(columns=["snr_third", "snr_fifth"]).assign(lexical_geminate=0),
                      gm.assign(lexical_geminate=1)], ignore_index=True)
    both["log_voi"] = np.log(both["voi_ms"])
    both["haryana"] = (both["region"] == "Haryana").astype(int)
    edges = ok["snr_db"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    both["snr_third"] = pd.cut(both["snr_db"], [-np.inf, edges[1], edges[2], np.inf], labels=["low", "middle", "high"])
    lines.append(f"\nvalidity check by SNR third (lexical geminates {len(gm)} tokens with channel features)")
    for t in ("low", "middle", "high"):
        s = both[both["snr_third"] == t]
        g = smf.mixedlm("log_voi ~ lexical_geminate * haryana", s, groups=s["speaker"]).fit(method="lbfgs")
        BANDS.append({"kind": "geminate", "band": t, "effect": np.exp(g.params["lexical_geminate"]),
                      "effect_haryana": np.exp(g.params["lexical_geminate"] + g.params["lexical_geminate:haryana"]),
                      "tokens": int(s["lexical_geminate"].sum())})
        lines.append(f"  SNR {t}: geminate x{np.exp(g.params['lexical_geminate']):.2f} of single in Delhi, "
                     f"x{np.exp(g.params['lexical_geminate'] + g.params['lexical_geminate:haryana']):.2f} in Haryana; "
                     f"geminate tokens {int(s['lexical_geminate'].sum())}")
    text = "\n".join(lines)
    (OUT / "channel_check.txt").write_text(text + "\n")
    pd.DataFrame(BANDS).to_csv(OUT / "channel_bands.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
