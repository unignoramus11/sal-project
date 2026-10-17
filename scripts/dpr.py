"""Doubled-phone rate (DPR) by state and district, with speaker-cluster bootstrap intervals.

DPR = geminate consonants / consonants (sal.phones: a length mark, or a run identical up to aspiration,
counted once). Computed on all 22,551 utterances (it is a property of the recognizer output, not of a
trained probe, so no split is involved).

Outputs (results/dpr/):
  <model>_utterances.parquet   metadata_idx, speaker, state, district, geminates, consonants
  <model>_by_state.csv          DPR (ratio of sums) with 95% CI over speakers; n speakers, n utterances
  <model>_by_district.csv
  <model>_haryana_vs_rest.csv   difference Haryana - other states, bootstrap CI and p-value
  <model>_geminate_types.csv    per consonant: share doubled in Haryana vs elsewhere
Run: uv run python scripts/dpr.py outputs/transcripts/w2v2_rebuild.parquet outputs/transcripts/zipa_rebuild.parquet
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sal import phones as P
from sal.paths import DATA, RESULTS
from sal.stats import _groups, cluster_bootstrap

meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")[["metadata_idx", "speaker", "state", "district"]]
out = RESULTS / "dpr"
out.mkdir(parents=True, exist_ok=True)


def ratio(v: np.ndarray) -> float:
    return float(v[:, 0].sum() / max(v[:, 1].sum(), 1))


for path in map(Path, sys.argv[1:]):
    model = path.stem.replace("_rebuild", "")
    tr = pd.read_parquet(path, columns=["metadata_idx", "processed_transcript"])
    df = meta.merge(tr, on="metadata_idx")
    ev = [P.consonant_events(P.segment(t)) for t in df["processed_transcript"]]
    df["geminates"] = [sum(g for _, g in e) for e in ev]
    df["consonants"] = [len(e) for e in ev]
    df.drop(columns="processed_transcript").to_parquet(out / f"{model}_utterances.parquet", index=False)

    def table(by):
        rows = []
        for key, g in df.groupby(by):
            v = g[["geminates", "consonants"]].to_numpy(float)
            ci = cluster_bootstrap(v, g["speaker"].to_numpy(), stat=ratio, n=1000, seed=0)
            rows.append({**dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,))),
                         "dpr": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"], "speakers": ci["n_clusters"],
                         "utterances": len(g), "consonants": int(g["consonants"].sum())})
        return pd.DataFrame(rows).sort_values("dpr", ascending=False)

    by_state = table("state")
    by_state.to_csv(out / f"{model}_by_state.csv", index=False)
    table(["state", "district"]).to_csv(out / f"{model}_by_district.csv", index=False)

    # Haryana minus the rest: resample speakers independently within each group
    rng = np.random.default_rng(0)
    groups = {}
    for name, mask in (("haryana", df["state"] == "Haryana"), ("rest", df["state"] != "Haryana")):
        g = df[mask]
        v = g[["geminates", "consonants"]].to_numpy(float)
        groups[name] = (v, _groups(g["speaker"].to_numpy()))
    diffs = []
    for _ in range(2000):
        r = {}
        for name, (v, gr) in groups.items():
            pick = rng.integers(0, len(gr), len(gr))
            r[name] = ratio(v[np.concatenate([gr[i] for i in pick])])
        diffs.append(r["haryana"] - r["rest"])
    diffs = np.array(diffs)
    point = ratio(groups["haryana"][0]) - ratio(groups["rest"][0])
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    pd.DataFrame([{"diff": point, "lo": np.quantile(diffs, 0.025), "hi": np.quantile(diffs, 0.975), "p": p,
                   "haryana": ratio(groups["haryana"][0]), "rest": ratio(groups["rest"][0])}]).to_csv(
        out / f"{model}_haryana_vs_rest.csv", index=False)

    # which consonants are doubled
    cnt = {"Haryana": Counter(), "rest": Counter()}
    tot = {"Haryana": Counter(), "rest": Counter()}
    for e, st in zip(ev, df["state"]):
        k = "Haryana" if st == "Haryana" else "rest"
        for c, g in e:
            s = P.stem(c).replace("ː", "")
            tot[k][s] += 1
            cnt[k][s] += g
    types = pd.DataFrame([{"consonant": c, "haryana_doubled": cnt["Haryana"][c], "haryana_total": tot["Haryana"][c],
                           "rest_doubled": cnt["rest"][c], "rest_total": tot["rest"][c]}
                          for c in set(tot["Haryana"]) | set(tot["rest"])])
    types["haryana_rate"] = types.haryana_doubled / types.haryana_total.clip(lower=1)
    types["rest_rate"] = types.rest_doubled / types.rest_total.clip(lower=1)
    types.sort_values("haryana_doubled", ascending=False).to_csv(out / f"{model}_geminate_types.csv", index=False)

    print(f"== {model}")
    print(by_state.round(4).to_string(index=False))
    print(pd.read_csv(out / f"{model}_haryana_vs_rest.csv").round(4).to_string(index=False))
