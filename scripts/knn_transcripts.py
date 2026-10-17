"""Nearest-neighbour transcript baseline: how much can be done by matching a test transcript to similar
training transcripts, with no learning about location?

TF-IDF over character n-grams (1-4, fitted on train), cosine similarity; the prediction is the location of
the most similar training utterance (k = 1) or the mean direction of the k most similar (k = 5, 25). Under
PRiSM's random split a large advantage over the speaker-disjoint split means that speakers repeat
recognisable content across their utterances (the same picture descriptions, the same wording), which a
transcript probe can memorise. Also reported: share of test utterances whose nearest training transcript
comes from the same speaker.

Output: results/tables/knn_transcripts.csv
Run:    uv run python scripts/knn_transcripts.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from sal.data.prompts import prompt_type
from sal.paths import DATA, OUTPUTS, RESULTS
from sal.probes import geo
from sal.stats import cluster_bootstrap

meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                       columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet"), on="metadata_idx")
rows = []
for model in ("w2v2", "zipa"):
    tr = pd.read_parquet(OUTPUTS / "transcripts" / f"{model}_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
    df = meta.merge(tr, on="metadata_idx").sort_values("metadata_idx").reset_index(drop=True)
    lat, lon = np.radians(df["latitude"].to_numpy()), np.radians(df["longitude"].to_numpy())
    xyz = geo.latlon_to_xyz(torch.tensor(lat), torch.tensor(lon)).numpy()
    targets = np.stack([lat, lon], 1)
    generic = prompt_type(df["metadata_idx"]) == "GENERIC"
    for proto in ("prism", "speaker_split"):
        a = np.flatnonzero(df[proto].to_numpy() == "train")
        b = np.flatnonzero(df[proto].to_numpy() == "test")
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=2)
        xa = vec.fit_transform(df["processed_transcript"].iloc[a])
        xb = vec.transform(df["processed_transcript"].iloc[b])
        sim = (xb @ xa.T).toarray()
        order = np.argsort(-sim, axis=1)[:, :25]
        same_spk = (df["speaker"].to_numpy()[a][order[:, 0]] == df["speaker"].to_numpy()[b]).mean()
        for k in (1, 5, 25):
            pred = xyz[a][order[:, :k]].mean(1)
            t = torch.tensor(targets[b])
            hit = geo.recall_at_k(torch.tensor(pred), t, 1).numpy().astype(float)
            km = geo.error_km(torch.tensor(pred), t).numpy()
            spk = df["speaker"].to_numpy()[b]
            ci = cluster_bootstrap(100 * hit, spk, n=1000)
            g = generic[b]
            rows.append({"transcripts": model, "protocol": proto, "k": k, "recall1": ci["estimate"],
                         "lo": ci["lo"], "hi": ci["hi"], "mean_km": km.mean(),
                         "shared_recall1": 100 * hit[g].mean(), "nn_same_speaker": same_spk if k == 1 else np.nan})
        print(model, proto, "nearest neighbour from the same speaker:", round(same_spk, 3), flush=True)
t = pd.DataFrame(rows)
t.to_csv(RESULTS / "tables" / "knn_transcripts.csv", index=False)
print(t.round(2).to_string(index=False))
