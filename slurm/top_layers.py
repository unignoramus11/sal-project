"""Print the top-k layers of a model by validation mean km in the layer sweep (speaker-disjoint split).
Usage: python slurm/top_layers.py <w2v2|whisper|zipa> [k]   (needs results/layer_sweep/<model>_rebuild_speaker_split.csv)
"""
import sys

import pandas as pd

model, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
df = pd.read_csv(f"results/layer_sweep/{model}_rebuild_speaker_split.csv")
rank = df.groupby("layer")["val_mean_km"].mean().sort_values()
print(" ".join(str(int(l)) for l in sorted(rank.index[:k])))
