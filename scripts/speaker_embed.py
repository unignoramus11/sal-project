"""Speaker embeddings (ECAPA-TDNN, SpeechBrain VoxCeleb model) for every Vaani-Hi utterance.

ECAPA is trained for speaker verification: it encodes voice and channel and is trained to ignore content.
A geolocation probe on these embeddings measures how far speaker and channel information alone go (RQ0).

Output: outputs/features/ecapa_rebuild/{emb.npy (N, 192) float32, index.parquet, meta.json}
Run:    uv run python scripts/speaker_embed.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from speechbrain.inference.speaker import EncoderClassifier

from sal.data.audio import iter_audio
from sal.paths import DATA, OUTPUTS

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet")
ap.add_argument("--out-dir", type=Path, default=OUTPUTS / "features" / "ecapa_rebuild")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=str(OUTPUTS / "cache" / "ecapa"), run_opts={"device": args.device})
model.eval()
args.out_dir.mkdir(parents=True, exist_ok=True)
ids, embs, t0 = [], [], time.time()
with torch.no_grad():
    for k, (mi, wav) in enumerate(iter_audio(args.data)):
        e = model.encode_batch(torch.from_numpy(wav).unsqueeze(0).to(args.device))
        embs.append(e.squeeze().cpu().numpy().astype(np.float32))
        ids.append(mi)
        if (k + 1) % 2000 == 0:
            print(f"{k + 1} utterances, {(k + 1) / (time.time() - t0):.1f} utt/s", flush=True)
np.save(args.out_dir / "emb.npy", np.stack(embs))
pd.DataFrame({"metadata_idx": ids}).to_parquet(args.out_dir / "index.parquet", index=False)
(args.out_dir / "meta.json").write_text(json.dumps({
    "model": "speechbrain/spkrec-ecapa-voxceleb", "dim": int(embs[0].shape[0]), "n": len(ids),
    "seconds": time.time() - t0}, indent=2))
print("done", len(ids))
