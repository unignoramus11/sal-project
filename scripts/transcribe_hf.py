"""Frame-level W2V2 phone recognition through Hugging Face transformers alone (no PRiSM stack; Mac MPS).

Same model and preprocessing as PRiSM's W2V2 runner (scripts/transcribe.py): on 200 Vaani-Hi utterances the
per-frame argmax ids agree on 100% of frames (199/200 utterances identical). Writes the columns that the
gemination analysis needs: the utterance key, per-frame argmax ids, the collapsed token string.

Output: --out parquet (id column, frame_ids, n_frames, tokens)
Run:    uv run python scripts/transcribe_hf.py --data data/vaani_districts/candidates_audio.parquet \
            --id-col audio_path --out outputs/transcripts/w2v2_districts.parquet --device mps
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

REPO = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--id-col", default="metadata_idx")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(REPO)
    model = Wav2Vec2ForCTC.from_pretrained(REPO).to(args.device).eval()
    vocab = {i: t for t, i in __import__("transformers").Wav2Vec2CTCTokenizer.from_pretrained(REPO).get_vocab().items()}
    blank = model.config.pad_token_id
    rows, t0 = [], time.time()
    pf = pq.ParquetFile(args.data)
    for batch in pf.iter_batches(batch_size=64, columns=[args.id_col, "audio"]):
        for key, audio in zip(batch.column(args.id_col).to_pylist(), batch.column("audio").to_pylist(), strict=True):
            wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
            x = fe(wav, sampling_rate=16000, return_tensors="pt").input_values.to(args.device)
            with torch.no_grad():
                ids = model(x).logits[0].argmax(-1).cpu().numpy()
            keep = [int(i) for k, i in enumerate(ids) if i != blank and (k == 0 or i != ids[k - 1])]
            rows.append({args.id_col: key, "frame_ids": ids.astype(np.int32).tolist(), "n_frames": len(ids),
                         "tokens": [vocab[i] for i in keep]})
            if len(rows) % 1000 == 0:
                print(f"{len(rows)} utterances, {len(rows) / (time.time() - t0):.1f} utt/s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"wrote {args.out}: {len(rows)} utterances")


if __name__ == "__main__":
    main()
