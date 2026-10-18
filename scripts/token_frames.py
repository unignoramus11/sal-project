"""Hidden states of every layer over each gemination token (input to scripts/geminate_layers.py).

Tokens come from results/gemination/layer_probe_tokens.parquet (scripts/geminate_layers.py --export-tokens):
accepted single stops between vowels and accepted lexical geminates, each with the measured span from closure
start to voice onset. For every utterance the model runs once (PRiSM preprocessing, extract_features.py
wrappers); each token's vector at each layer is the mean of the frames whose centres fall inside its span
(at least one frame).

Output: outputs/features/geminate_<model>/tokens.npy float16 (n_tokens, L, Dmax), tokens.parquet (row order),
        meta.json (layer names, widths).
Run (envs/prism, GPU): python scripts/token_frames.py --model w2v2
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_features import W2V2, Whisper, Zipa  # noqa: E402

MODELS = {"w2v2": W2V2, "whisper": Whisper, "zipa": Zipa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--tokens", type=Path, default=ROOT / "results" / "gemination" / "layer_probe_tokens.parquet")
    ap.add_argument("--audio", type=Path, default=ROOT / "data" / "vaani_districts" / "candidates_audio.parquet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="utterances (for a quick test)")
    args = ap.parse_args()

    tok = pd.read_parquet(args.tokens).reset_index(drop=True)
    by_utt = tok.groupby("audio_path").indices
    model = MODELS[args.model](args.device)
    out_dir = ROOT / "outputs" / "features" / f"geminate_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vecs, widths, done, t0 = None, None, 0, time.time()
    pf = pq.ParquetFile(args.audio)
    for batch in pf.iter_batches(batch_size=64, columns=["audio_path", "audio"]):
        for path, audio in zip(batch.column("audio_path").to_pylist(), batch.column("audio").to_pylist(), strict=True):
            if path not in by_utt:
                continue
            wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
            states = model(wav)                           # list of (T, D_l) tensors
            if vecs is None:
                widths = [int(h.shape[-1]) for h in states]
                vecs = np.zeros((len(tok), len(states), max(widths)), dtype=np.float16)
            T = int(states[0].shape[0])
            centres = (np.arange(T) + 0.5) / model.rate
            for i in by_utt[path]:
                a, b = tok.at[i, "span_start"], tok.at[i, "span_end"]
                sel = np.flatnonzero((centres >= a) & (centres <= b))
                if len(sel) == 0:
                    sel = np.array([min(T - 1, int((a + b) / 2 * model.rate))])
                for li, h in enumerate(states):
                    vecs[i, li, : widths[li]] = h[sel].float().mean(0).cpu().numpy()
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(by_utt)} utterances, {done / (time.time() - t0):.1f} utt/s", flush=True)
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break
    np.save(out_dir / "tokens.npy", vecs)
    tok.to_parquet(out_dir / "tokens.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps({"model": args.model, "layers": model.names, "widths": widths,
                                                   "utterances": done, "tokens": len(tok)}, indent=1))
    print(f"wrote {out_dir}: {len(tok)} tokens from {done} utterances")


if __name__ == "__main__":
    main()
