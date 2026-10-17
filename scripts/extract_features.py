"""Extract layer-wise hidden states for representation probes (PRiSM preprocessing per model).

Outputs in --out-dir:
  pooled.npy          float32 (N, L, Dmax): mean over valid frames of every layer
  index.parquet       metadata_idx, n_frames, offset (frame offset into the frames_* arrays)
  frames_L{l}.npy     float16 (total_frames, D) for each layer in --frame-layers
  meta.json           model, layer names and widths, frame rate, preprocessing notes

Models and layers:
  w2v2     facebook/wav2vec2-lv-60-espeak-cv-ft via PRiSM's Wav2Vec2PhonemeModel (processor normalisation,
           attention mask); 25 states (CNN/projection output + 24 transformer layers), 1024-d, 50 Hz.
  whisper  openai/whisper-small encoder; log-mel padded to 30 s as in PRiSM, output trimmed to
           ceil(samples / 320) frames; 13 states (embedding + 12 layers, last one after the final LN).
  zipa     anyspeech/zipa-large-crctc-ns-800k via PRiSM's ZipaCtcModel (lhotse fbank); 8 states:
           encoder_embed output, the 6 Zipformer stacks, and the combined encoder output. Stacks have
           different widths and are zero-padded to the widest, as Zipformer's own output combiner does.

Runs in envs/prism (PRiSM's pinned torch/k2). One utterance per forward pass, as PRiSM does.
Run: python scripts/extract_features.py --model w2v2 --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
        --out-dir outputs/features/w2v2_rebuild --frame-layers -1
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "prism"))
sys.path.insert(0, str(ROOT / "src"))
from sal.data.audio import iter_audio  # noqa: E402


class W2V2:
    rate = 50

    def __init__(self, device):
        from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_model
        self.m = build_wav2vec2phoneme_model(hf_repo="facebook/wav2vec2-lv-60-espeak-cv-ft").to(device).eval()
        self.device = device
        self.names = ["cnn_proj"] + [f"layer{i}" for i in range(1, 25)]

    @torch.no_grad()
    def __call__(self, wav):
        from src.model.wav2vec2phoneme.wav2vec2phoneme_model import preprocess_inputs_wav2vec2
        x = torch.from_numpy(wav).unsqueeze(0)
        inputs = preprocess_inputs_wav2vec2(self.m.processor, x, torch.tensor([x.shape[1]]), device=self.device)
        out = self.m.model(**inputs, output_hidden_states=True)
        return [h[0] for h in out.hidden_states]


class Whisper:
    rate = 50

    def __init__(self, device):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.enc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(device).eval()
        self.device = device
        self.names = ["embed"] + [f"layer{i}" for i in range(1, 13)]

    @torch.no_grad()
    def __call__(self, wav):
        feats = self.fe(wav, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)  # padded to 30 s
        out = self.enc(feats, output_hidden_states=True)
        n = min(math.ceil(len(wav) / 320), 1500)
        return [h[0, :n] for h in out.hidden_states]


class Zipa:
    rate = 50

    def __init__(self, device):
        from src.model.zipa.zipformer_crctc.builders import build_zipactc_model
        self.m = build_zipactc_model(work_dir=str(ROOT / "outputs" / "cache" / "zipactc"),
                                     hf_repo="anyspeech/zipa-large-crctc-ns-800k").to(device).eval()
        self.device = device
        enc = self.m.encoder.encoder
        self.names = ["embed"] + [f"stack{i}" for i in range(1, len(enc.encoders) + 1)] + ["combined"]
        self._stack_out = []
        for mod in enc.encoders:
            mod.register_forward_hook(lambda _m, _i, o: self._stack_out.append(o))
        self._embed_out = []
        self.m.encoder.encoder_embed.register_forward_hook(lambda _m, _i, o: self._embed_out.append(o))

    @torch.no_grad()
    def __call__(self, wav):
        self._stack_out.clear()
        self._embed_out.clear()
        x = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        enc_out, enc_lens = self.m.encode(x, torch.tensor([x.shape[1]], device=self.device))
        n = int(enc_lens[0])
        emb = self._embed_out[0][0][0]                              # (T, D) after Conv2dSubsampling
        stacks = [s[:, 0] for s in self._stack_out]                 # (T, B, D_i) -> (T, D_i)
        return [emb[:n]] + [s[:n] for s in stacks] + [enc_out[0, :n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["w2v2", "whisper", "zipa"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frame-layers", type=int, nargs="*", default=[-1])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = {"w2v2": W2V2, "whisper": Whisper, "zipa": Zipa}[args.model](args.device)
    ids = pd.read_parquet(args.data, columns=["metadata_idx", "audio_seconds"])
    n_utt = len(ids) if args.limit is None else min(args.limit, len(ids))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # probe one utterance for shapes
    first = model(next(iter_audio(args.data))[1])
    widths = [h.shape[-1] for h in first]
    L, dmax = len(first), max(widths)
    frame_layers = sorted({fl % L for fl in args.frame_layers})
    est_frames = int(np.ceil(ids["audio_seconds"].to_numpy()[:n_utt] * model.rate).sum() + 2 * n_utt)

    pooled = np.lib.format.open_memmap(args.out_dir / "pooled.npy", mode="w+", dtype=np.float32, shape=(n_utt, L, dmax))
    frames = {l: np.lib.format.open_memmap(args.out_dir / f"frames_L{l}.npy", mode="w+", dtype=np.float16,
                                           shape=(est_frames, widths[l])) for l in frame_layers}
    rows, offset, t0 = [], 0, time.time()
    for k, (mi, wav) in enumerate(iter_audio(args.data)):
        if k >= n_utt:
            break
        hs = model(wav)
        n = hs[0].shape[0]
        for l, h in enumerate(hs):
            assert h.shape[0] == n, (l, h.shape, n)
            pooled[k, l, : h.shape[-1]] = h.float().mean(0).cpu().numpy()
        for l in frame_layers:
            frames[l][offset: offset + n] = hs[l].half().cpu().numpy()
        rows.append({"metadata_idx": mi, "n_frames": n, "offset": offset})
        offset += n
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{n_utt} utterances, {(k + 1) / (time.time() - t0):.1f} utt/s", flush=True)
    pooled.flush()
    for l, f in frames.items():
        f.flush()
    pd.DataFrame(rows).to_parquet(args.out_dir / "index.parquet", index=False)
    meta = {"model": args.model, "layers": model.names, "widths": widths, "frame_rate": model.rate,
            "frame_layers": frame_layers, "total_frames": offset, "allocated_frames": est_frames,
            "n_utterances": len(rows), "seconds": time.time() - t0}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
