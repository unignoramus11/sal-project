"""Greedy CTC phone transcription of Vaani-Hi with PRiSM's own model wrappers.

Runs in the PRiSM inference environment (envs/prism, Python 3.10, torch 2.6 + k2). For each utterance
it stores:
  processed_transcript  the exact string PRiSM's runner returns (input to the transcript probe)
  tokens                the collapsed CTC output units (phones), for phone-level probes and DPR
  frame_ids             the per-frame argmax ids before collapsing, for alignment
The first --check utterances are also passed through PRiSM's runner.__call__ and must match exactly.

Run:  python scripts/transcribe.py --model w2v2 --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
          --out outputs/transcripts/w2v2_rebuild.parquet
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "prism"))

MODELS = {
    "w2v2": "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "zipa": "anyspeech/zipa-large-crctc-ns-800k",
}


def build(model: str, device: str):
    if model == "w2v2":
        from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_inference
        runner = build_wav2vec2phoneme_inference(hf_repo=MODELS[model], device=device)

        def units(ids):
            return runner.tokenizer.ids2tokens(ids)
    else:
        from src.model.zipa.zipformer_crctc.builders import build_zipactc_inference
        runner = build_zipactc_inference(
            work_dir=str(ROOT / "outputs" / "cache" / "zipactc"), hf_repo=MODELS[model],
            bpe_model=str(ROOT / "external" / "prism" / "src/model/zipa/resources/unigram_127.model"),
            device=device)
        runner.tokenizer._build_sentence_piece_processor()

        def units(ids):
            return [runner.tokenizer.sp.IdToPiece(int(i)) for i in ids]
    return runner, units


@torch.no_grad()
def transcribe_one(runner, units, model: str, wav: np.ndarray):
    speech, speech_length = runner.batchify(torch.from_numpy(wav))
    logits, _ = runner.inference_model.ctc_logits(speech, speech_length)
    frame_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    collapsed = runner.ctc_collapse(frame_ids.tolist())
    toks = units(collapsed)
    if model == "w2v2":
        text = "".join(toks).replace(runner.tokenizer.unk_symbol, "").replace(runner.tokenizer.pad_symbol, "")
        toks = [t for t in toks if t not in (runner.tokenizer.unk_symbol, runner.tokenizer.pad_symbol)]
    else:
        text = runner.tokenizer.ids2text(collapsed)
    return text, toks, frame_ids.astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    runner, units = build(args.model, args.device)
    parts_dir = args.out.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(args.data)
    done = {int(p.stem) for p in parts_dir.glob("*.parquet")}
    t0, n_done, buf, chunk_id, row = time.time(), 0, [], 0, 0
    for batch in pf.iter_batches(batch_size=args.chunk, columns=["metadata_idx", "audio"]):
        if args.limit is not None and row >= args.limit:
            break
        if chunk_id in done:
            chunk_id += 1
            row += batch.num_rows
            continue
        b = batch.to_pydict()
        for mi, audio in zip(b["metadata_idx"], b["audio"]):
            if args.limit is not None and row >= args.limit:
                break
            wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
            assert sr == 16000 and wav.ndim == 1
            text, toks, fids = transcribe_one(runner, units, args.model, wav)
            if row < args.check:
                ref = runner(torch.from_numpy(wav))[0]["processed_transcript"]
                if ref != text:
                    raise AssertionError(f"mismatch at {mi}: {ref!r} vs {text!r}")
            buf.append({"metadata_idx": mi, "processed_transcript": text, "tokens": toks,
                        "frame_ids": fids.tolist(), "n_frames": len(fids), "seconds": len(wav) / sr})
            row += 1
            n_done += 1
        pq.write_table(pa.Table.from_pylist(buf), parts_dir / f"{chunk_id:05d}.parquet")
        rate = n_done / (time.time() - t0)
        print(f"chunk {chunk_id} written, {row} rows, {rate:.1f} utt/s", flush=True)
        buf, chunk_id = [], chunk_id + 1

    parts = sorted(parts_dir.glob("*.parquet"))
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    table = table.replace_schema_metadata({"model": MODELS[args.model], "source": str(args.data)})
    pq.write_table(table, args.out)
    print(f"wrote {args.out} ({table.num_rows} rows); first {args.check} matched PRiSM's runner")


if __name__ == "__main__":
    main()
