"""Measure closure, VOT and following-vowel duration for every aligned candidate token (RQ2).

Input:  data/vaani_districts/alignments.parquet (scripts/align_districts.py) and candidates_audio.parquet.
Window: from the end of the preceding vowel letter to the start of the following vowel letter, widened by
        60 ms on each side and clipped to the word; the stop's aligned centre seeds the search
        (sal.closure.measure). The window comes from MMS alignment of the standard spelling, never from the
        phone recognizers.
Output: data/vaani_districts/closures.parquet (one row per token: measurements, landmarks, reject reason).
Run:    uv run python scripts/measure_closure.py --workers 8
"""
import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd
import soundfile as sf

from sal.closure import measure
from sal.paths import DATA

PAD = 0.06


def work(args):
    wav_bytes, toks = args
    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    out = []
    for t in toks:
        lo = t["prev_vowel_end"] if np.isfinite(t["prev_vowel_end"]) else t["stop_start"] - 0.08
        hi = t["next_vowel_start"] if np.isfinite(t["next_vowel_start"]) else t["stop_end"] + 0.08
        lo = max(t["word_start"] - PAD, lo - PAD, 0.0)
        hi = min(t["word_end"] + PAD, hi + PAD, len(x) / sr)
        centre = 0.5 * (t["stop_start"] + t["stop_end"])
        m = measure(x, sr, centre, lo, hi)
        out.append({"token_id": t["token_id"], **asdict(m), "win_start": lo, "win_end": hi})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alignments", default=str(DATA / "vaani_districts" / "alignments.parquet"))
    ap.add_argument("--out", default=str(DATA / "vaani_districts" / "closures.parquet"))
    args = ap.parse_args()
    al = pd.read_parquet(args.alignments).reset_index(names="token_id")
    audio = pd.read_parquet(DATA / "vaani_districts" / "candidates_audio.parquet", columns=["audio_path", "audio"])
    audio = audio.drop_duplicates("audio_path").set_index("audio_path")["audio"]
    jobs = [(audio[p], g.to_dict("records")) for p, g in al.groupby("audio_path")]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, res in enumerate(ex.map(work, jobs, chunksize=32)):
            rows += res
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} utterances", flush=True)
    m = pd.DataFrame(rows)
    out = al.merge(m, on="token_id")
    out.to_parquet(args.out, index=False)
    ok = out["reject"] == ""
    print(f"{len(out)} tokens, {ok.mean():.1%} accepted")
    print(out.loc[~ok, "reject"].value_counts().to_string())


if __name__ == "__main__":
    main()
