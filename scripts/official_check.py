"""The official Vaani-Hi (changelinglab/vaani-hi-geo, commit 264fb053): composition, speakers, leakage.

The official release has no speaker column. Each official clip is matched to its Vaani source by audio: the
release stores 32-bit float WAV written from 16-bit 16 kHz Vaani audio, so float * 32768 recovers the original
samples exactly, and a hash of the first 3 s of 16-bit samples plus the clip length (capped at 20 s, as the
release clips) identifies the source clip. Sources are the Vaani shards our rebuild fetched
(data/vaani_hi_rebuild/raw: the first 4 shards of each selected district, which is what PRiSM's recipe reads);
the Vaani file name carries the speaker code (sal.data.rebuild.speaker_from_path).

Output: data/vaani_hi_official/matched.parquet (official rows + matched source file, speaker)
        results/official/composition.csv, leakage.csv, speakers_per_split.csv, summary.txt
Run:    uv run python scripts/official_check.py
"""
import hashlib
import io
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from sal.data.rebuild import speaker_from_path
from sal.paths import DATA, RESULTS

OFF = DATA / "vaani_hi_official" / "data"
RAW = DATA / "vaani_hi_rebuild" / "raw"
OUT = RESULTS / "official"
SR, HEAD, CAP = 16000, 3 * 16000, 20 * 16000


def fingerprint(x16: np.ndarray) -> str:
    n = min(len(x16), CAP)
    return hashlib.blake2b(x16[:HEAD].tobytes() + n.to_bytes(4, "little"), digest_size=16).hexdigest()


def source_index(path: Path) -> list[dict]:
    rows = []
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=256, columns=["audio", "state", "district"]):
        for a, st, di in zip(b.column("audio").to_pylist(), b.column("state").to_pylist(),
                             b.column("district").to_pylist(), strict=True):
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
            if sr != SR or x.ndim != 1:
                continue
            rows.append({"fp": fingerprint(x), "source_path": a["path"], "source_shard": path.name,
                         "src_state": st, "src_district": di})
    return rows


def official_rows(path: Path) -> list[dict]:
    rows = []
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c != "audio"]
    for b in pf.iter_batches(batch_size=256):
        d = b.to_pydict()
        for k in range(b.num_rows):
            a = d["audio"][k]
            a = a["bytes"] if isinstance(a, dict) else a
            x, sr = sf.read(io.BytesIO(a), dtype="float32")
            x16 = np.round(np.clip(x, -1, 32767 / 32768) * 32768).astype(np.int16)
            r = {c: d[c][k] for c in cols}
            r.update({"fp": fingerprint(x16), "seconds": len(x) / sr, "sr": sr, "shard": path.name})
            rows.append(r)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        off = pd.DataFrame([r for rs in ex.map(official_rows, sorted(OFF.glob("*.parquet"))) for r in rs])
        src = pd.DataFrame([r for rs in ex.map(source_index, sorted(RAW.glob("*.parquet"))) for r in rs])
    dup = src["fp"].duplicated(keep=False)
    src = src[~dup]                                   # ambiguous fingerprints are not used for matching
    m = off.merge(src, on="fp", how="left")
    m["speaker"] = m["source_path"].map(lambda p: speaker_from_path(p) if isinstance(p, str) else None)
    m.drop(columns=["fp"]).to_parquet(DATA / "vaani_hi_official" / "matched.parquet", index=False)

    lines = [f"official rows {len(off)}; sample rates {dict(Counter(off['sr']))}; mean duration "
             f"{off['seconds'].mean():.2f} s; source clips indexed {len(src)} ({int(dup.sum())} ambiguous dropped); "
             f"matched {m['speaker'].notna().mean():.1%}"]
    comp = m.groupby(["state", "district"]).agg(utterances=("split", "size"), pincodes=("pincode", "nunique"),
                                                matched=("speaker", lambda s: s.notna().mean()),
                                                seconds=("seconds", "mean")).reset_index()
    comp.to_csv(OUT / "composition.csv", index=False)
    lines.append(f"split sizes {m['split'].value_counts().to_dict()}; pincodes {m['pincode'].nunique()}; states "
                 f"{m['state'].nunique()}; districts {m['district'].nunique()}")

    mm = m[m["speaker"].notna()]
    train_spk = set(mm.loc[mm["split"] == "train", "speaker"])
    leak = []
    for sp in ("validation", "val", "test"):
        t = mm[mm["split"] == sp]
        if len(t):
            leak.append({"split": sp, "matched_utterances": len(t), "speaker_in_train": t["speaker"].isin(train_spk).mean()})
    pd.DataFrame(leak).to_csv(OUT / "leakage.csv", index=False)
    for r in leak:
        lines.append(f"{r['split']}: {r['speaker_in_train']:.1%} of {r['matched_utterances']} matched utterances have "
                     f"their speaker in train")
    per = mm.groupby("speaker").size()
    lines.append(f"speakers {mm['speaker'].nunique()} among matched; utterances per speaker median {per.median():.0f}, "
                 f"mean {per.mean():.1f}, max {per.max()}")
    mm.groupby("split")["speaker"].nunique().to_csv(OUT / "speakers_per_split.csv")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
