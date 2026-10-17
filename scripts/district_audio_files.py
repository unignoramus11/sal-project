"""Fetch audio of transcribed Vaani district utterances that contain a candidate word (closure check, RQ2).

Same selection as scripts/district_audio.py, but downloads whole shards with the Hugging Face client
(parallel chunked transfer, ~3x faster than DuckDB's range reads here), filters each shard locally, keeps only
the matching rows and deletes the shard. Resumable: finished shards are kept as parts.

Output: data/vaani_districts/candidates_audio.parquet
Run:    uv run python scripts/district_audio_files.py --workers 4
"""
import argparse
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from sal.paths import DATA, RESULTS

REPO, REV = "ARTPARK-IISc/Vaani", "9e6ed4f94f7060e633cbc500480623ae7521522c"
DISTRICTS = ["Haryana/Rohtak", "Haryana/Jhajjar", "Haryana/CharkhiDadri", "Delhi/NewDelhi"]
COLS = ["state", "district", "language", "speakerID", "gender", "pincode", "duration", "transcript",
        "UtteranceSequenceID", "isTranscriptionAvailable", "audio"]

out = DATA / "vaani_districts" / "candidates_audio.parquet"
parts = DATA / "vaani_districts" / "candidates_parts"
words = pd.read_csv(RESULTS / "gemination" / "candidate_words.csv")["word"].tolist()
pattern = "(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + ")"


def one(path: str, tmp: Path) -> int:
    part = parts / (path.replace("audio/", "").replace("/", "__"))
    if part.exists():
        return pq.ParquetFile(part).metadata.num_rows
    for attempt in range(4):                    # transient transfer errors (xet/CAS) should not kill the run
        try:
            local = hf_hub_download(REPO, path, repo_type="dataset", revision=REV, local_dir=tmp)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"retry {path}: {type(e).__name__}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    t = pq.read_table(local, columns=COLS)
    mask = pc.and_(pc.equal(t["isTranscriptionAvailable"], "Yes"),
                   pc.fill_null(pc.match_substring_regex(t["transcript"], pattern), False))
    t = t.filter(mask)
    audio = t["audio"]
    t = t.drop_columns(["audio", "isTranscriptionAvailable"])
    t = t.append_column("audio_path", pc.struct_field(audio, "path")).append_column("audio", pc.struct_field(audio, "bytes"))
    t = t.append_column("filename", pa.array([path] * t.num_rows))
    if "speakerID" in t.column_names and not pa.types.is_string(t.schema.field("speakerID").type):
        t = t.set_column(t.column_names.index("speakerID"), "speakerID", pa.nulls(t.num_rows, pa.string()))
    tmp_part = part.with_suffix(".tmp")
    pq.write_table(t, tmp_part)
    tmp_part.rename(part)
    Path(local).unlink(missing_ok=True)
    return t.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    parts.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    shards = sorted(f.path for d in DISTRICTS
                    for f in api.list_repo_tree(REPO, path_in_repo=f"audio/{d}", repo_type="dataset", revision=REV)
                    if f.path.endswith(".parquet"))
    tmp = Path(tempfile.mkdtemp(prefix="vaani_", dir=DATA / "vaani_districts"))
    total = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, s, tmp): s for s in shards}
        for k, f in enumerate(as_completed(futs)):
            total += f.result()
            print(f"{k + 1}/{len(shards)} shards, {total} matching utterances", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tables = [pq.read_table(p) for p in sorted(parts.glob("*.parquet"))]
    pq.write_table(pa.concat_tables(tables, promote_options="default"), out)
    print(f"wrote {out}: {sum(t.num_rows for t in tables)} utterances from {len(tables)} shards")


if __name__ == "__main__":
    main()
