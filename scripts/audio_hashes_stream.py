"""Memory-bounded comparison of reconstructed Vaani-Hi audio to Mohit's reference.

Run from the sal-project-nov repository root:
    envs/prism/bin/python -u scripts/audio_hashes_stream.py

This tool does not write or overwrite the reference CSV.
"""
from collections import Counter
from hashlib import blake2b
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/rebuild_checks/audio_hashes.csv"
DATA = ROOT / "data/vaani_hi_rebuild/vaani_hi_rebuild.parquet"
FIELDS = ("audio_hash", "pincode", "speaker", "latitude", "longitude")


def main():
    if not REF.is_file() or not DATA.is_file():
        raise SystemExit(f"Missing input: reference={REF.is_file()}, dataset={DATA.is_file()}")

    ref = pd.read_csv(REF, dtype={
        "metadata_idx": "int64", "pincode": "string", "speaker": "string",
        "latitude": "string", "longitude": "string", "audio_hash": "string",
    })
    if ref["metadata_idx"].duplicated().any():
        raise SystemExit("ERROR: duplicate metadata_idx in reference")
    reference = ref.set_index("metadata_idx").to_dict(orient="index")
    seen = set()
    mismatches = Counter()
    examples = {}
    total = 0

    pf = pq.ParquetFile(DATA)
    print(f"Reference rows: {len(reference)}; Dataset rows: {pf.metadata.num_rows}", flush=True)
    for batch in pf.iter_batches(batch_size=32, columns=[
        "metadata_idx", "audio", "pincode", "speaker", "latitude", "longitude"
    ]):
        data = batch.to_pydict()
        for i, idx in enumerate(data["metadata_idx"]):
            idx = int(idx)
            total += 1
            if idx in seen:
                raise SystemExit(f"ERROR: duplicate dataset metadata_idx: {idx}")
            seen.add(idx)
            expected = reference.get(idx)
            if expected is None:
                mismatches["only_in_dataset"] += 1
                continue
            raw_audio = data["audio"][i]
            if isinstance(raw_audio, dict):
                raw_audio = raw_audio["bytes"]
            actual = {
                "audio_hash": blake2b(raw_audio, digest_size=12).hexdigest(),
                "pincode": str(data["pincode"][i]),
                "speaker": str(data["speaker"][i]),
                "latitude": f'{float(data["latitude"][i]):.5f}',
                "longitude": f'{float(data["longitude"][i]):.5f}',
            }
            for field in FIELDS:
                if str(expected[field]) != actual[field]:
                    mismatches[field] += 1
                    if len(examples.setdefault(field, [])) < 5:
                        examples[field].append((idx, str(expected[field]), actual[field]))
        if total % 2048 < len(data["metadata_idx"]):
            print(f"Processed {total}/{pf.metadata.num_rows}", flush=True)

    mismatches["only_in_reference"] = len(reference.keys() - seen)
    print(f"Processed {total} rows total")
    for field in (*FIELDS, "only_in_dataset", "only_in_reference"):
        print(f"{field:18s}: {mismatches[field]} differences")
        for example in examples.get(field, []):
            print(f"  metadata_idx={example[0]} reference={example[1]} actual={example[2]}")
    matched = not any(mismatches.values()) and total == len(reference)
    print("SAME DATA" if matched else "DIFFERENT DATA", flush=True)
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
