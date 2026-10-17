"""Rebuild Vaani-Hi from the public Vaani corpus, following the PRiSM recipe.

PRiSM builds Vaani-Hi with `src/recipe/geolocation/local/vaanihindi_sample.py` and
`vaanihindi_upload.py`. This module reproduces both scripts step by step:

1. For each of 12 Hindi-belt states (PRiSM order), sample up to 4 districts with `random.sample`
   under seed 42, drawing states in order from one generator.
2. For each district, read the first 600 rows of the first 4 parquet shards.
3. Group by pincode and keep pincodes with more than 450 utterances. As in PRiSM, the kept rows are
   concatenated in ascending pincode order, which fixes `metadata_idx`.
4. Map each pincode to the mean latitude and longitude of its post offices.
5. Split 75/10/15 within each pincode with PRiSM's `create_splits` (numpy seed 42, pandas
   `sample(frac=1, random_state=42)` per pincode).
6. Keep 16 kHz mono audio, clipped to 20 s.

Deviations from PRiSM, all recorded in the manifest:

- District order. PRiSM samples from `os.listdir`, whose order depends on the file system. We sample
  from the sorted list. States with at most 4 districts are unaffected (all districts are taken).
- Coordinates. PRiSM averages every row of the pincode table, including placeholder coordinates.
  We drop coordinates outside India's bounding box before averaging.
- Audio encoding. PRiSM re-encodes to 32-bit float WAV. Vaani audio is 16-bit PCM, so we keep the
  original bytes (bit-identical samples) unless a clip must be shortened.
- Corpus version. We pin a Vaani commit; PRiSM used a local copy of unknown date.
- Extra columns. We keep a speaker code for every row (from the Vaani file name), the explicit speaker
  ID where present, duration, transcript and source location.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm

REPO = "ARTPARK-IISc/Vaani"

# PRiSM vaanihindi_sample.py, in its order.
STATE_LIST = [
    "Chandigarh", "HimachalPradesh", "Delhi", "MadhyaPradesh", "Jharkhand", "Uttarakhand",
    "Bihar", "Chhattisgarh", "Haryana", "Rajasthan", "Punjab", "UttarPradesh",
]
SEED = 42
DISTRICTS_PER_STATE = 4
FILES_PER_DISTRICT = 4
SAMPLES_PER_FILE = 600
UTT_PER_PINCODE = 450
TARGET_SR = 16000
MAX_SECONDS = 20
INDIA_BBOX = {"lat": (6.0, 38.0), "lon": (68.0, 98.0)}

SOURCE_COLUMNS = [
    "audio", "language", "pincode", "gender", "state", "district", "speakerID", "duration",
    "isTranscriptionAvailable", "transcript", "UtteranceSequenceID", "languagesKnown",
]


@dataclass
class Plan:
    revision: str
    districts: dict[str, list[str]]
    files: list[tuple[str, str, str]] = field(default_factory=list)  # (state, district, repo path)


def resolve_revision(revision: str | None = None) -> str:
    """Return a commit hash for the Vaani dataset repo (the current head if none given)."""
    return HfApi().dataset_info(REPO, revision=revision).sha


def make_plan(revision: str) -> Plan:
    """Select districts and shards exactly as PRiSM does, with a sorted district list."""
    api = HfApi()
    rng = random.Random(SEED)
    districts: dict[str, list[str]] = {}
    files: list[tuple[str, str, str]] = []
    for state in STATE_LIST:
        all_d = sorted(
            t.path.split("/")[-1]
            for t in api.list_repo_tree(REPO, path_in_repo=f"audio/{state}", repo_type="dataset",
                                        revision=revision)
        )
        chosen = rng.sample(all_d, k=min(DISTRICTS_PER_STATE, len(all_d)))
        districts[state] = chosen
        for d in chosen:
            shard_names = sorted(
                t.path for t in api.list_repo_tree(REPO, path_in_repo=f"audio/{state}/{d}",
                                                   repo_type="dataset", revision=revision)
                if t.path.endswith(".parquet")
            )
            for n in range(min(FILES_PER_DISTRICT, len(shard_names))):
                prefix = f"audio/{state}/{d}/train-{n:05d}-of-"
                match = [p for p in shard_names if p.startswith(prefix)]
                if len(match) != 1:
                    raise RuntimeError(f"expected one shard for {prefix}, found {match}")
                files.append((state, d, match[0]))
    return Plan(revision=revision, districts=districts, files=files)


def _raw_path(raw_dir: Path, repo_path: str) -> Path:
    return raw_dir / (repo_path.replace("audio/", "").replace("/", "__"))


def fetch_head(repo_path: str, revision: str, out: Path, n_rows: int = SAMPLES_PER_FILE) -> int:
    """Read the first n_rows of a shard (only the row groups needed) and cache them locally."""
    if out.exists():
        return pq.ParquetFile(out).metadata.num_rows
    fs = HfFileSystem()
    with fs.open(f"datasets/{REPO}@{revision}/{repo_path}", "rb", block_size=64 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        groups, total = [], 0
        for g in range(pf.metadata.num_row_groups):
            groups.append(g)
            total += pf.metadata.row_group(g).num_rows
            if total >= n_rows:
                break
        cols = [c for c in SOURCE_COLUMNS if c in pf.schema_arrow.names]
        table = pf.read_row_groups(groups, columns=cols).slice(0, n_rows)
    table = table.append_column("source_file", pa.array([repo_path] * table.num_rows))
    table = table.append_column("source_row", pa.array(np.arange(table.num_rows, dtype=np.int32)))
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.rename(out)
    return table.num_rows


def download(plan: Plan, raw_dir: Path, workers: int = 8) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = {}
    with ThreadPoolExecutor(workers) as ex:
        for _, _, path in plan.files:
            jobs[ex.submit(fetch_head, path, plan.revision, _raw_path(raw_dir, path))] = path
        for fut in tqdm(as_completed(jobs), total=len(jobs), desc="shards"):
            fut.result()


def parse_coord(x) -> float:
    """PRiSM's parser: first number in the string, negated for S or W."""
    s = str(x).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    val = float(m.group(0))
    return -val if ("S" in s or "W" in s) else val


def pincode_coordinates(pincode_csv: Path) -> tuple[dict[str, tuple[float, float]], dict]:
    pm = pd.read_csv(pincode_csv, usecols=["Pincode", "Latitude", "Longitude"], low_memory=False)
    pm["Pincode"] = pm["Pincode"].astype(str).str.extract(r"(\d{6})")[0]
    pm["Latitude"] = pm["Latitude"].apply(parse_coord)
    pm["Longitude"] = pm["Longitude"].apply(parse_coord)
    n_rows = len(pm)
    pm = pm.dropna()
    lat_ok = pm["Latitude"].between(*INDIA_BBOX["lat"])
    lon_ok = pm["Longitude"].between(*INDIA_BBOX["lon"])
    kept = pm[lat_ok & lon_ok]
    means = kept.groupby("Pincode")[["Latitude", "Longitude"]].mean()
    stats = {"rows": n_rows, "rows_parsed": len(pm), "rows_in_bbox": len(kept),
             "pincodes": int(means.shape[0])}
    return {p: (r.Latitude, r.Longitude) for p, r in means.iterrows()}, stats


def speaker_from_path(path: str) -> str:
    """Speaker code from a Vaani file name.

    `IISc_VaaniProject_<M>_<State>_<District>_<code>_<n>_<prompt>_<id>_<start>_<end>.wav`. The code
    equals the explicit `speakerID` minus its `SID_` prefix wherever that column is filled.
    """
    parts = path.split("_")
    if len(parts) != 11 or parts[0] != "IISc" or parts[1] != "VaaniProject":
        raise ValueError(f"unexpected Vaani file name: {path}")
    return parts[5]


def create_splits(df: pd.DataFrame) -> pd.DataFrame:
    """PRiSM vaanihindi_upload.create_splits, unchanged in behaviour."""
    tr_frac, val_frac = 0.75, 0.1
    np.random.seed(SEED)
    df = df.copy()
    df["split"] = np.nan
    df["split"] = df["split"].astype(object)
    for pincode in df["pincode"].unique():
        pdf = df[df["pincode"] == pincode]
        n = len(pdf)
        pdf = pdf.sample(frac=1, random_state=SEED)
        n_train, n_val = int(n * tr_frac), int(n * val_frac)
        df.loc[pdf.index[:n_train], "split"] = "train"
        df.loc[pdf.index[n_train:n_train + n_val], "split"] = "val"
        df.loc[pdf.index[n_train + n_val:], "split"] = "test"
    df.loc[df["split"].isna(), "split"] = "train"
    return df


def prepare_audio(wav_bytes: bytes) -> tuple[bytes, float, bool]:
    """Return (16 kHz mono WAV bytes, duration in s, whether it was clipped or re-encoded)."""
    info = sf.info(io.BytesIO(wav_bytes))
    if info.samplerate == TARGET_SR and info.channels == 1 and info.frames <= TARGET_SR * MAX_SECONDS:
        return wav_bytes, info.frames / TARGET_SR, False
    if info.samplerate != TARGET_SR:
        raise ValueError(f"unexpected sample rate {info.samplerate}")
    data, _ = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=True)
    data = data[: TARGET_SR * MAX_SECONDS, 0]
    buf = io.BytesIO()
    sf.write(buf, data, TARGET_SR, subtype="PCM_16", format="WAV")
    return buf.getvalue(), len(data) / TARGET_SR, True


STRING_COLUMNS = ("language", "gender", "state", "district", "speakerID", "isTranscriptionAvailable",
                  "transcript", "languagesKnown", "source_file")


def _normalise(t: pa.Table) -> pa.Table:
    """Give every shard the same schema. A column that is entirely empty in a shard is stored by the
    Vaani parquet writer as float64 (all null); such columns become null strings, not "nan"."""
    for name in STRING_COLUMNS:
        if name not in t.column_names:
            continue
        col = t.column(name)
        if not pa.types.is_string(col.type):
            if col.null_count != len(col):
                raise TypeError(f"{name}: non-string values in a {col.type} column")
            t = t.set_column(t.column_names.index(name), name, pa.nulls(len(col), pa.string()))
    return t


def assemble(plan: Plan, raw_dir: Path, pincode_csv: Path, out_dir: Path) -> dict:
    """Combine cached shards into Vaani-Hi (rebuilt) and write data plus manifest."""
    tables = [_normalise(pq.read_table(_raw_path(raw_dir, p))) for _, _, p in plan.files]
    df = pa.concat_tables(tables).to_pandas()
    n_sampled = len(df)

    # PRiSM: iterate df.groupby("pincode") (ascending) and keep groups with > 450 rows.
    kept = [g for _, g in df.groupby("pincode") if g.shape[0] > UTT_PER_PINCODE]
    df = pd.concat(kept, ignore_index=True)

    coords, coord_stats = pincode_coordinates(pincode_csv)
    pins = df["pincode"].astype(str)
    missing = sorted(set(pins) - set(coords))
    df["latitude"] = [coords.get(p, (np.nan, np.nan))[0] for p in pins]
    df["longitude"] = [coords.get(p, (np.nan, np.nan))[1] for p in pins]
    if missing:
        raise RuntimeError(f"pincodes without coordinates: {missing}")

    df["metadata_idx"] = np.arange(len(df), dtype=np.int32)
    df = create_splits(df)
    df["lang_sym"] = "<unk>"

    df["source_audio_path"] = [rec["path"] for rec in df["audio"]]
    df["speaker"] = df["source_audio_path"].map(speaker_from_path)
    audio, dur, changed = [], [], 0
    for rec in tqdm(df["audio"], desc="audio"):
        b, d, c = prepare_audio(rec["bytes"])
        audio.append(b)
        dur.append(d)
        changed += c
    df["audio"] = audio
    df["audio_seconds"] = np.array(dur, dtype=np.float32)

    out = pd.DataFrame({
        "audio": df["audio"],
        "pincode": df["pincode"].astype(np.int32),
        "latitude": df["latitude"].astype(np.float32),
        "longitude": df["longitude"].astype(np.float32),
        "split": df["split"].astype(str),
        "metadata_idx": df["metadata_idx"],
        "lang_sym": df["lang_sym"],
        "language": df["language"],
        "gender": df["gender"],
        "state": df["state"],
        "district": df["district"],
        # extras for traceability
        "speaker": df["speaker"],
        "speaker_id": df["speakerID"].replace({"None": None}),
        "audio_seconds": df["audio_seconds"],
        "transcript_available": df["isTranscriptionAvailable"].eq("Yes"),
        "transcript": df["transcript"].where(df["isTranscriptionAvailable"].eq("Yes")),
        "source_file": df["source_file"],
        "source_audio_path": df["source_audio_path"],
        "source_row": df["source_row"],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "vaani_hi_rebuild.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), data_path, row_group_size=500)

    sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    pin_sha = hashlib.sha256(Path(pincode_csv).read_bytes()).hexdigest()
    manifest = {
        "plan": asdict(plan),
        "counts": {
            "sampled_rows": n_sampled,
            "rows": len(out),
            "pincodes": int(out["pincode"].nunique()),
            "splits": out["split"].value_counts().to_dict(),
            "states": out["state"].value_counts().to_dict(),
            "districts": out.groupby(["state", "district"]).size().rename("n").reset_index()
                            .to_dict("records"),
            "hours": float(out["audio_seconds"].sum() / 3600),
            "audio_reencoded": changed,
            "speaker_id_coverage": float(out["speaker_id"].notna().mean()),
            "speakers": int(out["speaker"].nunique()),
            "speaker_code_matches_id": float(
                (out.loc[out["speaker_id"].notna(), "speaker_id"]
                 == "SID_" + out.loc[out["speaker_id"].notna(), "speaker"]).mean()),
        },
        "pincode_table": {"path": str(pincode_csv), "sha256": pin_sha, **coord_stats},
        "output": {"path": str(data_path), "sha256": sha},
        "params": {"seed": SEED, "districts_per_state": DISTRICTS_PER_STATE,
                   "files_per_district": FILES_PER_DISTRICT, "samples_per_file": SAMPLES_PER_FILE,
                   "utt_per_pincode": UTT_PER_PINCODE, "max_seconds": MAX_SECONDS,
                   "india_bbox": INDIA_BBOX},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
