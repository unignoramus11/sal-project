"""Read Vaani-Hi audio from the dataset parquet as float32 waveforms in [-1, 1].

Scaling matches torchaudio.load (int16 / 32768), which PRiSM uses, so model inputs are identical.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def decode(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if x.ndim != 1:
        raise ValueError("expected mono audio")
    return x, sr


def iter_audio(path: Path, batch_rows: int = 256, start: int = 0,
               ids: set[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (metadata_idx, waveform) in file order, optionally only for `ids`, skipping `start` rows."""
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["metadata_idx", "audio"]):
        b = batch.to_pydict()
        for mi, a in zip(b["metadata_idx"], b["audio"]):
            seen += 1
            if seen <= start or (ids is not None and mi not in ids):
                continue
            x, sr = decode(a)
            if sr != 16000:
                raise ValueError(f"utterance {mi}: sample rate {sr}")
            yield mi, x
