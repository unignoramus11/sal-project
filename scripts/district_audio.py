"""Fetch audio of transcribed Vaani district utterances that contain a candidate word (closure check, RQ2).

Selects transcribed rows of Rohtak, Jhajjar, Charkhi Dadri (Haryana) and New Delhi whose transcript
contains any word from results/gemination/candidate_words.csv, and stores audio bytes, the original file
name (which carries the speaker code), and metadata. Vaani parquet files store audio in row groups that
mix transcribed and untranscribed rows, so the transfer is close to the full 92 GB; only matches are kept.

Output: data/vaani_districts/candidates_audio.parquet
Run:    uv run python scripts/district_audio.py
"""
import time

import duckdb
import pandas as pd

from sal.paths import DATA, RESULTS

DISTRICTS = ["Haryana/Rohtak", "Haryana/Jhajjar", "Haryana/CharkhiDadri", "Delhi/NewDelhi"]
words = pd.read_csv(RESULTS / "gemination" / "candidate_words.csv")["word"].tolist()
pattern = "(" + "|".join(sorted(words, key=len, reverse=True)) + ")"
out = DATA / "vaani_districts" / "candidates_audio.parquet"
out.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.sql("SET threads=16")
con.sql("CREATE SECRET hf (TYPE HUGGINGFACE, PROVIDER credential_chain)")
paths = ", ".join(f"'hf://datasets/ARTPARK-IISc/Vaani@9e6ed4f94f7060e633cbc500480623ae7521522c/audio/{d}/*.parquet'"
                  for d in DISTRICTS)
t0 = time.time()
con.execute(f"""COPY (
  SELECT state, district, language, speakerID, gender, pincode, duration, transcript, UtteranceSequenceID,
         audio.path AS audio_path, audio.bytes AS audio, filename
  FROM read_parquet([{paths}], filename=true)
  WHERE isTranscriptionAvailable = 'Yes' AND regexp_matches(transcript, ?)
) TO '{out}' (FORMAT parquet)""", [pattern])
n = con.sql(f"SELECT count(*) FROM '{out}'").fetchone()[0]
print(f"{n} utterances written to {out} in {time.time() - t0:.0f} s")
