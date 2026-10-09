"""Download the transcribed rows (no audio) of four Vaani districts to one local parquet file.

Needs: Vaani terms accepted on Hugging Face and `hf auth login` done. Takes ~5 min.
Run:   uv run --with duckdb python scripts/vaani_transcripts.py data/vaani_transcripts.parquet
"""
import sys
import time

import duckdb

DISTRICTS = ["Haryana/Rohtak", "Haryana/Jhajjar", "Haryana/CharkhiDadri", "Delhi/NewDelhi"]
out = sys.argv[1]

con = duckdb.connect()
con.sql("SET threads=32")
con.sql("CREATE SECRET hf (TYPE HUGGINGFACE, PROVIDER credential_chain)")
paths = ", ".join(f"'hf://datasets/ARTPARK-IISc/Vaani/audio/{d}/*.parquet'" for d in DISTRICTS)
t = time.time()
con.sql(f"""COPY (
  SELECT district, state, language, speakerID, pincode, duration, transcript, filename
  FROM read_parquet([{paths}], filename=true)
  WHERE isTranscriptionAvailable = 'Yes'
) TO '{out}' (FORMAT parquet)""")
n = con.sql(f"SELECT count(*) FROM '{out}'").fetchone()[0]
print(f"{n} rows in {time.time() - t:.0f} s")
