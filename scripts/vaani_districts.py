"""Metadata summary of Vaani district folders (ARTPARK-IISc/Vaani), without reading audio.

Reports hours, transcribed hours, language labels, speaker-ID coverage and pincodes per district.

Needs: Vaani terms accepted on Hugging Face and `hf auth login` done.
Run:   uv run --with duckdb python scripts/vaani_districts.py Haryana/Rohtak Delhi/NewDelhi
"""
import sys, time, duckdb
con = duckdb.connect()
con.sql("SET threads=32")
con.sql("CREATE SECRET hf (TYPE HUGGINGFACE, PROVIDER credential_chain)")
for d in sys.argv[1:]:
    t = time.time()
    src = f"'hf://datasets/ARTPARK-IISc/Vaani/audio/{d}/*.parquet'"
    con.sql(f"CREATE OR REPLACE TABLE m AS SELECT language, duration, speakerID, isTranscriptionAvailable AS tr, pincode, filename FROM read_parquet({src}, filename=true)")
    print(f"\n== {d} ({time.time()-t:.0f} s)")
    print(con.sql("""SELECT count(*) utts, round(sum(duration)/3600,1) h,
        round(sum(duration) FILTER (tr='Yes')/3600,1) h_tr, count(*) FILTER (tr='Yes') n_tr,
        round(sum(duration) FILTER (tr='Yes' AND language='Haryanvi')/3600,1) h_tr_haryanvi,
        count(DISTINCT speakerID) FILTER (coalesce(speakerID,'None') NOT IN ('None','')) spk,
        count(DISTINCT speakerID) FILTER (tr='Yes' AND coalesce(speakerID,'None') NOT IN ('None','')) spk_tr,
        round(avg((coalesce(speakerID,'None') IN ('None',''))::int),2) frac_no_spk,
        count(DISTINCT pincode) pins FROM m"""))
    print(con.sql("SELECT language, count(*) n, count(*) FILTER (tr='Yes') n_tr, round(sum(duration)/3600,1) h FROM m GROUP BY 1 ORDER BY n DESC LIMIT 6"))
    sys.stdout.flush()
