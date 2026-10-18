"""Blind manual audit of the automatic closure measurements (RQ2).

Picks N accepted main-analysis tokens (stops between vowels in speech; sal.gemination) balanced by region (Haryana / New Delhi) and stop letter, at most one token per speaker,
and exports for each a short WAV clip and a Praat TextGrid with the word and two empty interval tiers
("closure", "vot") for the annotator to fill. Files are numbered in random order and carry no region,
district, speaker or automatic measurement, so the annotation is blind. The key (token, region, automatic
landmarks relative to the clip) goes to a separate directory, read only by scripts/audit_compare.py.

Output: internal/audit/{mohit,nidhi}/audit_XX.{wav,TextGrid} (one copy per annotator), internal/audit/README.txt,
        internal/audit_key/key.csv
Run:    uv run python scripts/audit_tokens.py --n 50
"""
import argparse
import io
import shutil

import numpy as np
import pandas as pd
import soundfile as sf

from sal.gemination import token_contexts
from sal.paths import DATA, ROOT

MARGIN = 0.15
README = """Manual closure audit ({n} clips)

Work in your own folder (mohit/ or nidhi/). Open each audit_XX.wav with its TextGrid in Praat (select both,
View & Edit). Tier "word" marks the word that contains the stop (spelling shown). In tier "closure" draw one
interval from the end of voicing of the vowel before the stop to the release burst. In tier "vot" draw one
interval from the burst to the onset of voicing of the next vowel. Leave a tier empty if you cannot decide,
and write a note in your notes.csv (clip, note).
Annotate independently: do not compare with each other until both are done. Do not open internal/audit_key/.
Save each TextGrid in place (File > Save as text file, same name).
"""


def textgrid(dur: float, word: str, w0: float, w1: float) -> str:
    def tier(name, intervals):
        s = [f'    item [{name[0]}]:', '        class = "IntervalTier"', f'        name = "{name[1]}"',
             '        xmin = 0', f'        xmax = {dur:.4f}', f'        intervals: size = {len(intervals)}']
        for k, (a, b, t) in enumerate(intervals, 1):
            s += [f'        intervals [{k}]:', f'            xmin = {a:.4f}', f'            xmax = {b:.4f}',
                  f'            text = "{t}"']
        return s
    word_iv = [(0, w0, ""), (w0, w1, word), (w1, dur, "")]
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '', 'xmin = 0', f'xmax = {dur:.4f}',
             'tiers? <exists>', 'size = 3', 'item []:']
    lines += tier((1, "word"), word_iv) + tier((2, "closure"), [(0, dur, "")]) + tier((3, "vot"), [(0, dur, "")])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_parquet(DATA / "vaani_districts" / "closures.parquet")
    df["context"] = token_contexts(df["word"], df["stop"])
    df = df[(df["reject"] == "") & df["context"].isin(["vowel", "initial_schwa"])].copy()   # main-analysis tokens
    df["region"] = np.where(df["state"] == "Haryana", "Haryana", "Delhi")

    # equal tokens per region; within a region spread evenly over stops (round robin), one token per speaker
    picks = []
    for _, g in df.groupby("region"):
        g = g.sample(frac=1, random_state=args.seed)
        pools = {s: list(gs.index) for s, gs in g.groupby("stop")}
        used, take = set(), []
        while len(take) < args.n // 2 and any(pools.values()):
            for s in sorted(pools, key=lambda s: -len(pools[s])):
                while pools[s]:
                    i = pools[s].pop()
                    if g.at[i, "speaker"] not in used:
                        used.add(g.at[i, "speaker"])
                        take.append(i)
                        break
                if len(take) == args.n // 2:
                    break
        picks += take
    sel = df.loc[picks].sample(frac=1, random_state=args.seed).reset_index(drop=True)

    audio = pd.read_parquet(DATA / "vaani_districts" / "candidates_audio.parquet", columns=["audio_path", "audio"])
    audio = audio.drop_duplicates("audio_path").set_index("audio_path")["audio"]
    clips = ROOT / "internal" / "audit" / "mohit"
    keydir = ROOT / "internal" / "audit_key"
    clips.mkdir(parents=True, exist_ok=True)
    keydir.mkdir(parents=True, exist_ok=True)
    key = []
    for k, r in sel.iterrows():
        x, sr = sf.read(io.BytesIO(audio[r.audio_path]), dtype="float32")
        c0 = max(0.0, min(r.win_start, r.word_start) - MARGIN)
        c1 = min(len(x) / sr, max(r.win_end, r.word_end) + MARGIN)
        name = f"audit_{k + 1:02d}"
        sf.write(clips / f"{name}.wav", x[int(c0 * sr):int(c1 * sr)], sr)
        (clips / f"{name}.TextGrid").write_text(
            textgrid(c1 - c0, r.word, max(0.0, r.word_start - c0), min(c1 - c0, r.word_end - c0)), encoding="utf-8")
        key.append({"clip": name, "token_id": r.token_id, "region": r.region, "district": r.district,
                    "speaker": r.speaker, "word": r.word, "stop": r.stop, "clip_start": c0,
                    "auto_closure_start": r.closure_start - c0, "auto_burst": r.burst - c0,
                    "auto_voice_onset": r.voice_onset - c0, "auto_voi_ms": r.voi_ms, "auto_closure_ms": r.closure_ms,
                    "auto_vot_ms": r.vot_ms})
    pd.DataFrame(key).to_csv(keydir / "key.csv", index=False)
    (ROOT / "internal" / "audit" / "README.txt").write_text(README.format(n=len(sel)))
    shutil.copytree(clips, clips.parent / "nidhi", dirs_exist_ok=True)
    for who in ("mohit", "nidhi"):
        (clips.parent / who / "notes.csv").write_text("clip,note\n")
    k = pd.DataFrame(key)
    print(f"exported {len(k)} clips to {clips}")
    print(k.groupby(["region", "stop"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
