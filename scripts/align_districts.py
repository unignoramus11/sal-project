"""Forced alignment of candidate words in the Vaani district audio (closure check, RQ2).

For each utterance in data/vaani_districts/candidates_audio.parquet:
  1. spoken form of the transcript ({...} corrections and <...> tags removed), split into words;
  2. each word romanised with uroman and reduced to the MMS aligner's alphabet (a-z and ');
  3. torchaudio's MMS_FA (Pratap et al., 2023) aligns the romanised words to the audio;
  4. for every token of a candidate word, the span of the target stop's letters inside the word is located by
     romanising the word up to the stop (र|ख|ा -> "ra" | "rakh" -> stop letters 2..4, "kh").
The aligner works from the standard spelling and is independent of both phone recognizers, so the
location it gives cannot reflect their doubling decisions.

With --targets geminate the same is done for lexical voiceless geminates in any word (e.g., the च्छ of अच्छा),
the positive class of the layer-wise geminate probe (scripts/geminate_layers.py).

Output: data/vaani_districts/alignments.parquet, one row per candidate token: utterance key, speaker,
district, word, word start/end (s), stop letter span start/end (s), preceding and following vowel letter
spans (s), alignment score.
Run:    uv run --with uroman python scripts/align_districts.py --device cpu
"""
import argparse
import io
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
import uroman
from torchaudio.pipelines import MMS_FA as BUNDLE

from sal.data.rebuild import speaker_from_path
from sal.paths import DATA, RESULTS

STOPS = set("कखचछटठतथप")
MATRA = set("ािीुूृेैोौ")
INDEP = set("अआइईउऊऋएऐओऔ")
CONS = {chr(c) for c in range(0x915, 0x93A)}
VOWEL_LETTERS = set("aeiou")


def spoken_words(tx):
    tx = re.sub(r"\{[^}]*\}", " ", tx or "")
    tx = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", tx)
    return re.findall(r"[ऀ-ॿ]+", tx)


def target_positions(w):
    """Indices of single voiceless stops between vowels (same rule as scripts/stop_words.py)."""
    out = []
    for i in range(1, len(w) - 1):
        p, c, n = w[i - 1], w[i], w[i + 1]
        if c in STOPS and p != "्" and n not in ("्", "़") and n in MATRA and (p in MATRA or p in INDEP or p in CONS):
            out.append(i)
    return out


PAIRS = {(c, c) for c in STOPS} | {("क", "ख"), ("च", "छ"), ("ट", "ठ"), ("त", "थ")}


def geminate_positions(w):
    """Indices of the first letter of voiceless geminate clusters between vowels (क्क, च्छ, त्थ, ...)."""
    out = []
    for i in range(1, len(w) - 3):  # needs a letter after the cluster
        p, c, h, d, n = w[i - 1], w[i], w[i + 1], w[i + 2], w[i + 3]
        vowel_after = n in MATRA or (n in CONS and (i + 4 >= len(w) or w[i + 4] != "्"))   # पत्थर: थ keeps its vowel
        if h == "्" and (c, d) in PAIRS and vowel_after and (p in MATRA or p in INDEP or p in CONS) and p != "्":
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: align only utterances with index %% n == i")
    ap.add_argument("--targets", choices=["single", "geminate"], default="single",
                    help="single: candidate words' single stops; geminate: any word's voiceless geminate cluster")
    ap.add_argument("--data", default=None, help="input parquet (default data/vaani_districts/candidates_audio.parquet)")
    ap.add_argument("--out", default=None, help="output parquet (default data/vaani_districts/alignments.parquet)")
    args = ap.parse_args()

    ur = uroman.Uroman()
    model = BUNDLE.get_model().to(args.device).eval()
    tokenizer, aligner = BUNDLE.get_tokenizer(), BUNDLE.get_aligner()
    alphabet = set(BUNDLE.get_dict()) - {"-", "*"}
    words_set = set(pd.read_csv(RESULTS / "gemination" / "candidate_words.csv")["word"])

    def rom(w):
        return "".join(ch for ch in ur.romanize_string(w).lower() if ch in alphabet)

    # stream in small batches: the file holds ~3 GB of audio and loading it whole swaps a 12 GB machine to death
    pf = pq.ParquetFile(args.data or DATA / "vaani_districts" / "candidates_audio.parquet")
    cols = ["transcript", "audio", "audio_path", "district", "state", "language"]
    n_total = pf.metadata.num_rows if not args.limit else min(args.limit, pf.metadata.num_rows)

    def records():
        seen = 0
        for batch in pf.iter_batches(batch_size=128, columns=cols):
            for r in batch.to_pandas().itertuples(index=False):
                if seen == n_total:
                    return
                seen += 1
                yield r

    si, sn = map(int, args.shard.split("/"))
    rows, fails, done, t0 = [], 0, 0, time.time()
    for k, r in enumerate(records()):
        if k % sn != si:
            continue
        done += 1
        if done % 250 == 0:
            print(f"{done}/{n_total // sn} utterances, {len(rows)} tokens, {fails} failures, "
                  f"{done / (time.time() - t0):.1f} utt/s", flush=True)
        words = spoken_words(r.transcript)
        romans = [rom(w) for w in words]
        keep = [i for i, x in enumerate(romans) if x]
        def wanted(w):
            return w in words_set if args.targets == "single" else bool(geminate_positions(w))

        if not any(wanted(words[i]) for i in keep):
            continue
        wav, sr = sf.read(io.BytesIO(r.audio), dtype="float32")
        x = torch.from_numpy(wav).unsqueeze(0).to(args.device)
        try:
            with torch.inference_mode():
                emission, _ = model(x)
                spans = aligner(emission[0], tokenizer([romans[i] for i in keep]))
        except Exception:
            fails += 1
            continue
        sec_per_frame = x.shape[1] / emission.shape[1] / sr
        for j, i in enumerate(keep):
            w = words[i]
            if not wanted(w):
                continue
            chars = spans[j]                       # one TokenSpan per letter of romans[i]
            positions = target_positions(w) if args.targets == "single" else geminate_positions(w)
            width = 1 if args.targets == "single" else 3      # stop letter, or stop + virama + stop
            for pos in positions:
                a, b = len(rom(w[:pos])), len(rom(w[:pos + width]))
                if not (0 < a < b <= len(chars)):
                    continue
                before = [c for c in range(a) if romans[i][c] in VOWEL_LETTERS]
                after = [c for c in range(b, len(chars)) if romans[i][c] in VOWEL_LETTERS]
                rows.append({
                    "audio_path": r.audio_path, "speaker": speaker_from_path(r.audio_path), "district": r.district,
                    "state": r.state, "language": r.language, "word": w, "word_index": i, "stop": w[pos],
                    "geminate": args.targets == "geminate",
                    "word_start": chars[0].start * sec_per_frame, "word_end": chars[-1].end * sec_per_frame,
                    "stop_start": chars[a].start * sec_per_frame, "stop_end": chars[b - 1].end * sec_per_frame,
                    "prev_vowel_end": chars[before[-1]].end * sec_per_frame if before else np.nan,
                    "next_vowel_start": chars[after[0]].start * sec_per_frame if after else np.nan,
                    "next_vowel_end": chars[after[0]].end * sec_per_frame if after else np.nan,
                    "score": float(np.mean([c.score for c in chars])),
                })
    out = pd.DataFrame(rows)
    out.to_parquet(args.out or DATA / "vaani_districts" / "alignments.parquet", index=False)
    print(f"{len(out)} tokens from {out['audio_path'].nunique()} utterances; {fails} alignment failures")


if __name__ == "__main__":
    main()
