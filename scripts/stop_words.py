"""Count candidate words for the closure-duration check in Vaani transcripts.

A candidate is a spoken word form with a single voiceless stop (or affricate) between vowels in
Devanagari (e.g., रखा); फ is left out because it is often pronounced [f]. Uses the spoken form: `{...}` corrections and `<...>` tags are dropped.
Run: uv run python scripts/stop_words.py data/vaani_districts/transcripts.parquet results/gemination/candidate_words.csv
"""
import collections
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

# फ is excluded: it is often realised as the fricative [f] (e.g. काफी, फोटो), which has no closure.
STOPS = set("कखचछटठतथप")
MATRA = set("ािीुूृेैोौ")
INDEP = set("अआइईउऊऋएऐओऔ")
CONS = {chr(c) for c in range(0x915, 0x93A)}
HAL, NUKTA = "्", "़"
GEM = re.compile(r"([कखगघचछजझटठडढतथदधपफबभ])्\1|क्ख|च्छ|ट्ठ|त्थ|प्फ|ग्घ|ज्झ|ड्ढ|द्ध|ब्भ")


def spoken_words(tx):
    tx = re.sub(r"\{[^}]*\}", " ", tx or "")
    tx = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", tx)
    return re.findall(r"[ऀ-ॿ]+", tx)


def n_single_stops(w):
    n = 0
    for i in range(1, len(w) - 1):
        prev, ch, nxt = w[i - 1], w[i], w[i + 1]
        if ch in STOPS and prev != HAL and nxt not in (HAL, NUKTA) and nxt in MATRA \
                and (prev in MATRA or prev in INDEP or prev in CONS):
            n += 1
    return n


def region(d):
    return "Delhi" if d == "NewDelhi" else "Haryana"


t = pq.read_table(sys.argv[1]).to_pydict()
cand = collections.defaultdict(collections.Counter)
words = collections.defaultdict(collections.Counter)
spk = collections.defaultdict(set)
n_utt, n_tok, n_gem = collections.Counter(), collections.Counter(), collections.Counter()
for d, s, tx in zip(t["district"], t["speakerID"], t["transcript"]):
    n_utt[d] += 1
    for w in spoken_words(tx):
        r = region(d)
        words[r][w] += 1
        k = n_single_stops(w)
        if k:
            n_tok[d] += k
            cand[r][w] += 1
            if s and s != "None":
                spk[r].add(s)
        if GEM.search(w):
            n_gem[d] += 1

print("district       utts  candidate_tokens  geminate_spelled_tokens")
for d in n_utt:
    print(f"{d:13s} {n_utt[d]:5d} {n_tok[d]:17d} {n_gem[d]:24d}")
H, D = cand["Haryana"], cand["Delhi"]
for thr in (5, 10, 20):
    both = [w for w in H if H[w] >= thr and D[w] >= thr]
    print(f">= {thr} tokens in both regions: {len(both)} words, "
          f"{sum(H[w] for w in both)} Haryana / {sum(D[w] for w in both)} Delhi tokens")
print("speakers with ID:", len(spk["Haryana"]), "Haryana /", len(spk["Delhi"]), "Delhi")
top = sorted((w for w in H if H[w] >= 10 and D[w] >= 10), key=lambda w: -(H[w] + D[w]))
print("top words:", ", ".join(f"{w}({H[w]}/{D[w]})" for w in top[:30]))
WH, WD = words["Haryana"], words["Delhi"]
for a, b in [("रखा", "रक्खा"), ("रखी", "रक्खी"), ("रखे", "रक्खे"), ("कूलर", "कुल्लर"), ("पता", "पत्ता")]:
    print(f"{a} / {b}: Haryana {WH[a]} / {WH[b]}, Delhi {WD[a]} / {WD[b]}")

# Save the candidate list used by the closure-duration check (>= 10 tokens in both regions).
if len(sys.argv) > 2:
    rows = []
    for w in sorted((w for w in H if H[w] >= 10 and D[w] >= 10), key=lambda w: -(H[w] + D[w])):
        stops = [w[i] for i in range(1, len(w) - 1) if n_single_stops(w[i - 1:i + 2]) and w[i] in STOPS]
        rows.append({"word": w, "haryana": H[w], "delhi": D[w], "stops": "".join(stops)})
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
    print(f"wrote {len(rows)} candidate words to {sys.argv[2]}")
