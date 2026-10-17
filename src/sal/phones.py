"""Phone segmentation of recognizer output, and the doubled-phone rate (DPR).

The two recognizers write IPA differently: wav2vec2 emits eSpeak phonemes as tokens (`oː`, `tʃ`, `ɔːɹ`),
ZIPA emits characters (`o` + `ː`, `p` + `ʰ`). Both are re-segmented here by one rule so that "phone"
means the same thing for both:

1. Unicode NFD, so precomposed letters split into base + combining mark; ASCII `g` -> IPA `ɡ`;
   whitespace and the SentencePiece word marker `▁` are removed.
2. Every combining mark (category Mn, e.g. dental ̪, nasal ̃, tie bar ͡) and every modifier letter
   (category Lm, e.g. ʰ ʱ ʲ ʷ ː ˑ) attaches to the preceding segment.
3. `tʃ` and `dʒ` (the Hindi affricates च, ज) are joined into one segment.

Segmentation is lossless: "".join(segments) equals the normalised string.

DPR. A consonant is geminate if it carries a length mark (`kː`) or is part of a run of consonant segments
that are identical up to aspiration (`kk`, and `kkʰ`, the spelling of a geminate aspirate as in Bangru
[ɾək.kʰaː]). A run counts once. DPR = geminate consonants / consonants, both counted after
collapsing runs. Vowels are identified by their base letter (VOWELS below).
"""
from __future__ import annotations

import unicodedata

VOWELS = set("aeiouyæøœɐɑɒɔəɘɚɛɜɝɞɤɨɪɯɵɶʉʊʌʏᵻᵿ")
AFFRICATES = {("t", "ʃ"), ("d", "ʒ")}
DROP = set(" \t\n▁")


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join("ɡ" if c == "g" else c for c in s if c not in DROP)


def _attaches(c: str) -> bool:
    return unicodedata.category(c) in ("Mn", "Lm")


def segment(s: str) -> list[str]:
    s = normalise(s)
    segs: list[str] = []
    for c in s:
        if segs and _attaches(c):
            segs[-1] += c
        elif segs and (segs[-1], c) in AFFRICATES:
            segs[-1] += c
        else:
            segs.append(c)
    return segs


def base(seg: str) -> str:
    return seg[0]


def is_vowel(seg: str) -> bool:
    return base(seg) in VOWELS


def is_long(seg: str) -> bool:
    return "ː" in seg


ASPIRATION = ("ʰ", "ʱ")


def stem(seg: str) -> str:
    """Segment without aspiration marks: a geminate aspirate is written k + kʰ (closure, then release)."""
    return "".join(c for c in seg if c not in ASPIRATION)


def consonant_events(segs: list[str]) -> list[tuple[str, bool]]:
    """Collapse runs of consonants that are identical up to aspiration (kk, kkʰ, kʰkʰ); return
    (last segment of the run, geminate?) per consonant event."""
    out: list[tuple[str, bool]] = []
    i = 0
    while i < len(segs):
        s = segs[i]
        if is_vowel(s) or not s.isprintable():
            i += 1
            continue
        j = i
        while j + 1 < len(segs) and not is_vowel(segs[j + 1]) and stem(segs[j + 1]) == stem(s):
            j += 1
        out.append((segs[j], j > i or is_long(s) or is_long(segs[j])))
        i = j + 1
    return out


def dpr(s: str) -> tuple[int, int]:
    """(geminate consonants, consonants) for one transcript."""
    ev = consonant_events(segment(s))
    return sum(g for _, g in ev), len(ev)
