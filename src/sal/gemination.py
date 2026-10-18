"""Devanagari rules shared by the gemination scripts (RQ2).

A single voiceless stop is a candidate when it stands between vowels in the spelling (same rule as
scripts/stop_words.py). Its left context decides whether it is between vowels in speech too:
  vowel          preceded by a vowel sign or an independent vowel (दिखाई, दुकान)
  initial_schwa  preceded by the word's first consonant, whose inherent vowel is pronounced (र-खा, ra-khaa)
  medial_schwa   preceded by a later consonant, whose inherent vowel Hindi usually deletes (सकते /sakte/,
                 उसके /uske/): the stop then follows a consonant, and the measured "closure" can include it
Main analyses keep vowel and initial_schwa tokens; medial_schwa tokens are a sensitivity check.
"""
from __future__ import annotations

STOPS = set("कखचछटठतथप")
MATRA = set("ािीुूृेैोौ")
INDEP = set("अआइईउऊऋएऐओऔ")
CONS = {chr(c) for c in range(0x915, 0x93A)}


def target_positions(w: str) -> list[int]:
    """Indices of single voiceless stops between vowels in the spelling."""
    out = []
    for i in range(1, len(w) - 1):
        p, c, n = w[i - 1], w[i], w[i + 1]
        if c in STOPS and p != "्" and n not in ("्", "़") and n in MATRA and (p in MATRA or p in INDEP or p in CONS):
            out.append(i)
    return out


def left_context(w: str, pos: int) -> str:
    p = w[pos - 1]
    if p in MATRA or p in INDEP:
        return "vowel"
    return "initial_schwa" if pos == 1 else "medial_schwa"


def token_contexts(words, stops) -> list[str]:
    """Left context for alignment rows (word, stop letter), rows of one word in order of appearance."""
    out, seen = [], {}
    for w, s in zip(words, stops, strict=True):
        cands = [p for p in target_positions(w) if w[p] == s]
        k = seen.get((w, s), 0)
        out.append(left_context(w, cands[k % len(cands)]) if cands else "unknown")
        seen[(w, s)] = k + 1
    return out
