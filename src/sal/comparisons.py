"""The study's standard paired comparisons between probe configurations.

Used by scripts/summarise.py (configs carry the split, e.g. ..._speaker_split) and scripts/benchmark_eval.py
(configs carry no split). Each comparison is (a, b, what): a minus b on the same test utterances. Only pairs whose
two configs both exist are returned, so the list grows as runs arrive.

Representation-probe configs are named rp_<model>_rebuild_L<layers>_<probe> (scripts/run_rp.py), where <layers>
is the resolved layer index or indices (e.g. L24, or L1-2-4 for a learned weighted sum).
"""
from __future__ import annotations

import re

RP = re.compile(r"^rp_(?P<model>w2v2|zipa|whisper)_rebuild_L(?P<layers>[0-9]+(?:-[0-9]+)*)_(?P<probe>[a-z_]+)$")
RP_PROBES = ("attn", "attn_pe", "gru", "transformer", "chunkshuffle")


def standard(configs: set[str], suffix: str = "") -> list[tuple[str, str, str]]:
    """Standard comparisons among `configs`, for configs that end in `suffix` (e.g. "_speaker_split", or "" for the
    benchmark, whose configs carry no split)."""
    out: list[tuple[str, str, str]] = []

    def add(a: str, b: str, what: str) -> None:
        a, b = a + suffix, b + suffix
        if a in configs and b in configs:
            out.append((a, b, what))

    for m in ("w2v2", "zipa"):
        tp = f"tp_{m}_rebuild"
        add(f"{tp}_phone_bigram", f"{tp}_phone_unigram", "order (phone bigram vs unigram)")
        add(f"{tp}_char_bigram", f"{tp}_char_unigram", "order (char bigram vs unigram)")
        add(f"{tp}_phone_bigram", f"{tp}_char_bigram", "units (phones vs chars, bigram)")
        add(f"{tp}_phone_gru", f"{tp}_phone_shuffle", "order (BiGRU vs shuffled BiGRU)")
        add(f"{tp}_phone_gru", f"{tp}_char_gru", "units (phones vs chars, BiGRU)")
        add(f"{tp}_phone_gru", f"{tp}_phone_bigram", "sequence model vs bigram bag")
        add(f"{tp}_phone_bigram", "dense_ecapa_rebuild", "transcripts vs voice (phone bigram vs ECAPA)")
    for k in ("phone_bigram", "char_bigram", "phone_gru", "char_gru", "phone_shuffle"):
        add(f"tp_zipa_rebuild_{k}", f"tp_w2v2_rebuild_{k}", f"recognizer (ZIPA vs W2V2, {k})")

    # representation probes, per model and layer set actually present
    layer_sets = set()
    for c in configs:
        if not c.endswith(suffix):
            continue
        m = RP.match(c[: len(c) - len(suffix)] if suffix else c)
        if m and m["probe"] in RP_PROBES:
            layer_sets.add((m["model"], m["layers"]))
    for model, layers in sorted(layer_sets):
        rp = f"rp_{model}_rebuild_L{layers}"
        add(f"{rp}_gru", f"{rp}_chunkshuffle", "RP order (BiGRU vs chunk-shuffled BiGRU)")
        add(f"{rp}_gru", f"{rp}_attn", "RP order-aware vs order-free (BiGRU vs PRiSM's attention)")
        add(f"{rp}_attn_pe", f"{rp}_attn", "RP positions (attention with vs without)")
        add(f"{rp}_transformer", f"{rp}_attn", "RP order-aware vs order-free (Transformer vs attention)")
        if model in ("w2v2", "zipa"):
            tp = f"tp_{model}_rebuild"
            add(f"{tp}_char_gru", f"{rp}_attn", "PRiSM's comparison (transcript char BiGRU vs RP-Attn)")
            add(f"{tp}_phone_gru", f"{rp}_gru", "transcripts vs hidden states (BiGRU on both)")
    return out
