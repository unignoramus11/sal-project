"""Compare the manual closure audit with the automatic measurements (RQ2 validity check).

Reads the annotated TextGrids in internal/audit/{mohit,nidhi}/ and the key in internal/audit_key/key.csv.
For every annotator (and their mean, where both marked a token):
  - landmark error of the automatic method: |auto - human| for closure start, burst and voice onset
    (median, share within 10 and 20 ms);
  - closure and VOT durations: Pearson r, mean bias auto - human, and that bias per region. The region
    difference of the bias is the number that matters: a measurement that is wrong in the same way in both
    regions cannot create a Haryana effect, one that differs by region can.
Between annotators: the same duration statistics (inter-annotator agreement as the ceiling).
Output: results/gemination/audit_landmarks.csv, results/gemination/audit_summary.txt
Run:    uv run python scripts/audit_compare.py
"""
import re

import numpy as np
import pandas as pd

from sal.paths import RESULTS, ROOT

AUDIT, KEY = ROOT / "internal" / "audit", ROOT / "internal" / "audit_key" / "key.csv"


def marked(path, tier):
    """The annotator's interval: a labelled interval if any, else the middle one of exactly three."""
    raw = path.read_bytes()
    # Praat saves a TextGrid that holds non-Latin text (the Devanagari word tier) as UTF-16 unless told otherwise
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    text = raw.decode(enc, errors="replace")
    block = next((p for p in re.split(r'\n\s*item \[\d+\]:', text) if f'name = "{tier}"' in p), "")
    ivs = [(float(a), float(b), t.strip()) for a, b, t in
           re.findall(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"', block)]
    labelled = [(a, b) for a, b, t in ivs if t]
    if labelled:
        return labelled[0]
    if len(ivs) == 3:                       # boundaries drawn but no label typed
        return ivs[1][:2]
    return (np.nan, np.nan)


def read(who):
    rows = []
    for tg in sorted((AUDIT / who).glob("audit_*.TextGrid")):
        c0, c1 = marked(tg, "closure")
        v0, v1 = marked(tg, "vot")
        rows.append({"clip": tg.stem, "closure_start": c0, "burst": c1 if np.isfinite(c1) else v0,
                     "voice_onset": v1})
    d = pd.DataFrame(rows)
    d["closure_ms"] = 1000 * (d["burst"] - d["closure_start"])
    d["vot_ms"] = 1000 * (d["voice_onset"] - d["burst"])
    return d.set_index("clip")


def durations(a, b, region, label):
    out = []
    for col in ("closure_ms", "vot_ms"):
        m = a[col].notna() & b[col].notna()
        if m.sum() < 3:
            continue
        diff = a.loc[m, col] - b.loc[m, col]
        by = diff.groupby(region[m]).mean()
        out.append(f"{label} {col}: n={m.sum()}, r={np.corrcoef(a.loc[m, col], b.loc[m, col])[0, 1]:.3f}, "
                   f"bias={diff.mean():+.1f} ms, mean |diff|={diff.abs().mean():.1f} ms; bias by region "
                   + ", ".join(f"{k} {v:+.1f}" for k, v in by.items())
                   + (f"; region difference {by.get('Haryana', np.nan) - by.get('Delhi', np.nan):+.1f} ms"))
    return out


def main():
    key = pd.read_csv(KEY).set_index("clip")
    auto = key.rename(columns={"auto_closure_start": "closure_start", "auto_burst": "burst",
                               "auto_voice_onset": "voice_onset", "auto_closure_ms": "closure_ms",
                               "auto_vot_ms": "vot_ms"})
    ann = {w: read(w) for w in ("mohit", "nidhi") if (AUDIT / w).exists()}
    ann = {w: d for w, d in ann.items() if d["closure_ms"].notna().any()}
    if len(ann) == 2:
        a, b = ann["mohit"], ann["nidhi"]
        ann["mean"] = (a + b) / 2                          # NaN unless both marked the landmark
    lines, rows = [], []
    for who, h in ann.items():
        h = h.reindex(key.index)
        for lm in ("closure_start", "burst", "voice_onset"):
            e = 1000 * (auto[lm] - h[lm]).abs()
            e = e[e.notna()]
            rows.append({"annotator": who, "landmark": lm, "n": len(e), "median_abs_ms": e.median(),
                         "within_10ms": (e <= 10).mean(), "within_20ms": (e <= 20).mean()})
        lines += durations(auto, h, key["region"], f"auto vs {who}")
    if {"mohit", "nidhi"} <= ann.keys():
        lines += durations(ann["mohit"].reindex(key.index), ann["nidhi"].reindex(key.index), key["region"],
                           "mohit vs nidhi")
    out = RESULTS / "gemination"
    pd.DataFrame(rows).to_csv(out / "audit_landmarks.csv", index=False)
    (out / "audit_summary.txt").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
