"""Mid-evaluation deck (3 Nov 2026), built from results/ in the style of our seminar decks
(internal/notes/presentation_style.md): 16:9, sage content slides, deep-green title and takeaways, kicker +
claim title, Carlito body with en-dash bullets, italic grey captions, page numbers.

Output: slides/mid/mid_evaluation.pptx (+ .pdf with --pdf, via LibreOffice)
Run:    uv run --with python-pptx python slides/mid/build_deck.py --pdf
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
DATA = ROOT / "data"
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)

SAGE, DEEP, KICK, INK, BODY, CAP, LIGHT = "e2f0d9", "12352a", "5c8f72", "12352a", "26332d", "6e7b74", "e4ede7"
SERIF, SANS, MONO = "Caladea", "Carlito", "Liberation Mono"
W, H = Inches(13.333), Inches(7.5)
LM = Inches(0.6)
TW = Inches(12.1)
BODY_PT = 16

cfg = pd.read_csv(RES / "tables" / "configs.csv").set_index("config")
base = pd.read_csv(RES / "baselines.csv")
comp = pd.read_csv(RES / "tables" / "comparisons.csv").set_index(["a", "b"])
ctrl = pd.read_csv(RES / "tables" / "control.csv").set_index("probe")
mdl = pd.read_csv(RES / "mdl" / "summary.csv").set_index("config")
state = pd.read_csv(RES / "tables" / "state_level.csv").set_index("config")
knn = pd.read_csv(RES / "tables" / "knn_transcripts.csv").set_index(["transcripts", "protocol", "k"])
OFF = RES / "official"
CHK = RES / "rebuild_checks"


def rgb(h):
    return RGBColor.from_string(h)


def bl(name, proto, prompts="all", split="test"):
    r = base[(base.baseline == name) & (base.protocol == proto) & (base.prompts == prompts) & (base.split == split)]
    return r.iloc[0]


def cd(a, b, unit="recall1"):
    r = comp.loc[(a, b)]
    return f"{r[unit + '_diff']:+.1f} [{r[unit + '_lo']:.1f}, {r[unit + '_hi']:.1f}]"


def seedmean(c):
    v = [json.loads((d / "metrics.json").read_text())["metrics"]["test"]["recall1"]
         for d in sorted((RES / "runs").glob(f"{c}_s*")) if (d / "metrics.json").exists()]
    if not v:
        return float("nan"), 0.0, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


# ------------------------------------------------------------------ slide primitives

class Deck:
    def __init__(self):
        self.p = Presentation()
        self.p.slide_width, self.p.slide_height = W, H
        self.n = 0

    def _slide(self, dark=False):
        s = self.p.slides.add_slide(self.p.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(DEEP if dark else SAGE)
        self.n += 1
        if self.n > 1:
            self.text(s, str(self.n), W - Inches(0.9), H - Inches(0.45), Inches(0.5), Inches(0.3), size=9,
                      color=LIGHT if dark else "9aa79f", align=PP_ALIGN.RIGHT)
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=BODY_PT, color=BODY, font=SANS, bold=False, italic=False, align=PP_ALIGN.LEFT,
             spacing=None, anchor=MSO_ANCHOR.TOP):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            r = para.add_run()
            r.text = line
            f = r.font
            f.size, f.name, f.bold, f.italic = Pt(size), font, bold, italic
            f.color.rgb = rgb(color)
            if spacing:
                r._r.get_or_add_rPr().set("spc", str(spacing))
        return tb

    def header(self, s, kicker, title, dark=False):
        if kicker:
            self.text(s, kicker.upper(), LM, Inches(0.38), Inches(12), Inches(0.25), size=9,
                      color=KICK if not dark else "8fb5a0", bold=True, spacing=200)
        self.text(s, title, LM, Inches(0.68), TW, Inches(0.8), size=28, color=LIGHT if dark else INK,
                  font=SERIF, bold=True)

    def bullets(self, s, items, x=LM, y=Inches(1.8), w=TW, h=Inches(4.9), size=BODY_PT, color=BODY, gap=11):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        for i, it in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.space_after = Pt(gap)
            para.line_spacing = 1.15
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.32))))
            pPr.set("indent", str(-Emu(Inches(0.32))))
            pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": SANS}))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "–"}))
            r = para.add_run()
            r.text = it
            r.font.size, r.font.name = Pt(size), SANS
            r.font.color.rgb = rgb(color)
        return tb

    def image(self, s, path, x, y, w=None, h=None):
        return s.shapes.add_picture(str(path), x, y, width=w, height=h)

    def caption(self, s, txt, x, y, w, align=PP_ALIGN.LEFT):
        self.text(s, txt, x, y, w, Inches(0.3), size=10.5, color=CAP, italic=True, align=align)

    def stats(self, s, items, y=Inches(1.75), best=None):
        cw = TW / len(items)
        for i, (num, lab) in enumerate(items):
            x = LM + cw * i
            self.text(s, num, x, y, cw - Inches(0.2), Inches(0.8), size=38, font=SERIF, bold=True,
                      color=KICK if best == i else INK)
            self.text(s, lab, x, y + Inches(0.82), cw - Inches(0.3), Inches(0.8), size=12.5, color=CAP)

    def deflist(self, s, pairs, y=Inches(1.8), label_w=Inches(2.3), size=15, row=Inches(1.0)):
        for k, (lab, txt) in enumerate(pairs):
            yy = y + row * k
            self.text(s, lab, LM, yy, label_w, row, size=size, font=SERIF, bold=True, color=INK)
            self.text(s, txt, LM + label_w + Inches(0.2), yy, TW - label_w - Inches(0.2), row, size=size - 0.5)

    def save(self, path):
        self.p.save(path)


# ------------------------------------------------------------------ tables as booktabs images

def table_png(name, header, rows, col_align=None, bold_cells=(), group_rows=(), widths=None, fontsize=11):
    """Booktabs-style table rendered with matplotlib (Times-like serif), transparent background."""
    ncol = len(header)
    widths = widths or [1.0] * ncol
    tot = sum(widths)
    nrows = len(rows) + 1
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times", "Nimbus Roman", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(0.95 * tot, 0.27 * (nrows + 0.6)))
    ax.set_axis_off()
    ax.set_xlim(0, tot)
    ax.set_ylim(nrows + 0.3, -0.3)
    xs = [sum(widths[:i]) for i in range(ncol)]
    col_align = col_align or ["left"] + ["right"] * (ncol - 1)

    def put(i, j, txt, bold=False, italic=False):
        a = col_align[j]
        x = xs[j] + (0.05 if a == "left" else widths[j] - 0.05 if a == "right" else widths[j] / 2)
        ax.text(x, i, txt, ha=a if a != "center" else "center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal", style="italic" if italic else "normal", color="#111111")

    for j, h_ in enumerate(header):
        put(0, j, h_, bold=True)
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            put(i, j, v, bold=(i - 1, j) in bold_cells, italic=(i - 1) in group_rows and j == 0)
    ax.plot([0, tot], [-0.45, -0.45], color="black", lw=1.2)
    ax.plot([0, tot], [0.5, 0.5], color="black", lw=0.6)
    ax.plot([0, tot], [nrows - 0.5, nrows - 0.5], color="black", lw=1.2)
    for g in group_rows:
        ax.plot([0, tot], [g + 0.5, g + 0.5], color="#999999", lw=0.4)
    out = BUILD / f"{name}.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def fmt(m, sd=None):
    return f"{m:.1f}" if sd is None else f"{m:.1f} ± {sd:.1f}"


# ------------------------------------------------------------------ content

def build():
    d = Deck()
    splits = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")
    prompts = pd.read_parquet(DATA / "vaani_hi_rebuild" / "prompts.parquet")
    per_spk = splits.groupby("speaker").size()
    leak = 100 * splits.loc[splits.prism == "test", "speaker"].isin(set(splits.loc[splits.prism == "train", "speaker"])).mean()
    specific = 100 * (prompts.prompt_type == "SPECIFIC").mean()
    pic = pd.read_csv(CHK / "picture_lookup.csv").set_index(["prompts", "protocol", "baseline"])
    look = pd.read_csv(CHK / "lookalike.csv").set_index("probe")
    coords = pd.read_csv(CHK / "coordinates.csv")
    ocl = pd.read_csv(OFF / "scores_clean.csv")
    odup = pd.read_csv(OFF / "scores_by_duplication.csv").set_index("probe")
    oorder = pd.read_csv(OFF / "order_clean.csv").set_index(["protocol", "a"])
    NN = "nearest-neighbour transcript (1-NN)"

    # 1. title
    s = d._slide(dark=True)
    d.text(s, "MID EVALUATION  ·  PROJECT PROGRESS", LM, Inches(1.55), Inches(12), Inches(0.3), size=9,
           color="8fb5a0", bold=True, spacing=200)
    d.text(s, "Order, Duration, or Melody?", LM, Inches(2.0), Inches(12), Inches(0.9), size=42, font=SERIF,
           bold=True, color="ffffff")
    d.text(s, "Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech", LM, Inches(2.95),
           Inches(12), Inches(0.5), size=19, font=SERIF, italic=True, color=LIGHT)
    ln = s.shapes.add_connector(1, LM, Inches(3.8), LM + Inches(2.4), Inches(3.8))
    ln.line.color.rgb = rgb(KICK)
    d.text(s, "Mohit Kumar Singh and Nidhi Vaidya", LM, Inches(4.1), Inches(12), Inches(0.3), size=14, color=LIGHT)
    d.text(s, "Base paper: PRiSM, Bharadwaj et al., ACL 2026 (GEO-v, Vaani-Hi dialect geolocation)", LM,
           Inches(4.5), Inches(12), Inches(0.3), size=14, color=LIGHT)
    d.text(s, "3 November 2026", LM, Inches(5.6), Inches(12), Inches(0.3), size=12, color="9fb8aa")

    # 2. problem
    s = d._slide()
    d.header(s, "Problem", "PRiSM's transcript probe beats its representation probe on GEO-v")
    d.bullets(s, [
        "GEO-v asks a probe to place a Hindi-belt utterance on the map (Vaani-Hi, about 30 pincodes over ~1,500 km). "
        "PRiSM reports 146 km mean error for a probe over predicted phone strings and 253 km for a probe over the "
        "recognizer's hidden states.",
        "Only the representation probe hears voice, intonation and duration, yet it loses. Either phone strings carry "
        "more usable location signal, or the evaluation lets one probe succeed for other reasons.",
        "The authors credit phone order: the transcript probe is an RNN that reads the sequence, the representation "
        "probe pools frames. This was not tested.",
        "We test what the phone string carries (order, duration, topic, speaker) and whether the protocol rewards "
        "dialect at all.",
    ])

    # 3. objectives
    s = d._slide()
    d.header(s, "Objectives", "Four questions, each tied to a measurable property of speech")
    d.deflist(s, [
        ("RQ0  Speakers", "Does PRiSM's random split let probes recognise speakers and topics instead of dialects? "
                          "Speaker-disjoint split, speaker-only and picture-only baselines, control task."),
        ("RQ1  Order", "Does local phone order carry location beyond the phone inventory? Unigram vs bigram vs BiGRU "
                       "vs shuffled BiGRU, characters vs phones, two recognizers, MDL."),
        ("RQ2  Duration", "Is Haryanvi consonant lengthening in the recognizers' output, and in the audio? "
                          "Doubled-phone rate, then closure durations in Haryana vs New Delhi."),
        ("RQ3  Melody", "Does intonation, which transcripts discard, add location signal? Prosody probe and late "
                        "fusion with the transcript probe."),
    ], row=Inches(1.15))

    # 4. data
    s = d._slide()
    d.header(s, "Data", "Two versions of Vaani-Hi: our rebuild and PRiSM's release")
    d.stats(s, [(f"{len(splits):,}", "utterances in our rebuild\n(40.3 h, 26 pincodes, 9 states)"),
                (f"{len(per_spk):,}", f"speakers (codes from file names;\nmedian {per_spk.median():.0f}, "
                                      f"max {per_spk.max()} utterances each)"),
                ("26,385", "rows in PRiSM's release (access\n21 Oct), 30 pincodes, no speaker IDs"),
                (f"{specific:.0f}%", "of utterances describe a picture\nnamed after the speaker's district")])
    d.bullets(s, [
        "Rebuild: PRiSM's own selection code rerun on the public Vaani corpus at a pinned commit; a second machine "
        "reproduces it bit for bit. Speaker codes make speaker-disjoint evaluation possible.",
        "Release: PRiSM's GEO-v copy, used to reproduce its number and audit the benchmark as published.",
        "Speakers describe pictures, and two thirds of the pictures belong to the speaker's own district, so the "
        "topic of an utterance can reveal its location without any dialect feature.",
    ], y=Inches(3.65), h=Inches(3.2))

    # 5. method
    s = d._slide()
    d.header(s, "Method", "The same probes and references under two splits")
    d.deflist(s, [
        ("Splits", "PRiSM's random split within pincode, and a speaker-disjoint split stratified by pincode."),
        ("Probes", "PRiSM's TP-GRU (2-layer BiGRU over characters) with its settings; the same over phones and over "
                   "shuffled phones; MLPs over unigram or bigram counts; MLP over ECAPA voice embeddings."),
        ("References", "Majority location; a lookup by speaker ID; a lookup by picture ID; a nearest-neighbour "
                       "transcript matcher that learns nothing about location; a control task with permuted labels."),
        ("Recognizers", "W2V2 (wav2vec2-lv-60-espeak-cv-ft) and ZIPA-CR-CTC-NS large, greedy CTC, PRiSM's wrappers."),
        ("Statistics", "Recall@1 (PRiSM's metric) and mean km; 5 seeds; paired 95% CIs by bootstrap over speakers."),
    ], label_w=Inches(2.1), size=15, row=Inches(0.95))

    # 6. RQ0 speakers
    lk = ctrl.loc["dense_ecapa_rebuild_prism"]
    s = d._slide()
    d.header(s, "Results  ·  RQ0 speakers", "Under PRiSM's split, the speaker ID alone solves the task")
    d.stats(s, [(f"{leak:.1f}%", "test utterances whose speaker\nis also in training"),
                (f"{bl('speaker_lookup', 'prism').recall1:.1f}", "Recall@1 of a lookup that only\nknows the speaker ID"),
                (f"{cfg.loc['dense_ecapa_rebuild_prism', 'recall1_mean']:.1f} → "
                 f"{cfg.loc['dense_ecapa_rebuild_speaker_split', 'recall1_mean']:.1f}",
                 "voice embedding (ECAPA),\nrandom → speaker-disjoint"),
                (f"{lk.control_recall1:.1f}", f"ECAPA on control labels\n(control majority {lk.control_majority:.1f})")],
            best=1)
    d.bullets(s, [
        f"Majority baseline {bl('majority', 'prism').recall1:.1f} (random) and {bl('majority', 'speaker_split').recall1:.1f} "
        "(speaker-disjoint). A voice embedding trained to ignore content falls to the majority on new speakers.",
        "On control labels (each speaker gets another speaker's location) ECAPA still scores far above the control "
        "majority: most of its random-split score is speaker recognition.",
        f"Transcripts identify speakers too: a nearest-neighbour matcher reaches {knn.loc[('w2v2', 'prism', 1), 'recall1']:.1f} "
        f"on the random split and {knn.loc[('w2v2', 'speaker_split', 1), 'recall1']:.1f} on new speakers; for 25% of "
        "test utterances the closest training transcript is the same speaker's (chance 0.3%).",
    ], y=Inches(3.7), h=Inches(3.2))

    # 7. RQ0 topic: the picture alone
    pr, ps = pic.loc[("all", "prism", "picture lookup")], pic.loc[("all", "speaker_split", "picture lookup")]
    gs, gm = pic.loc[("shared", "speaker_split", "picture lookup")], pic.loc[("shared", "speaker_split", "majority")]
    s = d._slide()
    d.header(s, "Results  ·  RQ0 topic", "The picture a speaker describes reveals the location")
    d.stats(s, [(f"{pr.recall1:.1f}", f"picture-ID lookup, random split\n(majority {pic.loc[('all', 'prism', 'majority')].recall1:.1f}; PRiSM's probe 45.7)"),
                (f"{ps.recall1:.1f}", f"picture-ID lookup, new speakers\n(majority {pic.loc[('all', 'speaker_split', 'majority')].recall1:.1f})"),
                (f"{gs.recall1:.1f}", f"picture-ID lookup on shared\npictures, new speakers (majority {gm.recall1:.1f})")],
            best=0)
    d.bullets(s, [
        "The lookup predicts the most frequent training pincode among utterances describing the same picture. It "
        "uses no audio, yet it beats every probe on both splits.",
        "District-specific pictures are shown to all speakers of that district, so naming the picture names the "
        f"place. On new speakers the lookup ({ps.recall1:.1f}) beats our best transcript probe "
        f"({cfg.loc['tp_zipa_rebuild_phone_bigram_speaker_split', 'recall1_mean']:.1f}).",
        "Shared pictures are described across states, and there the lookup falls below the majority: utterances about "
        "shared pictures are a valid topic control.",
    ], y=Inches(3.7), h=Inches(3.2))

    # 8. main table
    rows, groups = [], []
    for name, b in [("Majority baseline", "majority"), ("Speaker lookup (ID only)", "speaker_lookup")]:
        rows.append([name, fmt(bl(b, "prism").recall1), f"{bl(b, 'prism').mean_km:.0f}",
                     fmt(bl(b, "speaker_split").recall1), f"{bl(b, 'speaker_split').mean_km:.0f}"])
    rows.append(["Picture lookup (ID only)", fmt(pr.recall1), "", fmt(ps.recall1), ""])
    for name, key in [("ECAPA voice embedding", "dense_ecapa_rebuild"), ("Prosody statistics", "dense_prosody_rebuild")]:
        a, b = cfg.loc[f"{key}_prism"], cfg.loc[f"{key}_speaker_split"]
        rows.append([name, fmt(a.recall1_mean, a.recall1_sd), f"{a.ens_km:.0f}", fmt(b.recall1_mean, b.recall1_sd), f"{b.ens_km:.0f}"])
    for rec, m in (("W2V2", "w2v2"), ("ZIPA", "zipa")):
        groups.append(len(rows))
        rows.append([f"Transcript probes, {rec}", "", "", "", ""])
        for lab, k in [("phone unigram", "phone_unigram"), ("phone bigram", "phone_bigram"),
                       ("char BiGRU (PRiSM)", "char_gru"), ("phone BiGRU", "phone_gru"), ("shuffled phone BiGRU", "phone_shuffle")]:
            a, b = cfg.loc[f"tp_{m}_rebuild_{k}_prism"], cfg.loc[f"tp_{m}_rebuild_{k}_speaker_split"]
            rows.append(["   " + lab, fmt(a.recall1_mean, a.recall1_sd), f"{a.ens_km:.0f}", fmt(b.recall1_mean, b.recall1_sd),
                         f"{b.ens_km:.0f}"])
    probe_rows = [(i, float(r[3].split(" ")[0])) for i, r in enumerate(rows) if r[3] and i > 2]
    png = table_png("table_main", ["Probe", "R@1 random", "km", "R@1 new speakers", "km"], rows,
                    bold_cells=[(max(probe_rows, key=lambda t: t[1])[0], 3)], group_rows=groups,
                    widths=[2.5, 1.3, 0.75, 1.8, 0.75], fontsize=10.5)
    s = d._slide()
    d.header(s, "Results  ·  rebuilt Vaani-Hi", "Every probe drops on new speakers; phone bigrams keep the most")
    im = d.image(s, png, LM, Inches(1.55), h=Inches(5.1))
    tx = LM + im.width + Inches(0.45)
    d.bullets(s, [
        "Recall@1 is mean ± sd over seeds (BiGRUs: 2 seeds so far); km is the seed ensemble's mean error.",
        "ECAPA loses 31 points between splits, the bag probes 3-13: transcripts carry location beyond the voice.",
        "On new speakers, ZIPA phone bigrams beat W2V2 phone bigrams by "
        f"{cd('tp_zipa_rebuild_phone_bigram_speaker_split', 'tp_w2v2_rebuild_phone_bigram_speaker_split')} Recall@1.",
    ], x=tx, y=Inches(1.7), w=W - tx - Inches(0.6), h=Inches(4.8), size=14.5)
    d.caption(s, "Table 1. Test Recall@1 (higher is better) and mean km (lower is better).", LM, Inches(6.8), im.width)

    # 9. lift figure
    s = d._slide()
    d.header(s, "Results  ·  common reference", "On new speakers, only bigrams beat the majority baseline")
    d.image(s, RES / "figures" / "fig_probes.png", LM, Inches(1.45), h=Inches(5.2))
    tx = LM + Inches(8.6)
    d.bullets(s, [
        "Left: Recall@1 minus the majority on the same utterances. Right: km saved over the best fixed location.",
        f"New speakers: ZIPA phone bigram {cfg.loc['tp_zipa_rebuild_phone_bigram_speaker_split', 'lift_recall1']:+.1f}, "
        f"W2V2 phone bigram {cfg.loc['tp_w2v2_rebuild_phone_bigram_speaker_split', 'lift_recall1']:+.1f}, ECAPA "
        f"{cfg.loc['dense_ecapa_rebuild_speaker_split', 'lift_recall1']:+.1f} points.",
        "Nearly every probe still saves 130-200 km on new speakers: coarse regional information survives.",
    ], x=tx, y=Inches(1.7), w=W - tx - Inches(0.6), h=Inches(4.8), size=14.5)
    d.caption(s, "Figure 1. Seed ensemble, 95% speaker-bootstrap CI. Open markers: trained and tested on shared pictures.",
              LM, Inches(6.8), Inches(8.3))

    # 10. topic, shared pictures
    g = cfg.loc["tp_w2v2_rebuild_phone_bigram_speaker_split_generic"]
    st = state.loc["tp_zipa_rebuild_phone_bigram_speaker_split"]
    s = d._slide()
    d.header(s, "Analysis  ·  topic", "Without topic, the transcript probes fall to the majority")
    d.stats(s, [(f"{st.all_state_acc:.0f}%", f"correct state, ZIPA phone bigrams,\nnew speakers (majority {st.all_majority:.0f}%)"),
                (f"{g.lift_recall1:+.1f}", "Recall@1 lift when trained and\ntested on shared pictures only"),
                (f"+{g.km_gain:.0f} km", f"km gain on shared pictures\n(95% CI {g.km_gain_lo:.0f} to {g.km_gain_hi:.0f})")])
    d.bullets(s, [
        "Shared pictures carry no location (previous check), but their share differs by state (12-58%), so they get "
        "their own baselines: majority 21.3 Recall@1, best constant 537 km.",
        "Trained and tested on shared pictures, no transcript probe beats the majority on Recall@1 or state; km gains "
        "are 48-94 km, and only the two W2V2 ones have CIs above zero (119 test speakers).",
        "Most of the transcript signal on new speakers is what people describe; what survives is regional.",
    ], y=Inches(3.7), h=Inches(3.2))

    # 11. checks of our own split
    e, z, w = (look.loc[c] for c in ("dense_ecapa_rebuild_speaker_split", "tp_zipa_rebuild_phone_bigram_speaker_split",
                                      "tp_w2v2_rebuild_phone_bigram_speaker_split"))
    rb = coords[(coords.data == "rebuild")].sort_values("km_from_cluster", ascending=False).iloc[0]
    s = d._slide()
    d.header(s, "Analysis  ·  checking our own split", "Look-alike speaker codes leak the voice, not the transcript")
    d.stats(s, [(f"{look.iloc[0].lookalike_n / look.iloc[0].all_n:.0%}", "of new-speaker test utterances have a\nlook-alike code in training"),
                (f"{e.lookalike_lift:+.1f} / {e.strict_lift:+.1f}", "ECAPA lift: look-alike utterances /\nthe strict rest"),
                (f"{z.strict_lift:+.1f}", f"ZIPA phone bigram lift on the strict\nrest ({z.all_lift:+.1f} on all)"),
                (f"{rb.km_from_cluster:.0f} km", f"pincode {int(rb.pincode)} ({rb.district}) off its\npost offices in the rebuild")],
            best=2)
    d.bullets(s, [
        "Speaker codes come from file names. Codes of one district whose voices are as close as two halves of one "
        "speaker's recordings are grouped (1,570 codes → 1,327 groups); gender agrees in 98% of close pairs.",
        f"ECAPA's small lift on new speakers comes only from look-alike codes; the transcript probes keep theirs "
        f"(W2V2 phone bigram {w.strict_lift:+.1f} on the strict rest).",
        "PRiSM's recipe averages a pincode's post offices, and some directory rows are hundreds of km off; Dhar lands "
        "88 km from Bhopal instead of 45 km from the other Dhar pincode. Recall@1 hardly moves; km does.",
    ], y=Inches(3.7), h=Inches(3.2))

    # 12. official release: data problems
    audit = (OFF / "audit.txt").read_text()
    dups = re.search(r"copies with different pincodes ([\d.]+)%", audit)[1]
    oleak = pd.read_csv(OFF / "leakage.csv").set_index("split")
    sh = pd.read_csv(OFF / "coordinates_audit.csv").iloc[0]
    s = d._slide()
    d.header(s, "Results  ·  PRiSM's release", "The released data repeats recordings and misplaces its largest class")
    d.stats(s, [(f"{100 * oleak.loc['test', 'exact_audio_in_train']:.0f}%", "test clips whose exact audio\nis also in training"),
                (f"{float(dups):.0f}%", "of repeated recordings carry\ndifferent pincodes in different rows"),
                (f"{sh.km_from_valid_median:.0f} km", f"error in the location of the largest\nclass (Shimla, {int(sh.rows):,} clips)")])
    d.bullets(s, [
        "26,385 rows hold only 21,481 distinct recordings. PRiSM's sampling script writes each district's first 600 "
        "clips from four shards to the same file names, so shards overwrite each other's audio while the metadata "
        "keeps all four label sets.",
        "Shimla is placed in Afghanistan: the release averages its 16 post offices, three of which have latitude and "
        "longitude swapped. Mean-km numbers on GEO-v are dominated by this point.",
    ], y=Inches(3.7), h=Inches(3.2))

    # 13. official release: scores
    gru, bg, ec = seedmean("tp_w2v2_official_char_gru_prism"), seedmean("tp_w2v2_official_phone_bigram_prism"), \
        seedmean("dense_ecapa_official_prism")
    s = d._slide()
    d.header(s, "Results  ·  PRiSM's release", "On PRiSM's own data, the voice alone beats PRiSM's probe")
    d.stats(s, [(f"{gru[0]:.1f}", f"PRiSM's TP-GRU with our code\n({gru[2]} seeds; paper: 45.7)"),
                (f"{bg[0]:.1f}", "bag of W2V2 phone bigrams\n(released split)"),
                (f"{ec[0]:.1f}", "ECAPA voice embedding,\nno words (released split)"),
                (f"{odup.loc[NN, 'all_recall1']:.1f}", "nearest-neighbour transcript\nmatch, nothing learned")], best=2)
    d.bullets(s, [
        "Our pipeline reproduces PRiSM's number on its data; the gap to our rebuild (about 21) comes from the data.",
        "A voice embedding that hears no words, and a matcher that learns nothing about location, both beat the "
        "probe the paper reports. The released split rewards recognising speakers.",
        f"ZIPA transcripts give the same picture (phone bigrams {seedmean('tp_zipa_official_phone_bigram_prism')[0]:.1f}).",
    ], y=Inches(3.7), h=Inches(3.2))

    # 14. corrected release, table
    def oc(proto, probe):
        return ocl[(ocl.protocol == proto) & (ocl.probe == probe)].recall1.iloc[0]
    orows = [["Majority", fmt(odup.loc["majority", "all_recall1"]), fmt(oc("prism", "majority")), fmt(oc("speaker_split", "majority"))],
             ["1-NN transcript match", fmt(odup.loc[NN, "all_recall1"]), fmt(oc("prism", NN)), fmt(oc("speaker_split", NN))]]
    for lab, key in [("ECAPA voice embedding", "dense_ecapa_official"), ("W2V2 phone bigram", "tp_w2v2_official_phone_bigram"),
                     ("ZIPA phone bigram", "tp_zipa_official_phone_bigram"), ("W2V2 char BiGRU (PRiSM)", "tp_w2v2_official_char_gru"),
                     ("W2V2 phone BiGRU", "tp_w2v2_official_phone_gru"), ("W2V2 shuffled phone BiGRU", "tp_w2v2_official_phone_shuffle")]:
        cells = []
        for c in (f"{key}_prism", f"{key}_prism_clean", f"{key}_speaker_split_clean"):
            m_, sd_, n_ = seedmean(c)
            cells.append(fmt(m_, sd_) if n_ else "–")
        orows.append([lab] + cells)
    png = table_png("table_official", ["Probe", "Released", "Corrected", "Corrected, new speakers"],
                    orows, widths=[2.7, 1.4, 1.9, 2.5], fontsize=12)
    ntest = ocl[ocl.protocol == "speaker_split"].test_pseudo_speakers.iloc[0]
    s = d._slide()
    d.header(s, "Results  ·  corrected release", "Once speakers are separated, every probe falls to the majority")
    im = d.image(s, png, LM, Inches(1.65), w=Inches(7.9))
    tx = LM + im.width + Inches(0.45)
    d.bullets(s, [
        "Corrected: one row per recording, conflicting labels removed, Shimla moved to its offices (19,731 rows).",
        "No speaker IDs, so speakers are clustered from voice embeddings; checked on our rebuild, this split leaks "
        "2.1% of test speakers.",
        f"Removing duplicates changes little; removing shared speakers removes almost everything ({ntest} test "
        "pseudo-speakers, so CIs are wide).",
    ], x=tx, y=Inches(1.7), w=W - tx - Inches(0.6), h=Inches(4.8), size=14.5)
    d.caption(s, "Table 2. Test Recall@1, mean ± sd over seeds. Released: data and split as published. Corrected: released "
                 "split on the corrected rows, or a pseudo-speaker-disjoint split.", LM, Inches(6.8), Inches(12))

    # 15. RQ1 order and units (rebuild)
    k = lambda c: mdl.loc[c, "gain_over_prior"]  # noqa: E731
    s = d._slide()
    d.header(s, "Results  ·  RQ1 order and units", "On new speakers, bigrams and phone units help a little")
    d.stats(s, [(cd("tp_w2v2_rebuild_phone_bigram_speaker_split", "tp_w2v2_rebuild_phone_unigram_speaker_split").split(" ")[0],
                 "phone bigram vs unigram, W2V2,\nnew speakers (Recall@1)"),
                (cd("tp_w2v2_rebuild_phone_bigram_speaker_split", "tp_w2v2_rebuild_char_bigram_speaker_split").split(" ")[0],
                 "phones vs characters, W2V2 bigrams,\nnew speakers"),
                (f"{k('mdl_zipa_rebuild_phone_bigram_speaker_split'):.1f} kbits",
                 f"MDL saving, ZIPA phone bigrams, new\nspeakers ({k('mdl_zipa_rebuild_phone_bigram_prism'):.1f} on the random split)")])
    d.bullets(s, [
        f"Bigram over unigram {cd('tp_w2v2_rebuild_phone_bigram_speaker_split', 'tp_w2v2_rebuild_phone_unigram_speaker_split')} "
        "Recall@1 and 33 km less error; MDL agrees on every speaker ordering.",
        "PRiSM tokenises by character, which splits kʰ into k + ʰ. Phones beat characters for W2V2 on Recall@1; in "
        "MDL, phone units help ZIPA (0.7 kbits) but not W2V2.",
        "Bigrams also identify words, and on shared pictures no probe beats the majority: the bigram advantage may be "
        "topic, not phonotactics. The 5-fold benchmark (1,327 speaker groups) will settle it.",
    ], y=Inches(3.7), h=Inches(3.2))

    # 16. RQ1 sequence model
    o_rel, o_sp = (oorder.loc[(P, "tp_w2v2_official_phone_gru")] for P in ("prism", "speaker_split"))
    r_w, r_z = comp.loc[("tp_w2v2_rebuild_phone_gru_speaker_split", "tp_w2v2_rebuild_phone_shuffle_speaker_split")], \
        comp.loc[("tp_zipa_rebuild_phone_gru_speaker_split", "tp_zipa_rebuild_phone_shuffle_speaker_split")]
    s = d._slide()
    d.header(s, "Results  ·  RQ1 sequence model", "Phone order helps the BiGRU only for speakers seen in training")
    d.stats(s, [(f"{o_rel['diff']:+.1f}", f"BiGRU − shuffled BiGRU, corrected\nrelease, shared speakers [{o_rel.lo:.1f}, {o_rel.hi:.1f}]"),
                (f"{o_sp['diff']:+.1f}", f"the same, new speakers\n[{o_sp.lo:.1f}, {o_sp.hi:.1f}]"),
                (f"{r_w.recall1_diff:+.1f} / {r_z.recall1_diff:+.1f}", "the same on our rebuild, new speakers\n(W2V2 / ZIPA)")],
            best=0)
    d.bullets(s, [
        "Shuffling the phones within each utterance keeps the inventory and removes order; both probes are scored on "
        "the same test utterances.",
        f"Order is worth {o_rel['diff']:.0f} points when test speakers are in training and nothing when they are not: "
        "the BiGRU memorises speakers' sequences. Bigram over unigram bags show the same split on the release.",
        "PRiSM's explanation of its result (order) holds only under its leaky split.",
    ], y=Inches(3.7), h=Inches(3.2))

    # 17. RQ2 recognizer doubling
    s = d._slide()
    d.header(s, "Results  ·  RQ2 duration", "Doubled phones in recognizer output depend on the recognizer")
    d.image(s, RES / "figures" / "fig_dpr_w2v2.png", LM, Inches(1.5), h=Inches(3.6))
    d.image(s, RES / "figures" / "fig_dpr_zipa.png", LM + Inches(6.3), Inches(1.5), h=Inches(3.6))
    hv_w = pd.read_csv(RES / "dpr" / "w2v2_haryana_vs_rest.csv").iloc[0]
    hv_z = pd.read_csv(RES / "dpr" / "zipa_haryana_vs_rest.csv").iloc[0]
    d.bullets(s, [
        f"W2V2 doubles {100 * hv_w.haryana:.2f}% of consonants in Haryana vs {100 * hv_w.rest:.2f}% elsewhere "
        f"(p = {hv_w.p:.3f}); ZIPA {100 * hv_z.haryana:.2f}% vs {100 * hv_z.rest:.2f}% (p = {hv_z.p:.2f}). Neither shows "
        "Haryanvi doubling, and Vaani-Hi has only 12 Haryana speakers.",
        "Recognizer output cannot settle gemination, so we measure it in the audio.",
    ], y=Inches(5.3), h=Inches(1.7), size=15)

    # 18. RQ2 acoustics
    mm = (RES / "gemination" / "mixed_model.txt").read_text()

    def eff(label):
        m_ = re.search(re.escape(label) + r".*?= x([\d.]+)[^(]*\(95% CI x([\d.]+) to x([\d.]+)\).*?tokens (\d+)", mm)
        return float(m_[1]), float(m_[2]), float(m_[3]), int(m_[4])
    voi, clo, val = eff("voiceless interval (primary)"), eff("closure, tokens with a burst"), \
        eff("validity: lexical geminate voiceless interval")
    s = d._slide()
    d.header(s, "Results  ·  RQ2 duration, acoustics", "In the audio, Haryana speakers hold voiceless stops 14% longer")
    d.stats(s, [(f"×{voi[0]:.2f}", f"voiceless interval, Haryana vs\nNew Delhi (95% CI {voi[1]:.2f}-{voi[2]:.2f})"),
                (f"×{clo[0]:.2f}", f"closure alone\n(95% CI {clo[1]:.2f}-{clo[2]:.2f})"),
                (f"×{val[0]:.2f}", f"lexical geminates vs single stops\n(validity, 95% CI {val[1]:.2f}-{val[2]:.2f})"),
                ("1-2%", "of these stops written doubled\nby either recognizer")], best=0)
    d.bullets(s, [
        f"{voi[3]:,} single voiceless stops between vowels in Rohtak, Jhajjar, Charkhi Dadri and New Delhi Vaani audio, "
        "found by forced alignment of the standard spelling and measured with Praat voicing and burst landmarks.",
        "Holds in each Haryana district (×1.13-1.15), in 45 of 57 words, and on speaker means without a model (×1.14); "
        "mixed model with word effects, speaking-rate control and a speaker intercept.",
        "Haryana single stops move about a quarter of the way to geminate length: lengthening, not doubling. A blind "
        "manual audit of 50 tokens by both authors is the next check.",
    ], y=Inches(3.7), h=Inches(3.2), size=15)

    # 19. plan
    s = d._slide()
    d.header(s, "Plan", "What remains for the final evaluation")
    d.deflist(s, [
        ("Benchmark", "A cleaned evaluation from the rebuild: 5-fold cross-validation over speaker groups (look-alike "
                      "codes merged), every speaker tested once, a shared-picture track, corrected coordinates. Built "
                      "and tested; all probes rerun on it in November."),
        ("Ada (Nidhi)", "Representation probes (PRiSM's RP-Attn) on both splits, then the order sweep over layers "
                        "(RQ1), the consonant-length probe by layer (RQ2), prosody contours (RQ3)."),
        ("RQ2", "Blind manual audit of 50 tokens; whether the encoders' hidden states keep the lengthening."),
        ("RQ3", "Prosody BiGRU on F0 and intensity contours; late fusion with the transcript probe."),
        ("Write-up", "ACL-format report with every table generated from results/ by script."),
    ], label_w=Inches(2.1), size=15, row=Inches(1.0))

    # 20. takeaways
    s = d._slide(dark=True)
    d.header(s, "Conclusion", "Takeaways", dark=True)
    d.bullets(s, [
        f"GEO-v rewards recognising speakers: on PRiSM's release a voice-only embedding ({ec[0]:.1f}) beats PRiSM's "
        f"reproduced probe ({gru[0]:.1f}), and every probe falls to the majority once speakers are separated.",
        f"The picture a speaker describes gives the location away ({pr.recall1:.1f} Recall@1 from the picture ID alone); "
        "without topic, transcript probes fall to the majority on new speakers.",
        f"Phone order helps PRiSM's probe only for seen speakers ({o_rel['diff']:+.1f} vs {o_sp['diff']:+.1f}).",
        "Haryana speakers hold voiceless stops 14% longer than New Delhi speakers, which the recognizers rarely write.",
    ], y=Inches(1.8), h=Inches(4.6), size=17, color=LIGHT)
    d.text(s, "Code, data recipe and every number: github.com/unignoramus11/sal-project", LM, Inches(6.6),
           Inches(12), Inches(0.3), size=11, italic=True, color="9fb8aa")

    # 21. appendix
    s = d._slide()
    d.header(s, "Appendix  ·  reference", "Metrics and limitations")
    d.deflist(s, [
        ("Metrics", "Recall@1: is the nearest distinct test location the true one (PRiSM's definition); mean "
                    "great-circle km; lift over the majority and km gain over the best constant on the same utterances."),
        ("Statistics", "5 seeds (BiGRUs 2-3 so far); 95% CIs and paired tests by bootstrap over speakers, because "
                       "utterances of one speaker are not independent."),
        ("Limitations", "12 Haryana speakers in Vaani-Hi; New Delhi stands for standard Hindi only approximately; "
                        "speakers on the release are estimated; recognizer output is not a phonetic gold standard."),
    ], label_w=Inches(2.0), size=15, row=Inches(1.25))
    d.text(s, "Bharadwaj et al. (2026). PRiSM: Benchmarking Phone Realization in Speech Models. ACL 2026, pages "
              "18093-18112.", LM, Inches(6.7), Inches(12), Inches(0.3), size=10, italic=True, color=CAP)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    out = HERE / "mid_evaluation.pptx"
    build().save(out)
    print("wrote", out)
    if args.pdf:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(out)], check=True,
                       capture_output=True)
        print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
