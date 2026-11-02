"""Mid-evaluation deck (3 Nov 2026), built from results/ on the course's Mid_Project_Template: white slides, the
template's section titles in its order (Project Title, Problem definition, Overall Methodology, Experiments
description, Work Done, Results, Analysis, Timelines for Completion, Summary), 7-minute talk + backup slides.
Each slide carries speaker notes: who presents it, the time budget and the points to make.

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
import pyarrow.parquet as pq
from matplotlib import font_manager
from matplotlib.lines import Line2D
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
DATA = ROOT / "data"
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)

# template look: white slides, black sans titles (Calibri; Carlito is metric-compatible), black body text
TITLE, BODY, MUTED, ACCENT, ACCENT2, RULE = "000000", "1a1a1a", "5f6368", "12352a", "5c8f72", "b7c9bd"
SANS = "Carlito"
W, H = Inches(13.333), Inches(7.5)
LM, TW = Inches(0.75), Inches(11.85)

cfg = pd.read_csv(RES / "tables" / "configs.csv").set_index("config")
base = pd.read_csv(RES / "baselines.csv")
comp = pd.read_csv(RES / "tables" / "comparisons.csv").set_index(["a", "b"])
knn = pd.read_csv(RES / "tables" / "knn_transcripts.csv").set_index(["transcripts", "protocol", "k"])
OFF, CHK = RES / "official", RES / "rebuild_checks"


def rgb(h):
    return RGBColor.from_string(h)


def bl(name, proto, prompts="all", split="test"):
    r = base[(base.baseline == name) & (base.protocol == proto) & (base.prompts == prompts) & (base.split == split)]
    return r.iloc[0]


def seedmean(c):
    v = [json.loads((d / "metrics.json").read_text())["metrics"]["test"]["recall1"]
         for d in sorted((RES / "runs").glob(f"{c}_s*")) if (d / "metrics.json").exists()]
    if not v:
        return float("nan"), 0.0, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def sg(x, nd=1):
    """Signed number with a true minus sign."""
    return f"{x:+.{nd}f}".replace("-", "−")


def fmt(m, sd=None):
    return f"{m:.1f}" if sd is None else f"{m:.1f} ± {sd:.1f}"


# ------------------------------------------------------------------ slide primitives

class Deck:
    def __init__(self):
        self.p = Presentation()
        self.p.slide_width, self.p.slide_height = W, H
        self.n = 0

    def slide(self, notes=""):
        s = self.p.slides.add_slide(self.p.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb("FFFFFF")
        self.n += 1
        if self.n > 1:
            self.text(s, str(self.n), W - Inches(0.9), H - Inches(0.45), Inches(0.5), Inches(0.3), size=10,
                      color=MUTED, align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=18, color=BODY, bold=False, italic=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        for i, line in enumerate(txt if isinstance(txt, list) else [txt]):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            r = para.add_run()
            r.text = line
            r.font.size, r.font.name, r.font.bold, r.font.italic = Pt(size), SANS, bold, italic
            r.font.color.rgb = rgb(color)
        return tb

    def title(self, s, title, claim=None):
        """Template title (top left, plain sans); our one-line finding underneath in the accent colour."""
        self.text(s, title, LM, Inches(0.42), TW, Inches(0.75), size=36, color=TITLE)
        if claim:
            self.text(s, claim, LM, Inches(1.1), TW, Inches(0.45), size=18, color=ACCENT, bold=True)

    @staticmethod
    def items(s, entries, x=LM, y=Inches(1.75), w=TW, h=Inches(5.2), size=18, gap=8):
        """entries: (level, label, text). level 0 = '•' bullet, level 1 = numbered sub-item ('1.' in label),
        level 2 = '–' sub-bullet. A label is set in bold before the text, as in the template."""
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        for i, (lvl, label, txt) in enumerate(entries):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.space_after = Pt(gap)
            para.line_spacing = 1.08
            pPr = para._p.get_or_add_pPr()
            indent = {0: 0.0, 1: 0.45, 2: 0.95}[lvl]
            hang = 0.28 if lvl != 1 else 0.4
            pPr.set("marL", str(Emu(Inches(indent + hang))))
            pPr.set("indent", str(-Emu(Inches(hang))))
            if lvl == 1:
                pPr.append(pPr.makeelement(qn("a:buNone"), {}))
            else:
                pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
                pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•" if lvl == 0 else "–"}))
            sz = size if lvl == 0 else size - 1.5
            if lvl == 1:
                r = para.add_run()
                r.text = label + "\t"
                r.font.size, r.font.name = Pt(sz), SANS
                r.font.color.rgb = rgb(BODY)
                label = None
            if label:
                r = para.add_run()
                r.text = label + (" " if txt else "")
                r.font.size, r.font.name, r.font.bold = Pt(sz), SANS, True
                r.font.color.rgb = rgb(BODY)
            if txt:
                r = para.add_run()
                r.text = txt
                r.font.size, r.font.name = Pt(sz), SANS
                r.font.color.rgb = rgb(BODY)
        return tb

    def image(self, s, path, x, y, w=None, h=None):
        return s.shapes.add_picture(str(path), x, y, width=w, height=h)

    def caption(self, s, txt, x, y, w):
        self.text(s, txt, x, y, w, Inches(0.3), size=11, color=MUTED, italic=True)

    def stat(self, s, num, label, x, y, w, color=ACCENT, size=34):
        self.text(s, num, x, y, w, Inches(0.65), size=size, color=color, bold=True)
        self.text(s, label, x, y + Inches(0.62), w, Inches(0.65), size=13, color=MUTED)

    def save(self, path):
        self.p.save(path)


def _arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(1, x1, y1, x2, y2)
    c.line.color.rgb = rgb(ACCENT2)
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def _box(s, txt, x, y, w, h, dashed=False, size=14):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    b.adjustments[0] = 0.12
    b.fill.solid()
    b.fill.fore_color.rgb = rgb("FFFFFF" if dashed else "f1f6f2")
    b.line.color.rgb = rgb("9aa0a6" if dashed else ACCENT2)
    b.line.width = Pt(1.25)
    if dashed:
        b.line.dash_style = 4
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(txt.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size, r.font.name, r.font.bold = Pt(size if i == 0 else size - 2), SANS, i == 0
        r.font.color.rgb = rgb(MUTED if dashed else (ACCENT if i == 0 else BODY))
    return b


# ------------------------------------------------------------------ tables and charts

for _f in Path.home().glob("Library/Fonts/Carlito-*.ttf"):
    font_manager.fontManager.addfont(str(_f))
# two shades of one hue (ordinal pair, validated on white): test speakers in training / new speakers
SHARED, NEW, LINK = "#12352a", "#5c8f72", "#b7cbbe"


def table_png(name, header, rows, col_align=None, bold_cells=(), group_rows=(), widths=None, fontsize=12):
    """Booktabs-style table (sans, as the slides), transparent background."""
    ncol, nrows = len(header), len(rows) + 1
    widths = widths or [1.0] * ncol
    tot = sum(widths)
    plt.rcParams.update({"font.family": "Carlito"})
    fig, ax = plt.subplots(figsize=(0.95 * tot, 0.3 * (nrows + 0.6)))
    ax.set_axis_off()
    ax.set_xlim(0, tot)
    ax.set_ylim(nrows + 0.3, -0.3)
    xs = [sum(widths[:i]) for i in range(ncol)]
    col_align = col_align or ["left"] + ["right"] * (ncol - 1)
    for i, r in enumerate([header] + rows):
        for j, v in enumerate(r):
            a = col_align[j]
            x = xs[j] + (0.05 if a == "left" else widths[j] - 0.05 if a == "right" else widths[j] / 2)
            ax.text(x, i, v, ha=a, va="center", fontsize=fontsize, color="#1a1a1a",
                    fontweight="bold" if i == 0 or (i - 1, j) in bold_cells else "normal",
                    style="italic" if (i - 1) in group_rows and j == 0 and i > 0 else "normal")
    for yy, lw in ((-0.45, 1.2), (0.5, 0.6), (nrows - 0.5, 1.2)):
        ax.plot([0, tot], [yy, yy], color="black", lw=lw)
    for g in group_rows:
        ax.plot([0, tot], [g + 0.5, g + 0.5], color="#999999", lw=0.4)
    out = BUILD / f"{name}.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def _axes(w, h):
    plt.rcParams.update({"font.family": "Carlito", "font.size": 14})
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9aa0a6")
    ax.tick_params(colors="#1a1a1a", length=0, labelsize=14)
    ax.xaxis.grid(True, color="#e3e7e4", lw=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def _legend(ax, labels):
    h = [Line2D([], [], marker="o", ls="", ms=10, color=c, label=lab) for c, lab in zip((SHARED, NEW), labels)]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=13.5,
              handletextpad=0.3, columnspacing=1.6, borderaxespad=0.3)


def _save(fig, name):
    out = BUILD / f"{name}.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def dumbbell(name, rows, xlabel, labels, w=11.6, h=4.5, xmax=105):
    """rows: (label, value with test speakers in training, value on new speakers), top to bottom; both labelled."""
    fig, ax = _axes(w, h)
    y = np.arange(len(rows))[::-1]
    for yi, (_, a, b) in zip(y, rows):
        ax.plot([min(a, b), max(a, b)], [yi, yi], color=LINK, lw=3, zorder=1, solid_capstyle="round")
        for v, c in ((a, SHARED), (b, NEW)):
            ax.scatter([v], [yi], s=130, color=c, zorder=3, edgecolor="white", linewidth=2)
        off = 0.02 * xmax + max(0.0, 0.03 * xmax - abs(a - b)) / 2      # keep labels clear of overlapping dots
        ax.text(max(a, b) + off, yi, f"{max(a, b):.1f}", va="center", ha="left", fontsize=13.5, color="#000000")
        ax.text(min(a, b) - off, yi, f"{min(a, b):.1f}", va="center", ha="right", fontsize=13.5, color="#3c4043")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-6, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=14, color="#5f6368")
    _legend(ax, labels)
    return _save(fig, name)


def forest(name, rows, xlabel, labels, w=8.2, h=5.2):
    """rows: (label, (est, lo, hi) with test speakers in training, (est, lo, hi) on new speakers)."""
    fig, ax = _axes(w, h)
    y = np.arange(len(rows))[::-1] * 1.0
    for yi, (_, a, b) in zip(y, rows):
        for (e, lo, hi), c, dy in ((a, SHARED, 0.17), (b, NEW, -0.17)):
            ax.plot([lo, hi], [yi + dy] * 2, color=c, lw=2.4, solid_capstyle="round", zorder=2)
            ax.scatter([e], [yi + dy], s=85, color=c, zorder=3, edgecolor="white", linewidth=1.5)
            if hi < 0.5:                                         # interval left of zero: label on its left
                ax.text(lo - 0.6, yi + dy, sg(e), va="center", ha="right", fontsize=13, color="#3c4043")
            else:
                ax.text(hi + 0.6, yi + dy, sg(e), va="center", ha="left", fontsize=13, color="#3c4043")
    ax.axvline(0, color="#5f6368", lw=1)
    los = [v[1] for r in rows for v in r[1:]]
    his = [v[2] for r in rows for v in r[1:]]
    ax.set_xlim(min(los) - 4, max(his) + 3.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=14, color="#5f6368")
    _legend(ax, labels)
    return _save(fig, name)


# ------------------------------------------------------------------ content

def build():
    d = Deck()
    I = Inches
    splits = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")
    meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                           columns=["metadata_idx", "state", "district", "pincode", "speaker", "audio_seconds"])
    prompts = pd.read_parquet(DATA / "vaani_hi_rebuild" / "prompts.parquet")
    leak = 100 * splits.loc[splits.prism == "test", "speaker"].isin(set(splits.loc[splits.prism == "train", "speaker"])).mean()
    specific = 100 * (prompts.prompt_type == "SPECIFIC").mean()
    pic = pd.read_csv(CHK / "picture_lookup.csv").set_index(["prompts", "protocol", "baseline"])
    look = pd.read_csv(CHK / "lookalike.csv").set_index("probe")
    coords = pd.read_csv(CHK / "coordinates.csv")
    ocl = pd.read_csv(OFF / "scores_clean.csv")
    odup = pd.read_csv(OFF / "scores_by_duplication.csv").set_index("probe")
    oorder = pd.read_csv(OFF / "order_clean.csv").set_index(["protocol", "a"])
    oleak = pd.read_csv(OFF / "leakage.csv").set_index("split")
    shimla = pd.read_csv(OFF / "coordinates_audit.csv").iloc[0]
    dups = float(re.search(r"copies with different pincodes ([\d.]+)%", (OFF / "audit.txt").read_text())[1])
    pc = pd.read_csv(RES / "tables" / "prosody_checks.csv").set_index(["subset", "comparison"])
    pm = json.loads((RES / "runs" / "prosody_gru_speaker_split_s42" / "metrics.json").read_text())["metrics"]["test"]
    mm = (RES / "gemination" / "mixed_model.txt").read_text()
    NN = "nearest-neighbour transcript (1-NN)"
    pr, ps = pic.loc[("all", "prism", "picture lookup")], pic.loc[("all", "speaker_split", "picture lookup")]
    gru, ec = seedmean("tp_w2v2_official_char_gru_prism"), seedmean("dense_ecapa_official_prism")
    o_rel, o_sp = (oorder.loc[(P, "tp_w2v2_official_phone_gru")] for P in ("prism", "speaker_split"))
    fk, fr = pc.loc[("all", "fusion - transcript, km")], pc.loc[("all", "fusion - transcript, Recall@1")]
    fks = pc.loc[("strict", "fusion - transcript, km")]

    def eff(label):
        m_ = re.search(re.escape(label) + r".*?= x([\d.]+)[^(]*\(95% CI x([\d.]+) to x([\d.]+)\).*?tokens (\d+)", mm)
        return float(m_[1]), float(m_[2]), float(m_[3]), int(m_[4])
    voi, clo = eff("voiceless interval (primary)"), eff("closure, tokens with a burst")
    n_runs = sum(1 for p in (RES / "runs").glob("*/metrics.json"))
    n_tx = {}
    for m in ("w2v2", "zipa"):
        for part in ("rebuild", "official", "districts"):
            f = ROOT / "outputs" / "transcripts" / f"{m}_{part}.parquet"
            if f.exists():
                n_tx[(m, part)] = pq.ParquetFile(f).metadata.num_rows
    n_utt_tx = sum(v for (m, _), v in n_tx.items() if m == "w2v2")

    # 1. Project Title ----------------------------------------------------------------------------------------
    s = d.slide("Mohit (10 s). Title, team, base paper. Hand over to the problem.")
    d.text(s, "Order, Duration, or Melody?", I(0.75), I(2.15), I(11.85), I(1.0), size=44, color=TITLE,
           align=PP_ALIGN.CENTER)
    d.text(s, "Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech", I(0.75), I(3.15),
           I(11.85), I(0.5), size=24, color=ACCENT, align=PP_ALIGN.CENTER)
    d.text(s, "Mohit Kumar Singh  ·  Nidhi Vaidya", I(0.75), I(4.2), I(11.85), I(0.4), size=20, color=BODY,
           align=PP_ALIGN.CENTER)
    d.text(s, ["Speech Analysis and Linguistics, Monsoon 2026  ·  Mid project evaluation, 3 November 2026",
               "Base paper: Bharadwaj et al., PRiSM: Benchmarking Phone Realization in Speech Models, ACL 2026"],
           I(0.75), I(4.85), I(11.85), I(0.8), size=15, color=MUTED, align=PP_ALIGN.CENTER)

    # 2. Problem definition -------------------------------------------------------------------------------------
    s = d.slide("Mohit (45 s).\n"
                "- PRiSM's geolocation task: place an utterance on the map; two probes on the same recognizer.\n"
                "- The puzzle: the probe that cannot hear voice, duration or intonation wins, and PRiSM's explanation "
                "(phone order) was never tested.\n"
                "- Four objectives, one per property of speech; RQ0 asks first whether the benchmark measures dialect.")
    d.title(s, "Problem definition")
    d.items(s, [
        (0, "Description:", "PRiSM (ACL 2026) benchmarks phone recognizers. In its geolocation task GEO-v, a probe "
                            "predicts where in the Hindi belt an utterance was recorded (Vaani-Hi, about 30 pincodes "
                            "over ~1,500 km). A transcript probe over the predicted phone string beats a "
                            "representation probe over the recognizer's hidden states: 146 km vs 253 km mean error."),
        (0, "Motivation:", "Only the representation probe hears voice, duration and intonation, yet it loses. PRiSM "
                           "credits phone order without testing it. If the benchmark rewards recognising speakers or "
                           "topics, the comparison says little about dialect."),
        (0, "Objectives:", ""),
        (1, "1.", "Speakers and topic (RQ0): does PRiSM's split let probes recognise speakers and picture topics "
                  "instead of dialect?"),
        (1, "2.", "Order (RQ1): does local phone order carry location beyond the phone inventory?"),
        (1, "3.", "Duration (RQ2): is Haryanvi consonant lengthening in the recognizers' output, and in the audio?"),
        (1, "4.", "Melody (RQ3): does intonation, which transcripts discard, add location signal?"),
    ], y=I(1.35), size=18, gap=7)

    # 3. Overall Methodology ------------------------------------------------------------------------------------
    s = d.slide("Mohit (35 s).\n"
                "- Four views of each utterance: phone string and hidden states from the recognizer, a voice "
                "embedding, and prosody contours; each feeds a probe that predicts latitude and longitude.\n"
                "- The representation-probe path is dashed: not run yet (storage on Ada).\n"
                "- References that learn nothing, and every probe under two splits: this is what exposes leakage.")
    d.title(s, "Overall Methodology")
    ys, rh = [1.45, 2.35, 3.25, 4.15], 0.7
    mid = [I(y + rh / 2) for y in ys]
    _box(s, "Utterance\naudio", I(0.75), I(ys[0]), I(1.3), I(ys[3] + rh - ys[0]))
    for txt, r0, r1 in [("Phone recognizer\nW2V2 · ZIPA", 0, 2), ("Voice encoder\nECAPA-TDNN", 2, 3),
                        ("Praat\nF0 · voicing · intensity", 3, 4)]:
        top, bot = ys[r0], ys[r1 - 1] + rh
        _box(s, txt, I(2.4), I(top), I(2.3), I(bot - top))
        _arrow(s, I(2.05), I((top + bot) / 2), I(2.4), I((top + bot) / 2))
    reps = ["Phone string", "Hidden states", "Voice embedding", "Contours, 100 per s"]
    probes = [("Transcript probes", "bag MLPs · BiGRU · shuffled BiGRU", False),
              ("Representation probe", "PRiSM's RP-Attn · not run yet", True),
              ("Voice probe", "MLP", False), ("Prosody probe", "BiGRU · late fusion", False)]
    for k, (rep, (pt, pst, dash)) in enumerate(zip(reps, probes)):
        _box(s, rep, I(5.05), I(ys[k]), I(2.1), I(rh), dashed=dash)
        _box(s, f"{pt}\n{pst}", I(7.5), I(ys[k]), I(3.15), I(rh), dashed=dash)
        _arrow(s, I(4.7), mid[k], I(5.05), mid[k])
        _arrow(s, I(7.15), mid[k], I(7.5), mid[k])
        _arrow(s, I(10.65), mid[k], I(11.0), mid[k])
    _box(s, "Location\nlatitude, longitude", I(11.0), I(ys[0]), I(1.6), I(ys[3] + rh - ys[0]))
    d.items(s, [
        (0, "References that learn nothing about dialect:", "majority location, speaker-ID lookup, picture-ID "
                                                             "lookup, nearest-neighbour transcript match."),
        (0, "Two splits for every probe:", "PRiSM's random split (test speakers also in training) and a "
                                           "speaker-disjoint split (new speakers)."),
        (0, "Metrics:", "Recall@1 (PRiSM's) and mean km; 5 seeds; 95% CIs by bootstrap over speakers."),
    ], y=I(5.15), h=I(1.9), size=16, gap=4)

    # 4. Experiments description: database ---------------------------------------------------------------------
    st = meta.groupby("state").agg(pin=("pincode", "nunique"), spk=("speaker", "nunique"), utt=("metadata_idx", "size"),
                                   hrs=("audio_seconds", lambda x: x.sum() / 3600)).sort_values("utt", ascending=False)
    names = {"MadhyaPradesh": "Madhya Pradesh", "HimachalPradesh": "Himachal Pradesh"}
    srows = [[names.get(r.state, r.state), str(int(r.pin)), f"{int(r.spk):,}", f"{int(r.utt):,}", f"{r.hrs:.1f}"]
             for r in st.reset_index().itertuples()]
    srows.append(["Total", str(meta.pincode.nunique()), f"{meta.speaker.nunique():,}", f"{len(meta):,}",
                  f"{meta.audio_seconds.sum() / 3600:.1f}"])
    t_states = table_png("db_states", ["State", "Pincodes", "Speakers", "Utterances", "Hours"], srows,
                         bold_cells=[(len(srows) - 1, j) for j in range(5)], widths=[1.9, 1.0, 1.0, 1.15, 0.8], fontsize=12.5)
    w2 = pd.read_parquet(ROOT / "outputs" / "transcripts" / "w2v2_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
    zp = pd.read_parquet(ROOT / "outputs" / "transcripts" / "zipa_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
    smp = meta.merge(prompts, on="metadata_idx").merge(w2, on="metadata_idx").merge(zp, on="metadata_idx", suffixes=("_w", "_z"))
    rows_s = []
    for idx in (9, 4391, 17440):                         # three real utterances (picked for clean recognizer output)
        r = smp.set_index("metadata_idx").loc[idx]
        rows_s += [[f"{r.district}, {names.get(r.state, r.state)}", "own district" if r.prompt_type == "SPECIFIC" else "shared",
                    "W2V2", r.processed_transcript_w[:30] + "…"],
                   ["", "", "ZIPA", r.processed_transcript_z[:30] + "…"]]
    t_samples = table_png("db_samples", ["District, state", "Picture", "Model", "Recognized phones (start)"],
                          rows_s, col_align=["left", "left", "left", "left"], widths=[2.1, 1.25, 0.75, 3.3], fontsize=12.5)
    s = d.slide("Nidhi (35 s).\n"
                f"- Vaani: crowd-sourced Indian speech, people describe pictures. We rebuilt PRiSM's Vaani-Hi subset "
                f"with PRiSM's own code: {len(meta):,} utterances, {meta.speaker.nunique():,} speakers; the speaker code in "
                "each file name is what makes a speaker-disjoint split possible.\n"
                "- Haryana has only 12 speakers here, so RQ2 uses separate Vaani audio from three Haryana districts and "
                "New Delhi.\n"
                "- Samples: two thirds of pictures belong to the speaker's own district (topic can give the place away); "
                "the two recognizers write the same speech differently.")
    d.title(s, "Experiments description")
    d.items(s, [
        (0, "Database:", ""),
        (1, "1.", f"Describe the DB: Vaani-Hi rebuilt from the public Vaani corpus with PRiSM's selection code: "
                  f"{len(meta):,} picture descriptions, {meta.audio_seconds.sum() / 3600:.1f} h, {meta.speaker.nunique():,} "
                  f"speakers, {specific:.0f}% about a picture of the speaker's own district. Also PRiSM's released copy "
                  f"(26,385 rows, no speaker IDs) and, for RQ2, Vaani audio from 3 Haryana districts and New Delhi "
                  f"({n_tx.get(('w2v2', 'districts'), 0):,} utterances)."),
        (1, "2.", "Show the exemplary samples from the DB:"),
    ], y=I(1.25), h=I(2.0), size=17, gap=4)
    d.image(s, t_states, I(0.85), I(3.25), w=I(5.0))
    d.image(s, t_samples, I(6.2), I(3.25), w=I(6.45))
    d.caption(s, "Rebuilt Vaani-Hi by state.", I(0.85), I(6.85), I(5.0))
    d.caption(s, "Three test utterances and the start of each recognizer's phone string.", I(6.2), I(6.85), I(6.4))

    # 5. Experiments description: experiments ------------------------------------------------------------------
    s = d.slide("Nidhi (30 s). One line per experiment; metrics in brackets. Every comparison is paired on the same "
                "test utterances, and CIs resample speakers because one speaker's utterances are not independent.")
    d.title(s, "Experiments description")
    d.items(s, [
        (0, "List of the experiments:", "(evaluation metrics in brackets)"),
        (1, "1.", "Speakers and topic (RQ0): every probe under PRiSM's split and a speaker-disjoint split; lookups by "
                  "speaker and picture ID; transcript matcher; voice probe; control task with permuted labels "
                  "[Recall@1, mean km, gain over the majority location]."),
        (1, "2.", "PRiSM's release: reproduce PRiSM's transcript probe; audit duplicates, leakage and coordinates; "
                  "rerun all probes on a corrected, speaker-disjoint version [Recall@1, mean km]."),
        (1, "3.", "Order (RQ1): unigram vs bigram bags, BiGRU vs the same BiGRU on shuffled phones, characters vs "
                  "phones, two recognizers [paired Recall@1 difference, km, MDL codelength]."),
        (1, "4.", "Duration (RQ2): doubled-phone rate in recognizer output; voiceless-stop and closure duration "
                  "measured in the audio, Haryana vs New Delhi [% doubled, duration ratio from mixed models]."),
        (1, "5.", "Melody (RQ3): pooled prosody statistics; BiGRU over F0, voicing and intensity contours; late "
                  "fusion with the transcript probe [Recall@1, mean km]."),
    ], y=I(1.35), size=18, gap=9)

    # 6. Work Done --------------------------------------------------------------------------------------------------
    s = d.slide("Nidhi (25 s). Left: what is complete, with the numbers behind it. Right: what is not, and why "
                "(the representation probes need frame caches that did not fit Ada's 50 GB quota).")
    d.title(s, "Work Done")
    d.items(s, [
        (0, "Data:", "rebuilt Vaani-Hi with PRiSM's code (identical on a second machine); speaker-disjoint split."),
        (0, "Transcripts:", f"phone strings from 2 recognizers for {n_utt_tx:,} utterances (rebuild, PRiSM's release, "
                            "district audio)."),
        (0, "Reproduction:", f"PRiSM's transcript probe on its release, {gru[0]:.1f} vs 45.7 Recall@1; audit of the "
                             "released data and a corrected version."),
        (0, "Probes:", f"{n_runs} trained probe runs: bags, BiGRUs, shuffled BiGRUs, voice and prosody probes, "
                       "control task, MDL."),
        (0, "Acoustics:", f"{voi[3]:,} voiceless stops located by forced alignment, measured in Praat, mixed models."),
        (0, "Prosody:", "contours for every utterance; prosody BiGRU and late fusion (on Ada)."),
        (0, "Final benchmark:", "5-fold, speaker-disjoint, topic-controlled; built and tested."),
    ], x=LM, y=I(1.3), w=I(7.6), size=16, gap=7)
    d.text(s, "Not yet", I(8.75), I(1.3), I(3.8), I(0.4), size=18, color=ACCENT, bold=True)
    d.items(s, [
        (0, "", "Representation probes (PRiSM's RP-Attn): frame caches did not fit the Ada quota."),
        (0, "", "Blind manual audit of 50 measured stops, by both of us."),
        (0, "", "Remaining seeds: BiGRU 3-5, prosody BiGRU 2-5."),
    ], x=I(8.75), y=I(1.8), w=I(3.85), size=16, gap=10)

    # 7. Results: speakers and topic ------------------------------------------------------------------------------
    c1 = dumbbell("chart_lookups", [
        ("Speaker-ID lookup", bl("speaker_lookup", "prism").recall1, bl("speaker_lookup", "speaker_split").recall1),
        ("Picture-ID lookup", pr.recall1, ps.recall1),
        ("Voice embedding (ECAPA)", cfg.loc["dense_ecapa_rebuild_prism", "recall1_mean"],
         cfg.loc["dense_ecapa_rebuild_speaker_split", "recall1_mean"]),
        ("Transcript match, nothing learned", knn.loc[("w2v2", "prism", 1), "recall1"],
         knn.loc[("w2v2", "speaker_split", 1), "recall1"]),
        ("ZIPA phone bigrams", cfg.loc["tp_zipa_rebuild_phone_bigram_prism", "recall1_mean"],
         cfg.loc["tp_zipa_rebuild_phone_bigram_speaker_split", "recall1_mean"]),
        ("PRiSM's BiGRU (characters)", cfg.loc["tp_w2v2_rebuild_char_gru_prism", "recall1_mean"],
         cfg.loc["tp_w2v2_rebuild_char_gru_speaker_split", "recall1_mean"]),
        ("Majority location", bl("majority", "prism").recall1, bl("majority", "speaker_split").recall1),
    ], "Test Recall@1 on rebuilt Vaani-Hi (%)", ("test speakers also in training (PRiSM's split)", "new speakers"),
        h=4.4)
    s = d.slide("Mohit (35 s). RQ0.\n"
                f"- Under PRiSM's split {leak:.0f}% of test speakers are also in training; knowing only the speaker ID "
                f"gives {bl('speaker_lookup', 'prism').recall1:.0f} Recall@1.\n"
                f"- Knowing only which picture was described gives {pr.recall1:.0f}; on new speakers still {ps.recall1:.0f}, "
                "above every probe.\n"
                "- Every trained probe drops on new speakers; ECAPA falls to the majority.")
    d.title(s, "Results", "Lookups that know nothing about dialect beat every probe")
    d.image(s, c1, LM, I(1.7), w=I(11.8))
    d.caption(s, f"Figure 1. Speaker-ID lookup: {leak:.1f}% of test utterances have their speaker in training under "
                 "PRiSM's split. Lookups predict the commonest training location of the same speaker or picture.",
              LM, I(6.95), I(11.8))

    # 8. Results: PRiSM's release --------------------------------------------------------------------------------
    def oc(proto, probe):
        return ocl[(ocl.protocol == proto) & (ocl.probe == probe)].recall1.iloc[0]

    def osm(key, P):
        return seedmean(f"{key}_{P}_clean")[0]
    c2 = dumbbell("chart_release", [
        ("Transcript match, nothing learned", oc("prism", NN), oc("speaker_split", NN)),
        ("ZIPA phone bigrams", osm("tp_zipa_official_phone_bigram", "prism"), osm("tp_zipa_official_phone_bigram", "speaker_split")),
        ("Voice embedding (ECAPA)", osm("dense_ecapa_official", "prism"), osm("dense_ecapa_official", "speaker_split")),
        ("PRiSM's BiGRU (characters)", osm("tp_w2v2_official_char_gru", "prism"), osm("tp_w2v2_official_char_gru", "speaker_split")),
        ("Majority location", oc("prism", "majority"), oc("speaker_split", "majority")),
    ], "Test Recall@1, corrected release (%)", ("released split", "speaker-disjoint split"), w=7.4, h=4.3, xmax=72)
    s = d.slide("Mohit (40 s). PRiSM's own data.\n"
                f"- Our pipeline reproduces PRiSM's number ({gru[0]:.1f} vs 45.7).\n"
                f"- The release repeats recordings: {100 * oleak.loc['test', 'exact_audio_in_train']:.0f}% of test clips are "
                "in training verbatim; Shimla is placed in Afghanistan.\n"
                f"- A voice embedding with no words scores {ec[0]:.1f}, above PRiSM's probe.\n"
                "- Corrected (one row per recording) and speaker-disjoint: everything falls to the majority.")
    d.title(s, "Results", "On PRiSM's own data the voice alone beats PRiSM's probe")
    stats = [(f"{gru[0]:.1f}", f"PRiSM's transcript probe, our code\n({gru[2]} seeds; paper 45.7)", BODY),
             (f"{ec[0]:.1f}", "voice embedding, no words", ACCENT),
             (f"{odup.loc[NN, 'all_recall1']:.1f}", "transcript match, nothing learned", BODY),
             (f"{100 * oleak.loc['test', 'exact_audio_in_train']:.0f}%", f"test clips also in training; {dups:.0f}% of\n"
                                                                         "repeats carry another pincode", BODY)]
    for k, (num, lab, col) in enumerate(stats):
        d.stat(s, num, lab, LM, I(1.75 + 1.25 * k), I(3.6), color=col, size=32)
    im = d.image(s, c2, I(4.75), I(1.75), w=I(7.85))
    d.caption(s, f"Figure 2. Recall@1 on the released split (left four). Right: the corrected release (one row per "
                 f"recording, Shimla moved {shimla.km_from_valid_median:.0f} km back from Afghanistan); speakers from "
                 "voice clusters.", LM, I(6.95), I(11.8))

    # 9. Results: order --------------------------------------------------------------------------------------------
    def cpair(a, b):
        r = comp.loc[(a, b)]
        return r.recall1_diff, r.recall1_lo, r.recall1_hi

    def opair(P, a):
        r = oorder.loc[(P, a)]
        return r["diff"], r.lo, r.hi
    c3 = forest("chart_order", [
        ("BiGRU vs shuffled\nrelease, W2V2", opair("prism", "tp_w2v2_official_phone_gru"),
         opair("speaker_split", "tp_w2v2_official_phone_gru")),
        ("BiGRU vs shuffled\nrebuild, W2V2", cpair("tp_w2v2_rebuild_phone_gru_prism", "tp_w2v2_rebuild_phone_shuffle_prism"),
         cpair("tp_w2v2_rebuild_phone_gru_speaker_split", "tp_w2v2_rebuild_phone_shuffle_speaker_split")),
        ("BiGRU vs shuffled\nrebuild, ZIPA", cpair("tp_zipa_rebuild_phone_gru_prism", "tp_zipa_rebuild_phone_shuffle_prism"),
         cpair("tp_zipa_rebuild_phone_gru_speaker_split", "tp_zipa_rebuild_phone_shuffle_speaker_split")),
        ("Bigram vs unigram\nrelease, W2V2", opair("prism", "tp_w2v2_official_phone_bigram"),
         opair("speaker_split", "tp_w2v2_official_phone_bigram")),
        ("Bigram vs unigram\nrebuild, W2V2", cpair("tp_w2v2_rebuild_phone_bigram_prism", "tp_w2v2_rebuild_phone_unigram_prism"),
         cpair("tp_w2v2_rebuild_phone_bigram_speaker_split", "tp_w2v2_rebuild_phone_unigram_speaker_split")),
        ("Bigram vs unigram\nrebuild, ZIPA", cpair("tp_zipa_rebuild_phone_bigram_prism", "tp_zipa_rebuild_phone_unigram_prism"),
         cpair("tp_zipa_rebuild_phone_bigram_speaker_split", "tp_zipa_rebuild_phone_unigram_speaker_split")),
    ], "Recall@1 gain from phone order, points (95% CI)", ("test speakers in training", "new speakers"), h=4.9)
    s = d.slide("Mohit (30 s). RQ1.\n"
                "- Each row: a probe that sees phone order minus the same probe without it, same test utterances.\n"
                f"- PRiSM's BiGRU gains {o_rel['diff']:.0f} points from order when speakers are shared and nothing on new "
                "speakers: it memorises speakers' sequences.\n"
                "- Bigram bags keep a gain on the rebuild's new speakers, but bigrams also identify words (topic).")
    d.title(s, "Results", "Phone order helps mostly when test speakers were seen in training")
    im = d.image(s, c3, LM, I(1.7), w=I(7.6))
    tx = LM + im.width + I(0.4)
    d.items(s, [
        (0, "", "Each row: a probe that sees phone order minus the same probe without it, paired on the same test "
                "utterances."),
        (0, "", f"PRiSM's BiGRU gains {o_rel['diff']:.0f} points from order on its own split and nothing on new "
                "speakers."),
        (0, "", "Bigram bags keep some gain on new speakers; bigrams also identify words, so part may be topic."),
    ], x=tx, y=I(1.85), w=W - tx - I(0.6), size=16, gap=10)
    d.caption(s, "Figure 3. Release: corrected rows, speaker-disjoint split from voice clusters (59 test speakers). Rebuild "
                 "BiGRUs: 2 seeds so far.", LM, I(6.95), I(11.8))

    # 10. Results: duration and melody ---------------------------------------------------------------------------
    hv_w = pd.read_csv(RES / "dpr" / "w2v2_haryana_vs_rest.csv").iloc[0]
    s = d.slide("Nidhi (35 s).\n"
                f"- RQ2: in the audio, Haryana speakers hold voiceless stops {100 * (voi[0] - 1):.0f}% longer than New Delhi "
                f"speakers (closure {100 * (clo[0] - 1):.0f}%), in each district and most words; the recognizers write "
                "only 1-2% of these stops doubled, so a transcript probe cannot use it.\n"
                "- RQ3: the prosody BiGRU alone does not beat the majority on new speakers; fused with the transcript "
                f"probe it lowers mean error by {-fk['diff']:.0f} km, also without look-alike speakers. One seed so far.")
    d.title(s, "Results", "Haryana stops are longer in the audio; intonation helps only in distance")
    d.text(s, "Duration (RQ2): Haryana vs New Delhi", LM, I(1.75), I(5.6), I(0.4), size=18, bold=True, color=BODY)
    d.stat(s, f"×{voi[0]:.2f}", f"voiceless interval [{voi[1]:.2f}, {voi[2]:.2f}]", LM, I(2.3), I(2.7))
    d.stat(s, f"×{clo[0]:.2f}", f"closure [{clo[1]:.2f}, {clo[2]:.2f}]", LM + I(2.85), I(2.3), I(2.7))
    d.items(s, [
        (0, "", f"{voi[3]:,} single voiceless stops between vowels; holds in each Haryana district and in 45 of 57 words."),
        (0, "", f"The recognizers write a double consonant for only 1-2% of them (W2V2: {100 * hv_w.haryana:.2f}% of "
                f"consonants in Haryana vs {100 * hv_w.rest:.2f}% elsewhere)."),
    ], x=LM, y=I(3.85), w=I(5.6), size=15, gap=8)
    x2 = I(7.0)
    d.text(s, "Melody (RQ3): new speakers", x2, I(1.75), I(5.6), I(0.4), size=18, bold=True, color=BODY)
    d.stat(s, f"{pm['recall1']:.1f}", f"prosody BiGRU, Recall@1\n(majority {bl('majority', 'speaker_split').recall1:.1f})",
           x2, I(2.3), I(2.7), color=BODY)
    d.stat(s, f"{sg(fk['diff'])} km", f"fused with the transcript probe\n[{sg(fk.lo)}, {sg(fk.hi)}]", x2 + I(2.85), I(2.3),
           I(2.9))
    d.items(s, [
        (0, "", f"Fusion leaves Recall@1 unchanged ({sg(fr['diff'])} [{sg(fr.lo)}, {sg(fr.hi)}]) and still saves "
                f"{-fks['diff']:.0f} km without look-alike speakers."),
        (0, "", "Pooled prosody statistics: below the majority on both splits. Contour BiGRU: 1 seed so far."),
    ], x=x2, y=I(3.85), w=I(5.6), size=15, gap=8)
    s.shapes.add_connector(1, I(6.65), I(1.8), I(6.65), I(6.4)).line.color.rgb = rgb(RULE)

    # 11. Analysis ---------------------------------------------------------------------------------------------------
    e = look.loc["dense_ecapa_rebuild_speaker_split"]
    dhar = coords[coords.data == "rebuild"].sort_values("km_from_cluster", ascending=False).iloc[0]
    s = d.slide("Nidhi (30 s). What the results mean, one line per objective; the last point is why the "
                "representation probe matters for the final.")
    d.title(s, "Analysis", "GEO-v as released measures memorised speakers and topics, not dialect")
    d.items(s, [
        (0, "Speakers:", f"{leak:.0f}% of PRiSM's test speakers are in training and "
                         f"{100 * oleak.loc['test', 'exact_audio_in_train']:.0f}% of its test clips are in training verbatim; "
                         "a voice-only probe and a matcher that learns nothing both beat PRiSM's probe."),
        (0, "Topic:", f"{specific:.0f}% of utterances describe a picture of the speaker's own district; the picture ID alone "
                      f"gives {pr.recall1:.1f} Recall@1, and on shared pictures the transcript probes fall to the majority."),
        (0, "Order:", f"the BiGRU's gain from order ({sg(o_rel['diff'])}) appears only for speakers seen in training, so "
                      "PRiSM's explanation holds only under its leaky split."),
        (0, "Our own split, checked:", f"look-alike speaker codes leak the voice (ECAPA {sg(e.lookalike_lift)} vs "
                                       f"{sg(e.strict_lift)}) but not the transcripts; PRiSM's coordinate recipe misplaces "
                                       f"Dhar by {dhar.km_from_cluster:.0f} km."),
        (0, "Duration:", "real in the audio but rarely written by the recognizers: only the representation probe could "
                         "use it."),
    ], y=I(1.75), size=17, gap=9)

    # 12. Timelines for Completion -----------------------------------------------------------------------------------
    trows = [["3-9 Nov", "Representation probes (RP-Attn, last layer, 3 recognizers); remaining seeds", "Nidhi, Mohit"],
             ["10-16 Nov", "Every probe on the cross-validated benchmark; blind manual audit", "Mohit; both"],
             ["17-23 Nov", "Order-aware representation probes across layers (RQ1); length in hidden states (RQ2)", "Nidhi, Mohit"],
             ["24-30 Nov", "Prosody seeds and fusion on the benchmark (RQ3); statistics and figures", "Nidhi, Mohit"],
             ["1-4 Dec", "Report (ACL format) and final presentation", "Both"]]
    t_time = table_png("timeline", ["When", "Task", "Owner"], trows, col_align=["left", "left", "left"],
                       widths=[1.3, 10.2, 1.6], fontsize=15)
    s = d.slide("Nidhi (20 s). Four weeks to the final on 4 December; the representation probes come first because "
                "PRiSM's central comparison needs them.")
    d.title(s, "Timelines for Completion")
    d.image(s, t_time, LM, I(1.6), w=I(11.8))

    # 13. Summary ------------------------------------------------------------------------------------------------------
    s = d.slide("Nidhi (20 s). Four findings and one line on what comes next. Then questions.")
    d.title(s, "Summary")
    d.items(s, [
        (0, "", f"PRiSM's GEO-v rewards recognising speakers: on its own data a voice-only probe ({ec[0]:.1f}) beats "
                f"PRiSM's reproduced transcript probe ({gru[0]:.1f}), and every probe falls to the majority once speakers "
                "are separated."),
        (0, "", f"The picture a speaker describes gives the location away ({pr.recall1:.1f} Recall@1 from its ID alone)."),
        (0, "", f"Phone order helps PRiSM's probe only for seen speakers ({sg(o_rel['diff'])} vs {sg(o_sp['diff'])})."),
        (0, "", f"Haryana speakers hold voiceless stops {100 * (voi[0] - 1):.0f}% longer than New Delhi speakers; the "
                "recognizers rarely write it. Intonation helps only in distance."),
        (0, "", "Next: representation probes and every probe on a speaker-disjoint, topic-controlled benchmark."),
    ], y=I(1.45), size=19, gap=12)
    d.text(s, "Code and every number: github.com/unignoramus11/sal-project", LM, I(6.75), I(11.8), I(0.3),
           size=13, color=MUTED, italic=True)

    # Backup slides (Q&A) -------------------------------------------------------------------------------------------
    rows, groups = [], []
    for name, b in [("Majority location", "majority"), ("Speaker-ID lookup", "speaker_lookup")]:
        rows.append([name, fmt(bl(b, "prism").recall1), f"{bl(b, 'prism').mean_km:.0f}",
                     fmt(bl(b, "speaker_split").recall1), f"{bl(b, 'speaker_split').mean_km:.0f}"])
    rows.append(["Picture-ID lookup", fmt(pr.recall1), "", fmt(ps.recall1), ""])
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
    t_main = table_png("table_main", ["Probe", "R@1 PRiSM split", "km", "R@1 new speakers", "km"], rows,
                       group_rows=groups, widths=[2.6, 1.55, 0.85, 1.85, 0.85], fontsize=12)
    s = d.slide("Backup for questions: every probe on the rebuilt data, both splits.")
    d.title(s, "Backup: rebuilt Vaani-Hi, all probes")
    d.image(s, t_main, LM, I(1.3), h=I(5.5))
    d.caption(s, "Recall@1 mean ± sd over seeds (BiGRUs 2 seeds); km: mean error of the seed ensemble.", LM, I(6.95), I(11.8))

    orows = [["Majority location", fmt(odup.loc["majority", "all_recall1"]), fmt(oc("prism", "majority")), fmt(oc("speaker_split", "majority"))],
             ["Transcript match", fmt(odup.loc[NN, "all_recall1"]), fmt(oc("prism", NN)), fmt(oc("speaker_split", NN))]]
    for lab, key in [("ECAPA voice embedding", "dense_ecapa_official"), ("W2V2 phone bigram", "tp_w2v2_official_phone_bigram"),
                     ("ZIPA phone bigram", "tp_zipa_official_phone_bigram"), ("W2V2 char BiGRU (PRiSM)", "tp_w2v2_official_char_gru"),
                     ("W2V2 phone BiGRU", "tp_w2v2_official_phone_gru"), ("W2V2 shuffled phone BiGRU", "tp_w2v2_official_phone_shuffle")]:
        cells = []
        for c in (f"{key}_prism", f"{key}_prism_clean", f"{key}_speaker_split_clean"):
            m_, sd_, n_ = seedmean(c)
            cells.append(fmt(m_, sd_) if n_ else "–")
        orows.append([lab] + cells)
    t_off = table_png("table_official", ["Probe", "Released", "Corrected", "Corrected, new speakers"], orows,
                      widths=[2.8, 1.4, 1.6, 2.9], fontsize=13)
    s = d.slide("Backup: PRiSM's release as published and corrected.")
    d.title(s, "Backup: PRiSM's release, all probes")
    d.image(s, t_off, LM, I(1.5), w=I(9.0))
    d.caption(s, "Recall@1, mean ± sd over seeds. Corrected: one row per recording, conflicting labels removed, Shimla moved; "
                 "speaker-disjoint split from voice clusters (2.1% speaker leakage when checked on our rebuild).",
              LM, I(6.95), I(11.8))

    z = look.loc["tp_zipa_rebuild_phone_bigram_speaker_split"]
    s = d.slide("Backup: checks of our own split (look-alike speaker codes, coordinates, picture lookup).")
    d.title(s, "Backup: checking our own split")
    d.items(s, [
        (0, "Look-alike speaker codes:", f"codes of one district whose voices are as close as two halves of one speaker's "
                                         f"recordings: {e.lookalike_n / e.all_n:.0%} of new-speaker test utterances have one "
                                         f"in training. ECAPA lift {sg(e.lookalike_lift)} on those vs {sg(e.strict_lift)} on "
                                         f"the rest; ZIPA phone bigrams {sg(z.strict_lift)} on the rest ({sg(z.all_lift)} on all)."),
        (0, "Coordinates:", f"PRiSM's recipe averages a pincode's post offices; some directory rows are hundreds of km off. "
                            f"Dhar 454552 is {dhar.km_from_cluster:.0f} km from its office cluster in our rebuild (Ranchi "
                            "834008: 61 km); in the release, Shimla 837 km and Sarguja 64 km."),
        (0, "Picture lookup on shared pictures:", f"{pic.loc[('shared', 'speaker_split', 'picture lookup')].recall1:.1f} vs "
                                                  f"majority {pic.loc[('shared', 'speaker_split', 'majority')].recall1:.1f} on "
                                                  "new speakers: shared pictures do remove topic."),
        (0, "Duplicates:", "3 recordings appear twice in our rebuild, against 4,559 in the release."),
    ], y=I(1.4), size=17, gap=12)

    s = d.slide("Backup: doubled phones in recognizer output, by state.")
    d.title(s, "Backup: doubled phones in recognizer output")
    d.image(s, RES / "figures" / "fig_dpr_w2v2.png", LM, I(1.4), h=I(4.4))
    d.image(s, RES / "figures" / "fig_dpr_zipa.png", I(6.85), I(1.4), h=I(4.4))
    d.caption(s, "Share of consonants written doubled, by state (speakers in brackets), 95% speaker-bootstrap CI. Haryana "
                 "does not double more with either recognizer.", LM, I(6.95), I(11.8))
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
