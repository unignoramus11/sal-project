"""Mid-evaluation deck (3 Nov 2026), built from results/ in the style of our seminar decks
(internal/notes/presentation_style.md: sage content slides, deep-green title and summary, kicker + claim title,
Carlito body, italic captions), with the sections, order and contents of the course's Mid_Project_Template:
problem definition (description, motivation, objectives), overall methodology (diagram), experiments description
(database with samples; experiments with metrics), work done, results, analysis, timelines, summary. Sized for a
7-minute talk; backup slides follow. Speaker notes give the presenter, the time and the points of each slide.

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
    return f"{sg(r[unit + '_diff'])} [{r[unit + '_lo']:.1f}, {r[unit + '_hi']:.1f}]".replace("-", "−")


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

    def _slide(self, dark=False, notes=""):
        s = self.p.slides.add_slide(self.p.slide_layouts[6])
        if notes:
            s.notes_slide.notes_text_frame.text = notes
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

def table_png(name, header, rows, col_align=None, bold_cells=(), group_rows=(), widths=None, fontsize=11,
              family="serif"):
    """Booktabs-style table rendered with matplotlib (Times-like serif), transparent background."""
    ncol = len(header)
    widths = widths or [1.0] * ncol
    tot = sum(widths)
    nrows = len(rows) + 1
    plt.rcParams["font.family"] = family
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


def sg(x, nd=1):
    """Signed number with a true minus sign."""
    return f"{x:+.{nd}f}".replace("-", "−")


def fmt(m, sd=None):
    return f"{m:.1f}" if sd is None else f"{m:.1f} ± {sd:.1f}"


# ------------------------------------------------------------------ charts in the deck's colours

from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

for _f in Path.home().glob("Library/Fonts/Carlito-*.ttf"):
    font_manager.fontManager.addfont(str(_f))
# two shades of one hue (ordinal pair, validated on the sage surface): speakers in training / new speakers
SHARED, NEW, LINK = "#12352a", "#5c8f72", "#a9c3b2"


def _axes(w, h):
    plt.rcParams.update({"font.family": "Carlito", "font.size": 13})
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9aa79f")
    ax.tick_params(colors="#" + BODY, length=0, labelsize=13)
    ax.xaxis.grid(True, color="#c9d9c1", lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _legend(ax, labels):
    h = [Line2D([], [], marker="o", ls="", ms=10, color=c, label=lab) for c, lab in zip((SHARED, NEW), labels)]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=12.5,
              handletextpad=0.3, columnspacing=1.6, borderaxespad=0.3)


def _save(fig, name):
    out = BUILD / f"{name}.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def dumbbell(name, rows, xlabel, labels, w=11.6, h=4.5, xmax=105):
    """rows: (label, value with speakers shared, value on new speakers), top to bottom. Both values labelled."""
    fig, ax = _axes(w, h)
    y = np.arange(len(rows))[::-1]
    for yi, (_, a, b) in zip(y, rows):
        ax.plot([min(a, b), max(a, b)], [yi, yi], color=LINK, lw=3, zorder=1, solid_capstyle="round")
        for v, c in ((a, SHARED), (b, NEW)):
            ax.scatter([v], [yi], s=120, color=c, zorder=3, edgecolor="#" + SAGE, linewidth=2)
        off = 0.02 * xmax + max(0.0, 0.03 * xmax - abs(a - b)) / 2       # keep labels clear of overlapping dots
        ax.text(max(a, b) + off, yi, f"{max(a, b):.1f}", va="center", ha="left", fontsize=12.5, color="#" + INK)
        ax.text(min(a, b) - off, yi, f"{min(a, b):.1f}", va="center", ha="right", fontsize=12.5, color="#" + BODY)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-6, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=13, color="#" + CAP)
    _legend(ax, labels)
    return _save(fig, name)


def forest(name, rows, xlabel, labels, w=8.2, h=4.7):
    """rows: (label, (est, lo, hi) with speakers shared, (est, lo, hi) on new speakers); paired differences."""
    fig, ax = _axes(w, h)
    y = np.arange(len(rows))[::-1] * 1.0
    for yi, (_, a, b) in zip(y, rows):
        for (e, lo, hi), c, dy in ((a, SHARED, 0.17), (b, NEW, -0.17)):
            ax.plot([lo, hi], [yi + dy] * 2, color=c, lw=2.2, solid_capstyle="round", zorder=2)
            ax.scatter([e], [yi + dy], s=80, color=c, zorder=3, edgecolor="#" + SAGE, linewidth=1.5)
            if hi < 0.5:                                         # interval left of zero: label on its left
                ax.text(lo - 0.6, yi + dy, sg(e), va="center", ha="right", fontsize=12, color="#" + BODY)
            else:
                ax.text(hi + 0.6, yi + dy, sg(e), va="center", ha="left", fontsize=12, color="#" + BODY)
    ax.axvline(0, color="#" + CAP, lw=1)
    los = [v[1] for r in rows for v in r[1:]]
    his = [v[2] for r in rows for v in r[1:]]
    ax.set_xlim(min(los) - 4, max(his) + 3.5)                    # room for labels on both sides
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=13, color="#" + CAP)
    _legend(ax, labels)
    return _save(fig, name)


def _arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(1, x1, y1, x2, y2)
    c.line.color.rgb = rgb(KICK)
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def _box(d, s, txt, x, y, w, h, dashed=False, size=13):
    from pptx.enum.shapes import MSO_SHAPE
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    b.adjustments[0] = 0.12
    b.fill.solid()
    b.fill.fore_color.rgb = rgb("f6faf3")
    b.line.color.rgb = rgb("9aa79f" if dashed else KICK)
    b.line.width = Pt(1.25)
    if dashed:
        b.line.dash_style = 4                                   # dash
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.08))
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size, r.font.name = Pt(size if i == 0 else size - 1.5), SANS
        r.font.bold = i == 0
        r.font.color.rgb = rgb(CAP if dashed else (INK if i == 0 else BODY))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return b



# ------------------------------------------------------------------ content

def build():
    d = Deck()
    I = Inches
    splits = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet")
    meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                           columns=["metadata_idx", "state", "district", "pincode", "speaker", "audio_seconds"])
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
    gk, gks = pc.loc[("all", "best constant - prosody, km")], pc.loc[("strict", "best constant - prosody, km")]
    e = look.loc["dense_ecapa_rebuild_speaker_split"]
    z = look.loc["tp_zipa_rebuild_phone_bigram_speaker_split"]
    dhar = coords[coords.data == "rebuild"].sort_values("km_from_cluster", ascending=False).iloc[0]

    def eff(label):
        m_ = re.search(re.escape(label) + r".*?= x([\d.]+)[^(]*\(95% CI x([\d.]+) to x([\d.]+)\).*?tokens (\d+)", mm)
        return float(m_[1]), float(m_[2]), float(m_[3]), int(m_[4])
    voi, clo, val = eff("voiceless interval (primary)"), eff("closure, tokens with a burst"), \
        eff("validity: lexical geminate voiceless interval")
    n_runs = sum(1 for p in (RES / "runs").glob("*/metrics.json"))
    n_tx = {}
    for part in ("rebuild", "official", "districts"):
        f = ROOT / "outputs" / "transcripts" / f"w2v2_{part}.parquet"
        if f.exists():
            n_tx[part] = pq.ParquetFile(f).metadata.num_rows
    names = {"MadhyaPradesh": "Madhya Pradesh", "HimachalPradesh": "Himachal Pradesh"}

    # 1. Project title ---------------------------------------------------------------------------------------------
    s = d._slide(dark=True, notes="Mohit (10 s). Title, team, base paper.")
    d.text(s, "MID EVALUATION  ·  SPEECH ANALYSIS AND LINGUISTICS", LM, I(1.55), I(12), I(0.3), size=9,
           color="8fb5a0", bold=True, spacing=200)
    d.text(s, "Order, Duration, or Melody?", LM, I(2.0), I(12), I(0.9), size=42, font=SERIF, bold=True, color="ffffff")
    d.text(s, "Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech", LM, I(2.95),
           I(12), I(0.5), size=19, font=SERIF, italic=True, color=LIGHT)
    ln = s.shapes.add_connector(1, LM, I(3.8), LM + I(2.4), I(3.8))
    ln.line.color.rgb = rgb(KICK)
    d.text(s, "Mohit Kumar Singh and Nidhi Vaidya", LM, I(4.1), I(12), I(0.3), size=14, color=LIGHT)
    d.text(s, "Base paper: PRiSM, Bharadwaj et al., ACL 2026 (GEO-v, Vaani-Hi dialect geolocation)", LM,
           I(4.5), I(12), I(0.3), size=14, color=LIGHT)
    d.text(s, "3 November 2026", LM, I(5.6), I(12), I(0.3), size=12, color="9fb8aa")

    # 2. Problem definition ------------------------------------------------------------------------------------------
    s = d._slide(notes="Mohit (40 s).\n"
                       "- GEO-v: place an utterance on the map; PRiSM probes the same recognizer two ways.\n"
                       "- The puzzle: the probe that cannot hear voice, duration or intonation wins, and the explanation "
                       "(phone order) was never tested.\n"
                       "- Four objectives, one per property of speech; RQ0 first asks whether the benchmark measures "
                       "dialect at all.")
    d.header(s, "Problem definition", "PRiSM's transcript probe beats its representation probe on GEO-v")
    d.deflist(s, [
        ("Description", "PRiSM (ACL 2026) probes phone recognizers on GEO-v: place a Hindi-belt utterance on the map "
                        "(Vaani-Hi, about 30 pincodes over ~1,500 km). A transcript probe over the predicted phones "
                        "scores 146 km mean error, a representation probe over hidden states 253 km."),
        ("Motivation", "Only the representation probe hears voice, duration and intonation, yet it loses. PRiSM credits "
                       "phone order without testing it, and if the benchmark rewards speakers or topics, the comparison "
                       "says little about dialect."),
    ], y=I(1.75), label_w=I(2.1), size=15.5, row=I(1.05))
    d.text(s, "Objectives", LM, I(3.95), I(2.1), I(0.4), size=15.5, font=SERIF, bold=True, color=INK)
    obj = [("RQ0  Speakers, topic", "Does PRiSM's split let probes recognise speakers and picture topics instead of dialect?"),
           ("RQ1  Order", "Does local phone order carry location beyond the phone inventory?"),
           ("RQ2  Duration", "Is Haryanvi consonant lengthening in the recognizers' output, and in the audio?"),
           ("RQ3  Melody", "Does intonation, which transcripts discard, add location signal?")]
    for k, (a, b) in enumerate(obj):
        yy = I(3.95 + 0.62 * k)
        d.text(s, a, LM + I(2.3), yy, I(2.4), I(0.5), size=15, bold=True, color=KICK)
        d.text(s, b, LM + I(4.75), yy, TW - I(4.75), I(0.5), size=15)

    # 3. Overall methodology -----------------------------------------------------------------------------------------
    s = d._slide(notes="Mohit (30 s).\n"
                       "- Four views of each utterance, each feeding a probe that predicts latitude and longitude.\n"
                       "- The representation-probe path is dashed: not run yet.\n"
                       "- References that learn nothing, and every probe under two splits: that is what exposes leakage.")
    d.header(s, "Overall methodology", "Four views of each utterance, each probed for location")
    ys, rh = [1.7, 2.62, 3.54, 4.46], 0.7
    mid = [I(y + rh / 2) for y in ys]
    _box(d, s, "Utterance\naudio", I(0.6), I(ys[0]), I(1.3), I(ys[3] + rh - ys[0]))
    for txt, r0, r1 in [("Phone recognizer\nW2V2 · ZIPA", 0, 2), ("Voice encoder\nECAPA-TDNN", 2, 3),
                        ("Praat\nF0 · voicing · intensity", 3, 4)]:
        top, bot = ys[r0], ys[r1 - 1] + rh
        _box(d, s, txt, I(2.25), I(top), I(2.3), I(bot - top))
        _arrow(s, I(1.9), I((top + bot) / 2), I(2.25), I((top + bot) / 2))
    reps = ["Phone string", "Hidden states", "Voice embedding", "Contours, 100 per s"]
    probes = [("Transcript probes", "bag MLPs · BiGRU · shuffled BiGRU", False),
              ("Representation probe", "PRiSM's RP-Attn · not run yet", True),
              ("Voice probe", "MLP", False), ("Prosody probe", "BiGRU · late fusion", False)]
    for k, (rep, (pt, pst, dash)) in enumerate(zip(reps, probes)):
        _box(d, s, rep, I(4.9), I(ys[k]), I(2.1), I(rh), dashed=dash)
        _box(d, s, f"{pt}\n{pst}", I(7.35), I(ys[k]), I(3.15), I(rh), dashed=dash)
        _arrow(s, I(4.55), mid[k], I(4.9), mid[k])
        _arrow(s, I(7.0), mid[k], I(7.35), mid[k])
        _arrow(s, I(10.5), mid[k], I(10.85), mid[k])
    _box(d, s, "Location\nlatitude, longitude", I(10.85), I(ys[0]), I(1.85), I(ys[3] + rh - ys[0]))
    d.bullets(s, [
        "References that learn nothing about dialect: majority location, speaker-ID lookup, picture-ID lookup, and a "
        "nearest-neighbour transcript match.",
        "Every probe under two splits: PRiSM's random split (test speakers also in training) and a speaker-disjoint "
        "split (new speakers). Recall@1 and mean km, 95% CIs by bootstrap over speakers.",
    ], y=I(5.5), h=I(1.5), size=15, gap=6)

    # 4. Experiments description: database ---------------------------------------------------------------------------
    w2 = pd.read_parquet(ROOT / "outputs" / "transcripts" / "w2v2_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
    zp = pd.read_parquet(ROOT / "outputs" / "transcripts" / "zipa_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
    smp = (meta.merge(prompts, on="metadata_idx").merge(w2, on="metadata_idx")
               .merge(zp, on="metadata_idx", suffixes=("_w", "_z")).set_index("metadata_idx"))
    rows_s = []
    for idx in (9, 4391, 17440):                         # three real utterances, picked for clean recognizer output
        r = smp.loc[idx]
        rows_s += [[f"{r.district}, {names.get(r.state, r.state)}",
                    "own district" if r.prompt_type == "SPECIFIC" else "shared", "W2V2", r.processed_transcript_w[:28] + "…"],
                   ["", "", "ZIPA", r.processed_transcript_z[:28] + "…"]]
    t_samples = table_png("db_samples", ["District, state", "Picture", "Model", "Recognized phones (start)"], rows_s,
                          col_align=["left"] * 4, widths=[2.1, 1.2, 0.75, 3.1], fontsize=12, family="Carlito")
    s = d._slide(notes="Nidhi (30 s).\n"
                       "- Vaani: crowd-sourced speech, people describe pictures. We rebuilt PRiSM's subset with PRiSM's code; "
                       "the speaker code in each file name makes a speaker-disjoint split possible.\n"
                       "- Two thirds of the pictures belong to the speaker's own district, so topic can give the place away.\n"
                       "- Samples: the two recognizers write the same speech differently.")
    d.header(s, "Experiments description  ·  database", "Two versions of Vaani-Hi: our rebuild and PRiSM's release")
    d.stats(s, [(f"{len(meta):,}", f"utterances in our rebuild\n({meta.audio_seconds.sum() / 3600:.1f} h, 26 pincodes, 9 states)"),
                (f"{len(per_spk):,}", f"speakers (codes from file names;\nmedian {per_spk.median():.0f}, max {per_spk.max()} utterances each)"),
                ("26,385", "rows in PRiSM's release,\n30 pincodes, no speaker IDs"),
                (f"{specific:.0f}%", "of utterances describe a picture\nof the speaker's own district")])
    d.bullets(s, [
        "Rebuild: PRiSM's selection code rerun on the public Vaani corpus; identical on two machines.",
        "Release: PRiSM's GEO-v copy, to reproduce its number and audit it.",
        f"RQ2: Vaani audio from 3 Haryana districts and New Delhi ({n_tx.get('districts', 0):,} utterances); "
        "Vaani-Hi has only 12 Haryana speakers.",
    ], x=LM, y=I(3.65), w=I(5.3), h=I(3.0), size=14.5, gap=8)
    im = d.image(s, t_samples, I(6.3), I(3.7), w=I(6.4))
    d.caption(s, "Three test utterances and the start of each recognizer's phone string.",
              I(6.3), I(6.85), I(6.4))

    # 5. Experiments description: experiments ------------------------------------------------------------------------
    s = d._slide(notes="Nidhi (25 s). One line per experiment; metrics on the right. Comparisons are paired on the same "
                       "test utterances; CIs resample speakers, because one speaker's utterances are not independent.")
    d.header(s, "Experiments description  ·  experiments", "Five experiments, each with its metric")
    d.deflist(s, [
        ("1  Speakers, topic", "Every probe under PRiSM's split and a speaker-disjoint split; lookups by speaker and "
                               "picture ID; transcript matcher; voice probe; control task.  [Recall@1, mean km, lift "
                               "over the majority]"),
        ("2  PRiSM's release", "Reproduce PRiSM's transcript probe; audit duplicates, leakage and coordinates; rerun "
                               "every probe on a corrected, speaker-disjoint version.  [Recall@1, mean km]"),
        ("3  Order", "Unigram vs bigram bags; BiGRU vs the same BiGRU on shuffled phones; characters vs phones; two "
                     "recognizers.  [paired Recall@1 difference, km, MDL codelength]"),
        ("4  Duration", "Doubled-phone rate in recognizer output; voiceless-stop and closure duration measured in "
                        "the audio, Haryana vs New Delhi.  [% doubled, duration ratio from mixed models]"),
        ("5  Melody", "Pooled prosody statistics; BiGRU over F0, voicing and intensity contours; late fusion with the "
                      "transcript probe.  [Recall@1, mean km]"),
    ], y=I(1.8), label_w=I(2.4), size=15, row=I(0.98))

    # 6. Work done ------------------------------------------------------------------------------------------------------
    s = d._slide(notes="Nidhi (20 s). The numbers behind the work, and what is not done yet: the representation probes "
                       "(frame caches did not fit the Ada quota), the manual audit, and the last seeds.")
    d.header(s, "Work done", "Most of the study is done; representation probes are not")
    d.stats(s, [(f"{sum(n_tx.values()):,}", "utterances transcribed by\n2 phone recognizers"),
                (f"{n_runs}", "trained probe runs\n(transcript, voice, prosody)"),
                (f"{voi[3]:,}", "voiceless stops located by forced\nalignment and measured in Praat"),
                (f"{gru[0]:.1f}", "PRiSM's number reproduced on\nits release (paper 45.7)")])
    d.deflist(s, [
        ("Done", "Rebuilt dataset and speaker-disjoint split; audit and corrected version of PRiSM's release; control "
                 "task and MDL; prosody contours, prosody BiGRU and fusion (Ada); a cross-validated benchmark for the "
                 "final, built and tested."),
        ("Not yet", "Representation probes (frame caches did not fit the Ada quota); blind manual audit of 50 stops by "
                    "both of us; BiGRU seeds 3-5 and prosody seeds 2-5."),
    ], y=I(3.75), label_w=I(1.6), size=15, row=I(1.2))

    # 7. Results: speakers and topic ------------------------------------------------------------------------------------
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
    ], "Test Recall@1 on rebuilt Vaani-Hi (%)", ("test speakers also in training (PRiSM's split)", "new speakers"))
    s = d._slide(notes="Mohit (35 s). RQ0.\n"
                       f"- Under PRiSM's split {leak:.0f}% of test speakers are in training; the speaker ID alone gives "
                       f"{bl('speaker_lookup', 'prism').recall1:.0f} Recall@1.\n"
                       f"- The picture ID alone gives {pr.recall1:.0f}, above PRiSM's 45.7; on new speakers still "
                       f"{ps.recall1:.0f}, above every probe.\n"
                       "- Every trained probe drops on new speakers; the voice embedding falls to the majority.")
    d.header(s, "Results  ·  RQ0 speakers and topic", "Lookups that know nothing about dialect beat every probe")
    d.image(s, c1, LM, I(1.5), w=I(12.1))
    d.caption(s, f"Figure 1. {leak:.1f}% of test utterances have their speaker in training under PRiSM's split. Lookups "
                 "predict the commonest training location of the same speaker or picture; trained probes: mean over seeds.",
              LM, I(6.85), I(12.1))

    # 8. Results: PRiSM's release -----------------------------------------------------------------------------------------
    def oc(proto, probe):
        return ocl[(ocl.protocol == proto) & (ocl.probe == probe)].recall1.iloc[0]

    def osm(key, P):
        return seedmean(f"{key}_{P}_clean")[0]
    c2 = dumbbell("chart_release", [
        ("Transcript match, nothing learned", oc("prism", NN), oc("speaker_split", NN)),
        ("ZIPA phone bigrams", osm("tp_zipa_official_phone_bigram", "prism"), osm("tp_zipa_official_phone_bigram", "speaker_split")),
        ("Voice embedding (ECAPA)", osm("dense_ecapa_official", "prism"), osm("dense_ecapa_official", "speaker_split")),
        ("PRiSM's BiGRU (characters)", osm("tp_w2v2_official_char_gru", "prism"), osm("tp_w2v2_official_char_gru", "speaker_split")),
        ("Shuffled phone BiGRU", osm("tp_w2v2_official_phone_shuffle", "prism"), osm("tp_w2v2_official_phone_shuffle", "speaker_split")),
        ("Majority location", oc("prism", "majority"), oc("speaker_split", "majority")),
    ], "Test Recall@1 on the corrected release (%)", ("released split (speakers shared)", "speaker-disjoint split"),
        w=7.6, h=4.6, xmax=72)
    s = d._slide(notes="Mohit (40 s). PRiSM's own data.\n"
                       f"- Our pipeline reproduces PRiSM's number: {gru[0]:.1f} against 45.7.\n"
                       f"- A voice embedding that hears no words scores {ec[0]:.1f}; a matcher that learns nothing "
                       f"{odup.loc[NN, 'all_recall1']:.1f}: both beat the reported probe.\n"
                       f"- The release repeats recordings ({100 * oleak.loc['test', 'exact_audio_in_train']:.0f}% of test clips "
                       "are in training verbatim) and puts Shimla in Afghanistan.\n"
                       "- Corrected and split by speaker (right): everything falls to the majority.")
    d.header(s, "Results  ·  PRiSM's release", "On PRiSM's own data the voice alone beats PRiSM's probe")
    for k, (num, lab, best) in enumerate([
            (f"{gru[0]:.1f}", f"PRiSM's transcript probe, our code ({gru[2]} seeds; paper 45.7)", False),
            (f"{ec[0]:.1f}", "voice embedding, no words", True),
            (f"{odup.loc[NN, 'all_recall1']:.1f}", "transcript match, nothing learned", False),
            (f"{100 * oleak.loc['test', 'exact_audio_in_train']:.0f}%", f"test clips also in training; {dups:.0f}% of repeats "
                                                                        "carry another pincode", False)]):
        yy = I(1.7 + 1.25 * k)
        d.text(s, num, LM, yy, I(3.6), I(0.6), size=30, font=SERIF, bold=True, color=KICK if best else INK)
        d.text(s, lab, LM, yy + I(0.6), I(3.6), I(0.6), size=11.5, color=CAP)
    d.image(s, c2, I(4.6), I(1.6), w=I(8.1))
    d.caption(s, f"Figure 2. Left: released data and split (Recall@1). Right: corrected release (one row per recording, Shimla "
                 f"moved back {shimla.km_from_valid_median:.0f} km from Afghanistan); speakers clustered from voices.",
              LM, I(6.85), I(12.1))

    # 9. Results: order ---------------------------------------------------------------------------------------------------
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
    ], "Recall@1 gain from phone order, points (95% CI)", ("test speakers in training", "new speakers"), h=5.2)
    s = d._slide(notes="Mohit (30 s). RQ1.\n"
                       "- Each row: a probe that sees phone order minus the same probe without it, same test utterances.\n"
                       f"- PRiSM's BiGRU gains {o_rel['diff']:.0f} points from order when speakers are shared and nothing "
                       "on new speakers: it memorises speakers' sequences.\n"
                       "- Bigram bags keep a gain on new speakers in the rebuild, but bigrams also identify words.")
    d.header(s, "Results  ·  RQ1 order", "Phone order helps mostly when test speakers were seen in training")
    im = d.image(s, c3, LM, I(1.5), w=I(8.2))
    tx = LM + im.width + I(0.4)
    d.bullets(s, [
        "Each row: a probe that sees phone order minus the same probe without it, paired on the same test utterances.",
        f"PRiSM's BiGRU gains {o_rel['diff']:.0f} points from order on its own split and nothing on new speakers: it "
        "memorises speakers' sequences.",
        "Bigram bags keep a gain on the rebuild's new speakers, but they also identify words, and on shared pictures no "
        "probe beats the majority.",
    ], x=tx, y=I(1.7), w=W - tx - I(0.6), h=I(4.8), size=14.5)
    d.caption(s, "Figure 3. Release: corrected rows, speaker-disjoint split from voice clusters (59 test speakers). Rebuild "
                 "BiGRUs: 2 seeds so far.", LM, I(6.85), I(12.1))

    # 10. Results: duration ---------------------------------------------------------------------------------------------
    s = d._slide(notes="Nidhi (30 s). RQ2.\n"
                       f"- {voi[3]:,} single voiceless stops between vowels, found by forced alignment of the standard "
                       "spelling and measured with Praat landmarks.\n"
                       f"- Haryana speakers hold them {100 * (voi[0] - 1):.0f}% longer (closure {100 * (clo[0] - 1):.0f}%), in "
                       "every district and most words.\n"
                       "- The recognizers write only 1-2% of them doubled, so a transcript probe cannot see it.")
    d.header(s, "Results  ·  RQ2 duration", "In the audio, Haryana speakers hold voiceless stops 14% longer")
    d.stats(s, [(f"×{voi[0]:.2f}", f"voiceless interval, Haryana vs\nNew Delhi (95% CI {voi[1]:.2f}-{voi[2]:.2f})"),
                (f"×{clo[0]:.2f}", f"closure alone\n(95% CI {clo[1]:.2f}-{clo[2]:.2f})"),
                (f"×{val[0]:.2f}", f"lexical geminates vs single stops\n(validity, 95% CI {val[1]:.2f}-{val[2]:.2f})"),
                ("1-2%", "of these stops written doubled\nby either recognizer")], best=0)
    d.bullets(s, [
        f"{voi[3]:,} single voiceless stops between vowels in Rohtak, Jhajjar, Charkhi Dadri and New Delhi Vaani audio, "
        "found by forced alignment of the standard spelling and measured with Praat voicing and burst landmarks.",
        "Holds in each Haryana district (×1.13-1.15), in 45 of 57 words, and on speaker means without a model (×1.14); "
        "mixed model with word effects, speaking-rate control and a speaker intercept.",
        "Haryana single stops move about a quarter of the way to geminate length: lengthening, not doubling, and the "
        "recognizers rarely write it.",
    ], y=I(3.7), h=I(3.2), size=15)

    # 11. Results: melody -----------------------------------------------------------------------------------------------
    s = d._slide(notes="Nidhi (25 s). RQ3.\n"
                       "- Pooled prosody statistics carry nothing; the contour BiGRU gives no Recall@1 gain on new speakers.\n"
                       f"- Fused with the transcript probe it lowers mean error by {-fk['diff']:.0f} km, also without "
                       "look-alike speakers. One seed so far.")
    d.header(s, "Results  ·  RQ3 melody", "Intonation adds a little regional information, in distance only")
    d.stats(s, [(f"{pm['recall1']:.1f}", f"prosody BiGRU, Recall@1 on new\nspeakers (majority {bl('majority', 'speaker_split').recall1:.1f})"),
                (f"+{gks['diff']:.0f} km", f"prosody alone vs best fixed location,\nno look-alike speakers [{gks.lo:.0f}, {gks.hi:.0f}]"),
                (f"{sg(fk['diff'])} km", f"fused with the transcript BiGRU,\nmean error [{sg(fk.lo)}, {sg(fk.hi)}]"),
                (sg(fr['diff']), f"fused, Recall@1\n[{sg(fr.lo)}, {sg(fr.hi)}]")], best=2)
    d.bullets(s, [
        "Pooled prosody statistics stay below the majority on both splits. A BiGRU over F0, voicing and intensity "
        "contours (run on Ada) brings no Recall@1 gain on new speakers.",
        f"Its distance gain over a fixed location (+{gk['diff']:.0f} km) comes mostly from speakers with a look-alike code "
        "in training: like the voice embedding, contours carry the voice.",
        f"Late fusion (weight 0.85 on the transcript, chosen on validation) lowers mean error from 441 to 426 km, also "
        f"without look-alike speakers ({sg(fks['diff'])} km). One seed so far.",
    ], y=I(3.7), h=I(3.2), size=15)

    # 12. Analysis --------------------------------------------------------------------------------------------------------
    s = d._slide(notes="Nidhi (30 s). What the results mean, one line per objective; the last line is why the "
                       "representation probes matter for the final.")
    d.header(s, "Analysis", "GEO-v as released measures speakers and topics, not dialect")
    d.deflist(s, [
        ("Speakers", f"{leak:.0f}% of PRiSM's test speakers are in training and "
                     f"{100 * oleak.loc['test', 'exact_audio_in_train']:.0f}% of its test clips are there verbatim; a "
                     "voice-only probe and a matcher that learns nothing both beat PRiSM's probe."),
        ("Topic", f"The picture ID alone gives {pr.recall1:.1f} Recall@1; on shared pictures, which carry no location, the "
                  "transcript probes fall to the majority."),
        ("Order", f"The BiGRU's gain from order ({sg(o_rel['diff'])}) appears only for speakers seen in training, so "
                  "PRiSM's explanation holds only under its leaky split."),
        ("Own split", f"Look-alike speaker codes leak the voice (ECAPA {sg(e.lookalike_lift)} vs {sg(e.strict_lift)}) but "
                      f"not the transcripts; PRiSM's coordinate recipe misplaces Dhar by {dhar.km_from_cluster:.0f} km."),
        ("Duration", "Real in the audio but rarely written by the recognizers: only a representation probe could use it."),
    ], y=I(1.75), label_w=I(1.9), size=15, row=I(0.98))

    # 13. Timelines for completion ----------------------------------------------------------------------------------------
    s = d._slide(notes="Nidhi (15 s). Four weeks to the final on 4 December; representation probes come first because "
                       "PRiSM's central comparison needs them.")
    d.header(s, "Timelines for completion", "Four weeks to the final evaluation on 4 December")
    d.deflist(s, [
        ("3-9 Nov", "Representation probes (PRiSM's RP-Attn, last layer, 3 recognizers); remaining seeds.  Nidhi, Mohit"),
        ("10-16 Nov", "Every probe on the cross-validated, speaker-disjoint benchmark; blind manual audit.  Mohit; both"),
        ("17-23 Nov", "Order-aware representation probes across layers (RQ1); length in hidden states (RQ2).  Nidhi, Mohit"),
        ("24-30 Nov", "Prosody seeds and fusion on the benchmark (RQ3); statistics and figures.  Nidhi, Mohit"),
        ("1-4 Dec", "Report in ACL format and final presentation.  Both"),
    ], y=I(1.8), label_w=I(1.7), size=15.5, row=I(0.95))

    # 14. Summary -----------------------------------------------------------------------------------------------------------
    s = d._slide(dark=True, notes="Nidhi (20 s). Four findings and what comes next. Then questions.")
    d.header(s, "Summary", "Takeaways", dark=True)
    d.bullets(s, [
        f"GEO-v rewards recognising speakers: on PRiSM's release a voice-only embedding ({ec[0]:.1f}) beats PRiSM's "
        f"reproduced probe ({gru[0]:.1f}), and every probe falls to the majority once speakers are separated.",
        f"The picture a speaker describes gives the location away ({pr.recall1:.1f} Recall@1 from the picture ID alone); "
        "without topic, transcript probes fall to the majority on new speakers.",
        f"Phone order helps PRiSM's probe only for seen speakers ({sg(o_rel['diff'])} vs {sg(o_sp['diff'])} Recall@1).",
        "Haryana speakers hold voiceless stops 14% longer than New Delhi speakers; the recognizers rarely write it.",
        f"Intonation helps only in distance: fused with the transcript probe it saves {-fk['diff']:.0f} km.",
    ], y=I(1.75), h=I(4.7), size=16.5, color=LIGHT, gap=10)
    d.text(s, "Code, data recipe and every number: github.com/unignoramus11/sal-project", LM, I(6.6),
           I(12), I(0.3), size=11, italic=True, color="9fb8aa")

    # Backup slides for questions ----------------------------------------------------------------------------------------
    lk = ctrl.loc["dense_ecapa_rebuild_prism"]
    s = d._slide(notes="Backup: speaker leakage in numbers.")
    d.header(s, "Backup  ·  RQ0 speakers", "Under PRiSM's split, the speaker ID alone solves the task")
    d.stats(s, [(f"{leak:.1f}%", "test utterances whose speaker\nis also in training"),
                (f"{bl('speaker_lookup', 'prism').recall1:.1f}", "Recall@1 of a lookup that only\nknows the speaker ID"),
                (f"{cfg.loc['dense_ecapa_rebuild_speaker_split', 'recall1_mean']:.1f}",
                 f"voice embedding (ECAPA) on new speakers\n({cfg.loc['dense_ecapa_rebuild_prism', 'recall1_mean']:.1f} on PRiSM's split)"),
                (f"{lk.control_recall1:.1f}", f"ECAPA on control labels\n(control majority {lk.control_majority:.1f})")], best=1)
    d.bullets(s, [
        f"Majority baseline {bl('majority', 'prism').recall1:.1f} (random) and {bl('majority', 'speaker_split').recall1:.1f} "
        "(speaker-disjoint). On control labels (each speaker gets another speaker's location) ECAPA still scores far above "
        "the control majority: most of its random-split score is speaker recognition.",
        f"A nearest-neighbour transcript matcher reaches {knn.loc[('w2v2', 'prism', 1), 'recall1']:.1f} on the random split "
        f"and {knn.loc[('w2v2', 'speaker_split', 1), 'recall1']:.1f} on new speakers; for 25% of test utterances the closest "
        "training transcript is the same speaker's (chance 0.3%).",
    ], y=I(3.7), h=I(3.2))

    g = cfg.loc["tp_w2v2_rebuild_phone_bigram_speaker_split_generic"]
    st = state.loc["tp_zipa_rebuild_phone_bigram_speaker_split"]
    s = d._slide(notes="Backup: topic control with shared pictures.")
    d.header(s, "Backup  ·  topic", "Without topic, the transcript probes fall to the majority")
    d.stats(s, [(f"{st.all_state_acc:.0f}%", f"correct state, ZIPA phone bigrams,\nnew speakers (majority {st.all_majority:.0f}%)"),
                (sg(g.lift_recall1), "Recall@1 lift when trained and\ntested on shared pictures only"),
                (f"+{g.km_gain:.0f} km", f"km gain on shared pictures\n(95% CI {g.km_gain_lo:.0f} to {g.km_gain_hi:.0f})"),
                (f"{pic.loc[('shared', 'speaker_split', 'picture lookup')].recall1:.1f}",
                 f"picture lookup on shared pictures,\nnew speakers (majority {pic.loc[('shared', 'speaker_split', 'majority')].recall1:.1f})")])
    d.bullets(s, [
        "Shared pictures carry no location (the picture lookup is below the majority on them), but their share differs by "
        "state (12-58%), so they get their own baselines: majority 21.3 Recall@1, best constant 537 km.",
        "Trained and tested on shared pictures, no transcript probe beats the majority on Recall@1 or state; km gains are "
        "48-94 km, and only the two W2V2 ones have CIs above zero (119 test speakers).",
    ], y=I(3.7), h=I(3.2))

    s = d._slide(notes="Backup: checks of our own split.")
    d.header(s, "Backup  ·  checking our own split", "Look-alike speaker codes leak the voice, not the transcript")
    d.stats(s, [(f"{e.lookalike_n / e.all_n:.0%}", "of new-speaker test utterances have a\nlook-alike code in training"),
                (f"{sg(e.lookalike_lift)} / {sg(e.strict_lift)}", "ECAPA lift: look-alike utterances /\nthe strict rest"),
                (sg(z.strict_lift), f"ZIPA phone bigram lift on the strict\nrest ({sg(z.all_lift)} on all)"),
                (f"{dhar.km_from_cluster:.0f} km", f"pincode {int(dhar.pincode)} ({dhar.district}) off its\npost offices in the rebuild")],
            best=2)
    d.bullets(s, [
        "Speaker codes come from file names. Codes of one district whose voices are as close as two halves of one "
        "speaker's recordings are grouped (1,570 codes → 1,327 groups); gender agrees in 98% of close pairs.",
        "PRiSM's recipe averages a pincode's post offices, and some directory rows are hundreds of km off; Dhar lands 88 km "
        "from Bhopal instead of 45 km from the other Dhar pincode. Recall@1 hardly moves; km does.",
    ], y=I(3.7), h=I(3.2))

    k_ = lambda c: mdl.loc[c, "gain_over_prior"]  # noqa: E731
    s = d._slide(notes="Backup: bigrams, phone units and MDL on the rebuild.")
    d.header(s, "Backup  ·  RQ1 order and units", "On new speakers, bigrams and phone units help a little")
    d.stats(s, [(cd("tp_w2v2_rebuild_phone_bigram_speaker_split", "tp_w2v2_rebuild_phone_unigram_speaker_split").split(" ")[0],
                 "phone bigram vs unigram, W2V2,\nnew speakers (Recall@1)"),
                (cd("tp_w2v2_rebuild_phone_bigram_speaker_split", "tp_w2v2_rebuild_char_bigram_speaker_split").split(" ")[0],
                 "phones vs characters, W2V2 bigrams,\nnew speakers"),
                (f"{k_('mdl_zipa_rebuild_phone_bigram_speaker_split'):.1f} kbits",
                 f"MDL saving, ZIPA phone bigrams, new\nspeakers ({k_('mdl_zipa_rebuild_phone_bigram_prism'):.1f} on the random split)")])
    d.bullets(s, [
        f"Bigram over unigram {cd('tp_w2v2_rebuild_phone_bigram_speaker_split', 'tp_w2v2_rebuild_phone_unigram_speaker_split')} "
        "Recall@1 and 33 km less error; MDL agrees on every speaker ordering.",
        "PRiSM tokenises by character, which splits kʰ into k + ʰ. Phones beat characters for W2V2 on Recall@1; in MDL, "
        "phone units help ZIPA (0.7 kbits) but not W2V2.",
    ], y=I(3.7), h=I(3.2))

    hv_w = pd.read_csv(RES / "dpr" / "w2v2_haryana_vs_rest.csv").iloc[0]
    hv_z = pd.read_csv(RES / "dpr" / "zipa_haryana_vs_rest.csv").iloc[0]
    s = d._slide(notes="Backup: doubled phones in recognizer output by state.")
    d.header(s, "Backup  ·  RQ2 duration", "Doubled phones in recognizer output depend on the recognizer")
    d.image(s, RES / "figures" / "fig_dpr_w2v2.png", LM, I(1.5), h=I(3.6))
    d.image(s, RES / "figures" / "fig_dpr_zipa.png", LM + I(6.3), I(1.5), h=I(3.6))
    d.bullets(s, [
        f"W2V2 doubles {100 * hv_w.haryana:.2f}% of consonants in Haryana vs {100 * hv_w.rest:.2f}% elsewhere "
        f"(p = {hv_w.p:.3f}); ZIPA {100 * hv_z.haryana:.2f}% vs {100 * hv_z.rest:.2f}% (p = {hv_z.p:.2f}). Neither shows "
        "Haryanvi doubling, and Vaani-Hi has only 12 Haryana speakers.",
    ], y=I(5.35), h=I(1.5), size=15)

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
    png = table_png("table_main", ["Probe", "R@1 random", "km", "R@1 new speakers", "km"], rows, group_rows=groups,
                    widths=[2.5, 1.3, 0.75, 1.8, 0.75], fontsize=10.5)
    s = d._slide(notes="Backup: all probes on the rebuild.")
    d.header(s, "Backup  ·  rebuilt Vaani-Hi", "Every probe drops on new speakers; phone bigrams keep the most")
    d.image(s, png, LM, I(1.55), h=I(5.1))
    d.caption(s, "Table A1. Test Recall@1 (mean ± sd over seeds; BiGRUs 2 seeds) and mean km of the seed ensemble.",
              LM, I(6.85), I(12.1))

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
    png = table_png("table_official", ["Probe", "Released", "Corrected", "Corrected, new speakers"], orows,
                    widths=[2.7, 1.4, 1.9, 2.5], fontsize=12)
    s = d._slide(notes="Backup: all probes on PRiSM's release.")
    d.header(s, "Backup  ·  PRiSM's release", "Once speakers are separated, every probe falls to the majority")
    d.image(s, png, LM, I(1.65), w=I(8.5))
    d.caption(s, "Table A2. Test Recall@1, mean ± sd over seeds. Released: data and split as published. Corrected: released "
                 "split on the corrected rows, or a pseudo-speaker-disjoint split.", LM, I(6.85), I(12.1))

    s = d._slide(notes="Backup: metrics and limitations.")
    d.header(s, "Backup  ·  reference", "Metrics and limitations")
    d.deflist(s, [
        ("Metrics", "Recall@1: is the nearest distinct test location the true one (PRiSM's definition); mean great-circle "
                    "km; lift over the majority and km gain over the best constant on the same utterances."),
        ("Statistics", "5 seeds (BiGRUs 2-3 so far); 95% CIs and paired tests by bootstrap over speakers, because "
                       "utterances of one speaker are not independent."),
        ("Limitations", "12 Haryana speakers in Vaani-Hi; New Delhi stands for standard Hindi only approximately; speakers "
                        "on the release are estimated; recognizer output is not a phonetic gold standard."),
    ], label_w=I(2.0), size=15, row=I(1.25))
    d.text(s, "Bharadwaj et al. (2026). PRiSM: Benchmarking Phone Realization in Speech Models. ACL 2026, pages "
              "18093-18112.", LM, I(6.7), I(12), I(0.3), size=10, italic=True, color=CAP)
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
