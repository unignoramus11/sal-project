"""Mid-evaluation deck (3 Nov 2026), built from results/ in the style of our seminar decks
(internal/notes/presentation_style.md): 16:9, sage content slides, deep-green title and takeaways, kicker +
claim title, Carlito body with en-dash bullets, italic grey captions, page numbers.

Output: slides/mid/mid_evaluation.pptx (+ .pdf with --pdf, via LibreOffice)
Run:    uv run --with python-pptx python slides/mid/build_deck.py --pdf
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)

SAGE, DEEP, KICK, INK, BODY, CAP, LIGHT = "e2f0d9", "12352a", "5c8f72", "12352a", "26332d", "6e7b74", "e4ede7"
SERIF, SANS, MONO = "Caladea", "Carlito", "Liberation Mono"
W, H = Inches(13.333), Inches(7.5)
LM = Inches(0.6)

cfg = pd.read_csv(RES / "tables" / "configs.csv").set_index("config")
base = pd.read_csv(RES / "baselines.csv")
comp = pd.read_csv(RES / "tables" / "comparisons.csv")
ctrl = pd.read_csv(RES / "tables" / "control.csv").set_index(["probe"]) if (RES / "tables" / "control.csv").exists() else None
mdl = pd.read_csv(RES / "mdl" / "summary.csv").set_index("config")
state = pd.read_csv(RES / "tables" / "state_level.csv").set_index("config")
knn = pd.read_csv(RES / "tables" / "knn_transcripts.csv").set_index(["transcripts", "protocol", "k"])


def rgb(h):
    return RGBColor.from_string(h)


def bl(name, proto, prompts="all", split="test"):
    r = base[(base.baseline == name) & (base.protocol == proto) & (base.prompts == prompts) & (base.split == split)]
    return r.iloc[0]


def has(c):
    return c in cfg.index


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
            self.text(s, str(self.n), W - Inches(0.9), H - Inches(0.45), Inches(0.5), Inches(0.3), size=8,
                      color=LIGHT if dark else "9aa79f", align=PP_ALIGN.RIGHT)
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=14, color=BODY, font=SANS, bold=False, italic=False, align=PP_ALIGN.LEFT,
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
            self.text(s, kicker.upper(), LM, Inches(0.38), Inches(12), Inches(0.25), size=8,
                      color=KICK if not dark else "8fb5a0", bold=True, spacing=200)
        self.text(s, title, LM, Inches(0.68), Inches(12.1), Inches(0.8), size=27, color=LIGHT if dark else INK,
                  font=SERIF, bold=True)

    def bullets(self, s, items, x, y, w, h, size=13.5, color=BODY, gap=9):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        for i, it in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.space_after = Pt(gap)
            para.line_spacing = 1.18
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.3))))
            pPr.set("indent", str(-Emu(Inches(0.3))))
            bu = pPr.makeelement(qn("a:buFont"), {"typeface": SANS})
            pPr.append(bu)
            ch = pPr.makeelement(qn("a:buChar"), {"char": "–"})
            pPr.append(ch)
            r = para.add_run()
            r.text = it
            r.font.size, r.font.name = Pt(size), SANS
            r.font.color.rgb = rgb(color)
        return tb

    def image(self, s, path, x, y, w=None, h=None):
        return s.shapes.add_picture(str(path), x, y, width=w, height=h)

    def caption(self, s, txt, x, y, w, align=PP_ALIGN.CENTER):
        self.text(s, txt, x, y, w, Inches(0.3), size=9, color=CAP, italic=True, align=align)

    def stats(self, s, items, y, best=None):
        n = len(items)
        cw = Inches(12.1) / n
        for i, (num, lab) in enumerate(items):
            x = LM + cw * i
            self.text(s, num, x, y, cw - Inches(0.2), Inches(0.8), size=34, font=SERIF, bold=True,
                      color=KICK if best == i else INK)
            self.text(s, lab, x, y + Inches(0.75), cw - Inches(0.3), Inches(0.6), size=10, color=CAP)

    def deflist(self, s, pairs, y, label_w=Inches(2.1), size=11.5, row=Inches(0.62)):
        for k, (lab, txt) in enumerate(pairs):
            yy = y + row * k
            self.text(s, lab, LM, yy, label_w, row, size=size, font=SERIF, bold=True, color=INK)
            self.text(s, txt, LM + label_w + Inches(0.2), yy, Inches(12.1) - label_w - Inches(0.2), row, size=size - 0.5)

    def save(self, path):
        self.p.save(path)


# ------------------------------------------------------------------ tables as booktabs images

def table_png(name, header, rows, col_align=None, bold_cells=(), group_rows=(), widths=None, fontsize=9.5):
    """Booktabs-style table rendered with matplotlib (Times-like serif), transparent background."""
    ncol = len(header)
    widths = widths or [1.0] * ncol
    tot = sum(widths)
    fig_w = 0.95 * tot
    row_h = 0.24
    nrows = len(rows) + 1
    fig, ax = plt.subplots(figsize=(fig_w, row_h * (nrows + 0.6)))
    ax.set_axis_off()
    ax.set_xlim(0, tot)
    ax.set_ylim(nrows + 0.3, -0.3)
    xs = [sum(widths[:i]) for i in range(ncol)]
    col_align = col_align or ["left"] + ["right"] * (ncol - 1)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times", "Nimbus Roman", "DejaVu Serif"]

    def put(i, j, txt, bold=False, italic=False):
        a = col_align[j]
        x = xs[j] + (0.05 if a == "left" else widths[j] - 0.05 if a == "right" else widths[j] / 2)
        ax.text(x, i, txt, ha={"left": "left", "right": "right", "center": "center"}[a], va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal", style="italic" if italic else "normal",
                color="#111111")

    for j, h in enumerate(header):
        put(0, j, h, bold=True)
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

    # 1. title
    s = d._slide(dark=True)
    d.text(s, "MID EVALUATION  ·  PROJECT PROGRESS", LM, Inches(1.55), Inches(12), Inches(0.3), size=8,
           color="8fb5a0", bold=True, spacing=200)
    d.text(s, "Order, Duration, or Melody?", LM, Inches(2.0), Inches(12), Inches(0.9), size=40, font=SERIF,
           bold=True, color="ffffff")
    d.text(s, "Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech", LM, Inches(2.9),
           Inches(12), Inches(0.5), size=17, font=SERIF, italic=True, color=LIGHT)
    ln = s.shapes.add_connector(1, LM, Inches(3.75), LM + Inches(2.4), Inches(3.75))
    ln.line.color.rgb = rgb("5c8f72")
    d.text(s, "Mohit Kumar Singh and Nidhi Vaidya", LM, Inches(4.05), Inches(12), Inches(0.3), size=12, color=LIGHT)
    d.text(s, "Base paper: PRiSM, Bharadwaj et al., ACL 2026 (GEO-v, Vaani-Hi dialect geolocation)", LM,
           Inches(4.4), Inches(12), Inches(0.3), size=12, color=LIGHT)
    d.text(s, "3 November 2026", LM, Inches(5.6), Inches(12), Inches(0.3), size=10, color="9fb8aa")

    # 2. problem
    s = d._slide()
    d.header(s, "Problem", "PRiSM's transcript probe beats its representation probe on GEO-v")
    d.bullets(s, [
        "GEO-v asks a probe to place a Hindi-belt utterance on the map (Vaani-Hi, 26 pincodes over ~1,500 km). "
        "PRiSM reports 146 km mean error for a probe over predicted phone strings and 253 km for a probe over "
        "the recognizer's hidden states.",
        "Only the representation probe hears the voice, intonation and duration, so the result says that phone "
        "strings alone carry more usable location signal, or that the comparison is unfair.",
        "The authors suggest the probes differ in more than input: the transcript probe is an RNN that sees "
        "order, the representation probe pools frames without order. This was not tested.",
        "We test what the phone string actually carries (order, duration, topic, speaker) and whether the "
        "evaluation protocol lets either probe succeed without dialect information.",
    ], LM, Inches(1.75), Inches(12.1), Inches(4.5))

    # 3. objectives
    s = d._slide()
    d.header(s, "Objectives", "Four questions, each tied to a measurable property of speech")
    d.deflist(s, [
        ("RQ0  Speakers", "Does PRiSM's random split let probes recognise speakers instead of dialects? Speaker-"
                          "disjoint split, speaker-only baselines, control task."),
        ("RQ1  Order", "Does local phone order carry location beyond the phone inventory? Unigram vs bigram vs "
                       "BiGRU vs shuffled BiGRU, chars vs phones, two recognizers, MDL."),
        ("RQ2  Duration", "Do recognizers encode Haryanvi gemination (e.g., kullar for kūlar)? Doubled-phone "
                          "rate, then acoustic closure durations in Haryana vs New Delhi."),
        ("RQ3  Melody", "Does intonation, discarded by transcripts, add location signal? Prosody probe and "
                        "late fusion with the transcript probe."),
    ], Inches(1.8), label_w=Inches(2.3), size=13, row=Inches(1.0))

    # 4. data
    s = d._slide()
    d.header(s, "Data", "We rebuilt Vaani-Hi from the public corpus with PRiSM's own recipe")
    d.stats(s, [("22,551", "utterances, 40.3 h"), ("26", "pincodes in 9 states"), ("1,570", "speakers"),
                ("68%", "describe a picture of the\nspeaker's own district")], Inches(1.9))
    d.bullets(s, [
        "Access to the official Vaani-Hi copy is still pending, so we reran PRiSM's selection code on the public "
        "Vaani corpus (commit pinned). Ada rebuilds it independently as a checksum test.",
        "Every Vaani file name carries a speaker code; utterances per speaker are skewed (median 3, max 309).",
        "Speakers describe pictures; two thirds of the pictures are named after the speaker's own district, so "
        "the topic of an utterance can reveal its location without any dialect feature.",
    ], LM, Inches(3.55), Inches(12.1), Inches(3))

    # 5. RQ0 leakage stats
    lk = ctrl.loc["dense_ecapa_rebuild_prism"] if ctrl is not None and "dense_ecapa_rebuild_prism" in ctrl.index else None
    s = d._slide()
    d.header(s, "Results  ·  RQ0 speaker leakage", "Under PRiSM's split, the speaker ID alone solves the task")
    sl = bl("speaker_lookup", "prism")
    d.stats(s, [("97.9%", "test utterances whose speaker\nis also in training (random split)"),
                (f"{sl.recall1:.1f}", "Recall@1 of a lookup that only\nknows the speaker ID"),
                (f"{cfg.loc['dense_ecapa_rebuild_prism', 'recall1_mean']:.1f} → "
                 f"{cfg.loc['dense_ecapa_rebuild_speaker_split', 'recall1_mean']:.1f}",
                 "speaker-embedding probe (ECAPA),\nrandom → speaker-disjoint split"),
                (f"{lk.control_recall1:.1f}" if lk is not None else "–",
                 "ECAPA on control labels (random\nother speaker's location)")], Inches(1.9), best=1)
    d.bullets(s, [
        f"Majority baseline: {bl('majority', 'prism').recall1:.1f} (random) and "
        f"{bl('majority', 'speaker_split').recall1:.1f} (speaker-disjoint). Our speaker-disjoint split assigns whole "
        "speakers to train, validation or test, stratified by pincode; 0% of test speakers are seen in training.",
        "A voice embedding trained to ignore content reaches 41.6 Recall@1 on the random split and falls to the "
        "majority level on new speakers. With labels that carry speaker identity but no dialect it still scores "
        f"{lk.control_recall1:.1f} (control majority {lk.control_majority:.1f})." if lk is not None else
        "A voice embedding trained to ignore content falls to the majority level on new speakers.",
        "Transcripts leak too: a nearest-neighbour matcher on character n-grams reaches "
        f"{knn.loc[('w2v2', 'prism', 1), 'recall1']:.1f} Recall@1 on the random split and "
        f"{knn.loc[('w2v2', 'speaker_split', 1), 'recall1']:.1f} on new speakers; for 25% of test utterances the "
        "closest training transcript is the same speaker's (chance 0.3%).",
        "All further results are reported under both splits; the speaker-disjoint numbers measure dialect.",
    ], LM, Inches(3.55), Inches(12.1), Inches(3))

    # 6. results table
    rows, bold = [], []
    order = [("Majority baseline", None), ("Speaker lookup (ID only)", None),
             ("ECAPA speaker embedding", "dense_ecapa_rebuild"), ("Prosody statistics (pooled)", "dense_prosody_rebuild")]
    tp = [("W2V2", "w2v2"), ("ZIPA", "zipa")]
    for name, key in order:
        if key is None:
            b = "majority" if name.startswith("Majority") else "speaker_lookup"
            rows.append([name, fmt(bl(b, "prism").recall1), f"{bl(b, 'prism').mean_km:.0f}",
                         fmt(bl(b, "speaker_split").recall1), f"{bl(b, 'speaker_split').mean_km:.0f}"])
        else:
            a, b = cfg.loc[f"{key}_prism"], cfg.loc[f"{key}_speaker_split"]
            rows.append([name, fmt(a.recall1_mean, a.recall1_sd), f"{a.ens_km:.0f}", fmt(b.recall1_mean, b.recall1_sd),
                         f"{b.ens_km:.0f}"])
    groups = []
    for rec, m in tp:
        groups.append(len(rows))
        rows.append([f"Transcript probe, {rec}", "", "", "", ""])
        for lab, k in [("char unigram", "char_unigram"), ("phone unigram", "phone_unigram"),
                       ("char bigram", "char_bigram"), ("phone bigram", "phone_bigram"),
                       ("char BiGRU (PRiSM)", "char_gru"), ("phone BiGRU", "phone_gru"),
                       ("shuffled phone BiGRU", "phone_shuffle")]:
            c = f"tp_{m}_rebuild_{k}"
            if not (has(f"{c}_prism") and has(f"{c}_speaker_split")):
                continue
            a, b = cfg.loc[f"{c}_prism"], cfg.loc[f"{c}_speaker_split"]
            rows.append(["   " + lab, fmt(a.recall1_mean, a.recall1_sd), f"{a.ens_km:.0f}",
                         fmt(b.recall1_mean, b.recall1_sd), f"{b.ens_km:.0f}"])
    spk_vals = [(i, float(r[3].split(" ")[0])) for i, r in enumerate(rows) if r[3] and i > 1]
    bold.append((max(spk_vals, key=lambda t: t[1])[0], 3))
    png = table_png("table_main", ["Probe", "R@1 random", "km", "R@1 spk-disjoint", "km"], rows, bold_cells=bold,
                    group_rows=groups, widths=[2.4, 1.15, 0.6, 1.35, 0.6], fontsize=9)
    s = d._slide()
    d.header(s, "Results  ·  baselines and transcript probes", "Every probe drops on new speakers; phone bigrams keep the most")
    pic = d.image(s, png, LM, Inches(1.55), h=Inches(5.0))
    tx = LM + pic.width + Inches(0.4)
    cmp_ = comp.set_index(["a", "b"])

    def cd(a, b):
        r = cmp_.loc[(a, b)]
        return f"{r.recall1_diff:+.1f} [{r.recall1_lo:.1f}, {r.recall1_hi:.1f}]"
    d.bullets(s, [
        "Recall@1 as mean ± sd over 5 seeds; km is the mean error of the seed ensemble.",
        "ECAPA loses 31 points between splits, the bag probes 3-13: transcripts carry location that is not "
        "only the voice.",
        "On new speakers, ZIPA phone bigrams beat W2V2 phone bigrams by "
        f"{cd('tp_zipa_rebuild_phone_bigram_speaker_split', 'tp_w2v2_rebuild_phone_bigram_speaker_split')} R@1 "
        "and 33 km (paired, speaker bootstrap).",
    ], tx, Inches(1.7), W - tx - Inches(0.6), Inches(4.5), size=12)
    d.caption(s, "Table 1. Test Recall@1 (higher is better) and mean km (lower is better) on rebuilt Vaani-Hi.",
              LM, Inches(6.75), pic.width, align=PP_ALIGN.LEFT)

    # 7. figure: lift
    s = d._slide()
    d.header(s, "Results  ·  common reference", "Only bigrams beat the majority baseline on new speakers")
    pic = d.image(s, RES / "figures" / "fig_probes.png", LM, Inches(1.5), w=Inches(8.3))
    tx = LM + Inches(8.6)
    d.bullets(s, [
        "Left: Recall@1 minus the majority baseline on the same test utterances. Right: km saved over the best "
        "single fixed location (601-607 km).",
        f"Speaker-disjoint lifts: ZIPA phone bigram {cfg.loc['tp_zipa_rebuild_phone_bigram_speaker_split', 'lift_recall1']:+.1f}, "
        f"W2V2 phone bigram {cfg.loc['tp_w2v2_rebuild_phone_bigram_speaker_split', 'lift_recall1']:+.1f}, ECAPA "
        f"{cfg.loc['dense_ecapa_rebuild_speaker_split', 'lift_recall1']:+.1f} points.",
        "Every probe except pooled prosody saves 134-204 km on new speakers, so coarse regional information "
        "survives even where Recall@1 does not.",
    ], tx, Inches(1.7), W - tx - Inches(0.6), Inches(4.8), size=12)
    d.caption(s, "Figure 1. Seed ensemble, 95% speaker-bootstrap CI. Open markers: trained and tested on shared pictures.",
              LM, Inches(6.75), Inches(8.3))

    # 8. topic
    g = cfg.loc["tp_w2v2_rebuild_phone_bigram_speaker_split_generic"]
    st = state.loc["tp_zipa_rebuild_phone_bigram_speaker_split"]
    s = d._slide()
    d.header(s, "Analysis  ·  what the transcript probe reads", "On new speakers, most of the transcript signal is topic")
    d.stats(s, [(f"{st.all_state_acc:.0f}%", f"correct state, ZIPA phone bigrams,\nnew speakers (majority {st.all_majority:.0f}%)"),
                (f"{g.lift_recall1:+.1f}", "Recall@1 lift when trained and tested\non shared pictures only"),
                (f"+{g.km_gain:.0f} km", f"km gain on shared pictures\n(95% CI {g.km_gain_lo:.0f} to {g.km_gain_hi:.0f})")],
            Inches(1.9))
    d.bullets(s, [
        "Shared pictures remove topic as a cue. Their share differs by state (12% in Madhya Pradesh, 58% in "
        "Chhattisgarh), so they get their own baselines: majority 21.3 Recall@1, best constant 537 km.",
        "Trained and tested on shared pictures, no transcript probe beats the majority on Recall@1 or state; km "
        "gains are 48-94 km and only the two W2V2 ones have CIs above zero, just (119 test speakers).",
        "So the speaker-disjoint signal of bag probes is mostly what people describe; what survives without topic "
        "is at most regional. This is prediction P1 from our pre-registered list.",
    ], LM, Inches(3.55), Inches(12.1), Inches(3))

    # 9. RQ1
    s = d._slide()
    d.header(s, "Results  ·  RQ1 order and units", "Local phone order adds signal, and richer phone strings help")
    k = lambda c: mdl.loc[c, "gain_over_prior"]  # noqa: E731
    d.bullets(s, [
        f"Bigram over unigram (W2V2 phones, new speakers): {cd('tp_w2v2_rebuild_phone_bigram_speaker_split', 'tp_w2v2_rebuild_phone_unigram_speaker_split')} "
        "Recall@1 and 33 km [15, 51] less error. Local order matters even without a sequence model.",
        f"Phones over characters: {cd('tp_w2v2_rebuild_phone_bigram_speaker_split', 'tp_w2v2_rebuild_char_bigram_speaker_split')} "
        "Recall@1 for W2V2. PRiSM tokenises by character, which splits kʰ into k + ʰ.",
        "MDL (online coding of the pincode, speaker-ordered blocks): bits saved over the location prior on new "
        f"speakers. ZIPA phone bigram {k('mdl_zipa_rebuild_phone_bigram_speaker_split'):.1f} kbits, W2V2 phone bigram "
        f"{k('mdl_w2v2_rebuild_phone_bigram_speaker_split'):.1f}, unigrams 5.0-6.0. On the random split the same "
        f"probes save {k('mdl_zipa_rebuild_phone_bigram_prism'):.1f} kbits, three times more.",
        "ZIPA beats W2V2 on all three MDL orderings (1.2-1.5 kbits); phone units help ZIPA (0.7 kbits) but not "
        "W2V2 (0.0), matching ZIPA's richer diacritic output.",
    ], LM, Inches(1.75), Inches(12.1), Inches(4.8))

    # 9b. RQ1 sequence model
    pairs = {(m, P): (f"tp_{m}_rebuild_phone_gru_{P}", f"tp_{m}_rebuild_phone_shuffle_{P}")
             for m in ("w2v2", "zipa") for P in ("prism", "speaker_split")}
    if all(v in cmp_.index for v in pairs.values()):
        g = {k: cmp_.loc[v] for k, v in pairs.items()}
        nseed = int(min(cfg.loc[c, "seeds"] for c in cfg.index if ("_gru_" in c or "_shuffle_" in c) and "control" not in c))
        s = d._slide()
        d.header(s, "Results  ·  RQ1 sequence model", "Order helps the BiGRU on PRiSM's split; on new speakers it is unclear")
        lab = {"prism": "random split", "speaker_split": "new speakers"}
        d.stats(s, [(f"{g[k].recall1_diff:+.1f}", f"{k[0].upper()}, {lab[k[1]]}\n(95% CI {g[k].recall1_lo:.1f} to {g[k].recall1_hi:.1f})")
                    for k in [("w2v2", "prism"), ("zipa", "prism"), ("w2v2", "speaker_split"), ("zipa", "speaker_split")]],
                Inches(1.9))
        d.bullets(s, [
            "Recall@1 of PRiSM's transcript probe (2-layer BiGRU over phones) minus the same probe on phones shuffled "
            "within each utterance, which keeps the inventory and removes order. Paired on the same test utterances.",
            f"On the random split order helps with both recognizers ({g[('w2v2', 'prism')].recall1_diff:+.1f} and "
            f"{g[('zipa', 'prism')].recall1_diff:+.1f} points, {-g[('w2v2', 'prism')].km_diff:.0f} and "
            f"{-g[('zipa', 'prism')].km_diff:.0f} km). On new speakers neither gains on Recall@1 "
            f"({g[('w2v2', 'speaker_split')].recall1_diff:+.1f} for W2V2, {g[('zipa', 'speaker_split')].recall1_diff:+.1f} for ZIPA), "
            "so part of the random-split gain may be memorised speaker sequences; the remaining seeds must settle it.",
            "The BiGRU stays below a bag of bigrams on both splits, and far below PRiSM's 45.7 on the official data, "
            f"which we cannot access yet. Preliminary: {nseed}-2 seeds per condition so far; 5 by the final.",
        ], LM, Inches(3.55), Inches(12.1), Inches(3))

    # 10. RQ2 DPR
    s = d._slide()
    d.header(s, "Results  ·  RQ2 duration", "Doubled phones in recognizer output depend on the recognizer")
    d.image(s, RES / "figures" / "fig_dpr_w2v2.png", LM, Inches(1.55), w=Inches(5.6))
    d.image(s, RES / "figures" / "fig_dpr_zipa.png", LM + Inches(6.2), Inches(1.55), w=Inches(5.6))
    hv_w = pd.read_csv(RES / "dpr" / "w2v2_haryana_vs_rest.csv").iloc[0]
    hv_z = pd.read_csv(RES / "dpr" / "zipa_haryana_vs_rest.csv").iloc[0]
    d.bullets(s, [
        f"W2V2: Haryana doubles least, {100 * hv_w.haryana:.2f}% of consonants vs {100 * hv_w.rest:.2f}% elsewhere "
        f"(p = {hv_w.p:.3f}). ZIPA: {100 * hv_z.haryana:.2f}% vs {100 * hv_z.rest:.2f}% (p = {hv_z.p:.2f}). Neither shows "
        "the Haryanvi doubling PRiSM's attribution example suggests; Haryana has only 12 speakers.",
        "So the recognizers' output cannot settle gemination; next slide measures it in the audio.",
    ], LM, Inches(5.35), Inches(12.1), Inches(1.6), size=12)

    # 10b. RQ2 acoustics
    mm = (RES / "gemination" / "mixed_model.txt").read_text()
    if "voiceless interval (primary)" in mm:
        import re as _re

        def eff(label):
            m_ = _re.search(_re.escape(label) + r".*?= x([\d.]+)[^(]*\(95% CI x([\d.]+) to x([\d.]+)\).*?tokens (\d+)", mm)
            return float(m_[1]), float(m_[2]), float(m_[3]), int(m_[4])
        voi, clo, val = eff("voiceless interval (primary)"), eff("closure, tokens with a burst"), \
            eff("validity: lexical geminate voiceless interval")
        lee = _re.search(r"Lee bounds x([\d.]+) to x([\d.]+)", mm)
        acc = _re.search(r"acceptance \(main contexts\): Haryana ([\d.]+), Delhi ([\d.]+)", mm)
        s = d._slide()
        d.header(s, "Results  ·  RQ2 duration, acoustics", "In the audio, Haryana speakers hold voiceless stops 14% longer")
        d.stats(s, [(f"x{voi[0]:.2f}", f"voiceless interval, Haryana vs New Delhi\n(95% CI {voi[1]:.2f} to {voi[2]:.2f})"),
                    (f"x{clo[0]:.2f}", f"closure alone\n(95% CI {clo[1]:.2f} to {clo[2]:.2f})"),
                    (f"x{val[0]:.2f}", f"lexical geminates vs single stops\n(validity check, 95% CI {val[1]:.2f} to {val[2]:.2f})")],
                Inches(1.9), best=0)
        d.bullets(s, [
            f"{voi[3]:,} single voiceless stops between vowels in Rohtak, Jhajjar, Charkhi Dadri and New Delhi Vaani "
            "audio, located by MMS forced alignment of the standard spelling (independent of the phone recognizers) "
            "and measured with Praat voicing and burst landmarks.",
            "Holds in each Haryana district alone (x1.13 to x1.15), in 45 of 57 words, and on speaker means without "
            "a model (x1.14).",
            "Mixed model with word fixed effects, following-vowel duration as rate control and a random intercept per "
            f"speaker. Measured share {100 * float(acc[1]):.0f}% (Haryana) vs {100 * float(acc[2]):.0f}% (New Delhi); "
            f"Lee bounds for that difference x{float(lee[1]):.2f} to x{float(lee[2]):.2f}.",
            "Lexical geminates (अच्छा, बच्चे, मिट्टी) come out much longer, so the measure tracks length. Haryana single "
            "stops move about a quarter of the way toward geminate length: lengthening, not categorical doubling. "
            "A blind manual audit of 50 tokens by both authors is the next check.",
        ], LM, Inches(3.55), Inches(12.1), Inches(3), size=12.5)

    # 10c. RQ2 layers (when the layer-wise geminate probe has run)
    lay = RES / "gemination" / "geminate_layers_w2v2.csv"
    if lay.exists():
        L = pd.read_csv(lay)
        best = L.loc[L["auc_within"].idxmax()]
        sig = L[L["haryana_lo"] > 0]
        s = d._slide()
        d.header(s, "Results  ·  RQ2 duration, layers",
                 f"W2V2 represents consonant length best around layer {int(best.layer)}")
        pic = d.image(s, RES / "figures" / "fig_geminate_layers_w2v2.png", LM, Inches(1.5), w=Inches(7.6))
        tx = LM + Inches(7.9)
        d.bullets(s, [
            f"A linear probe per layer separates lexical geminates from single stops (speaker-grouped "
            f"cross-validation, {int(best.tokens):,} tokens). Best within place of articulation: layer "
            f"{int(best.layer)}, AUC {best.auc_within:.2f} (all tokens {best.auc:.2f} [{best.auc_lo:.2f}, {best.auc_hi:.2f}]).",
            f"Haryana single stops look more geminate-like than New Delhi ones in {len(sig)} of {len(L)} layers "
            f"(at layer {int(best.layer)}: {best.haryana_logit:+.2f} logit [{best.haryana_lo:+.2f}, {best.haryana_hi:+.2f}]).",
            ("So the encoder carries the lengthening measured in the audio; ZIPA and Whisper follow on Ada."
             if len(sig) else "So the encoder separates lexical geminates, but the Haryana lengthening of single stops "
             "is not visible to it; ZIPA and Whisper follow on Ada."),
        ], tx, Inches(1.7), W - tx - Inches(0.6), Inches(4.8), size=12)
        d.caption(s, "Figure. Left: probe AUC by layer, all tokens and within place. Right: Haryana shift for single stops.",
                  LM, Inches(6.75), Inches(7.6))

    # 11. next steps
    s = d._slide()
    d.header(s, "Plan", "What remains before the final evaluation")
    d.deflist(s, [
        ("Running now", "TP-BiGRU over chars, phones and shuffled phones (2 recognizers x 2 splits; PC seeds 42-43, "
                        "Ada seeds 44-46); BiGRU control task and MDL; prosody contours."),
        ("On Ada", "Feature extraction (W2V2, Whisper, ZIPA), layer sweep, representation probes with and without "
                   "order at matched size, transcription check."),
        ("RQ2", "Blind manual audit of 50 tokens by both authors; does recognizer doubling track measured length; "
                "which encoder layers represent consonant length (geminate probe by layer, Ada)."),
        ("RQ3", "Prosody BiGRU on F0 and intensity contours; late fusion with the transcript probe."),
        ("Write-up", "ACL-format report with all tables generated from results/ by script."),
    ], Inches(1.8), label_w=Inches(2.0), size=13, row=Inches(0.85))

    # 12. takeaways
    s = d._slide(dark=True)
    d.header(s, "Conclusion", "Takeaways", dark=True)
    d.bullets(s, [
        "PRiSM's random split leaks speakers: a lookup on the speaker ID reaches 98 Recall@1, and a voice "
        "embedding drops from 41.6 to the majority level once speakers are disjoint.",
        "On new speakers, phone bigrams keep a real but small advantage (+8 to +11 points over majority), and "
        "most of it comes from what speakers describe rather than how they pronounce.",
        "Local order (bigrams) and richer phone strings help on new speakers, by Recall@1, km and MDL alike; the "
        "BiGRU gains from order on PRiSM's split, while on new speakers the recognizers disagree (preliminary).",
        "Doubling in recognizer output is recognizer-specific, but the audio shows it: Haryana speakers hold "
        "voiceless stops 14% longer (closure 26%) than New Delhi speakers.",
    ], LM, Inches(1.8), Inches(12.1), Inches(4.5), size=14, color=LIGHT)
    d.text(s, "Code, data recipe and every number: github.com/unignoramus11/sal-project", LM, Inches(6.6),
           Inches(12), Inches(0.3), size=10, italic=True, color="9fb8aa")

    # 13. appendix: setup
    s = d._slide()
    d.header(s, "Appendix  ·  reference", "Setup, metrics and limitations")
    d.deflist(s, [
        ("Metrics", "Recall@1 (nearest distinct location of the test split, PRiSM definition), mean great-circle "
                    "km, lift over majority and km gain over the best constant location on the same utterances."),
        ("Statistics", "5 seeds; 95% CIs and paired tests by bootstrap over speakers (1,000-2,000 resamples), "
                       "because utterances of one speaker are not independent."),
        ("Probes", "PRiSM's TP-GRU (2-layer BiGRU 256, char tokens) and its settings; bag MLPs over unigram or "
                   "bigram frequencies; ECAPA-TDNN embeddings with an MLP; angular loss throughout."),
        ("Recognizers", "W2V2 (wav2vec2-lv-60-espeak-cv-ft) and ZIPA-CR-CTC-NS large, greedy CTC, run with PRiSM's "
                        "own wrappers (output checked against PRiSM's runner)."),
        ("Limitations", "Rebuilt rather than official Vaani-Hi; 12 Haryana speakers; picture prompts tie topic to "
                        "place; recognizer output is not a phonetic gold standard."),
    ], Inches(1.75), label_w=Inches(1.9), size=11.5, row=Inches(0.8))
    d.text(s, "Bharadwaj et al. (2026). PRiSM: Benchmarking Phone Realization in Speech Models. ACL 2026, pages "
              "18093-18112.", LM, Inches(6.75), Inches(12), Inches(0.3), size=8.5, italic=True, color=CAP)
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
