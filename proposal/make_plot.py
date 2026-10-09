"""GEO-v Recall@1 from PRiSM (Bharadwaj et al., 2026), Tables 4 and 7."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["MultiIPA", "W2V2P-\nLV60", "W2V2P-\nXLSR53", "ZIPA-\nCTC", "ZIPA-\nCTC-NS",
          "POWSM", "POWSM-\nCTC", "WavLM", "Whisper"]
tp   = [44.5, 45.7, 44.5, 44.1, 44.7, 43.7, 42.9, np.nan, np.nan]   # Table 4, TP
rp   = [24.5, 24.6, 21.5, 23.0, 24.0, 21.5, 14.1, 23.5, 35.0]       # Table 4, RP (last layer)
rpw  = [34.1, 34.9, 35.5, 34.3, 32.8, 32.0, 24.4, 30.8, 24.0]       # Table 7, RP (weighted layers)

# mean over the 7 PR systems that have all three probes
m = lambda v: float(np.mean(v[:7]))
models.append("Mean\n(7 PR)")
tp.append(m(tp)); rp.append(m(rp)); rpw.append(m(rpw))
print("means TP %.1f RP-last %.1f RP-wsum %.1f" % (tp[-1], rp[-1], rpw[-1]))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(6.3, 1.75))
x = np.arange(len(models)); w = 0.26
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
labels = ["Transcript probe (bi-GRU)", "Repr. probe, last layer (attn. pool)", "Repr. probe, weighted layers (attn. pool)"]
for i, (vals, c, lab) in enumerate(zip([tp, rp, rpw], cols, labels)):
    bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=c, label=lab,
                  hatch="////" if i == 2 else None, edgecolor="white", linewidth=0.4)
    v = vals[-1]
    if i == 1:
        ax.text(x[-1] + (i - 1) * w, v - 1.0, f"{v:.1f}", ha="center", va="top", fontsize=5.8,
                color="white", rotation=90)
    else:
        ax.text(x[-1] + (i - 1) * w, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color="#222")
for i in range(7, 9):
    ax.text(x[i] - w, 1.0, "n/a", ha="center", va="bottom", fontsize=6, color="#666", rotation=90)
ax.axvline(x[-1] - 0.55, color="#999", lw=0.5, ls="--")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=6.8)
ax.set_ylabel("GEO-v Recall@1 (%)")
ax.set_ylim(0, 52)
ax.yaxis.grid(True, color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False, fontsize=7,
          handlelength=1.4, columnspacing=1.2)
fig.tight_layout(pad=0.3)
fig.savefig("figs/geov_gap.pdf")
fig.savefig("figs/geov_gap.png", dpi=200)
