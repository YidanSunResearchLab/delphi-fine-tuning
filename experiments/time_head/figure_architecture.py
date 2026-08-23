"""
figure_architecture.py -- the model as it stands now, in the style of Delphi-2M Fig. 1c.

    python experiments/time_head/figure_architecture.py
    -> results/architecture.png (300 dpi) + results/architecture.pdf (vector)

Two colour codes, deliberately kept apart:
  salmon  = new compared to GPT-2          (Delphi-2M's own additions -- same meaning as the
                                            paper figure's legend)
  purple  = new in THIS repo, opt-in       (time_head=False reproduces every delivered
                                            checkpoint bit-for-bit)

Every label is traceable to code: box text names the module, the right-hand panel gives the
exact expressions from Delphi.forward(). If model.py changes, this script is the thing to
update -- it is drawn, not photographed, so it cannot silently go stale in a way a reader
can check against the code.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# ---------------------------------------------------------------- palette (Fig. 1c's own)
C = dict(
    salmon=("#F7A9A0", "#B03A2E"),      # new vs GPT-2
    purple=("#C9A8DC", "#5B2D8E"),      # new in this repo (opt-in)
    green=("#C7E5D0", "#1E7B4F"),       # the categorical head
    red=("#F5A0A0", "#B03A2E"),         # the time head's output
    blue=("#A9B8DC", "#39497F"),        # plain Linear / feed forward
    yellow=("#EEEEBE", "#8A8A3C"),      # LayerNorm
    orange=("#F7CFA0", "#B5762A"),      # attention
    pink=("#F8D7DF", "#B3667C"),        # token embedding
    plain=("#FFFFFF", "#333333"),
    grey=("#F0F0F0", "#B0B0B0"),
)

# Coordinates are TENTHS OF AN INCH in both axes, so rounded corners come out circular
# without any mutation_aspect fiddling and nothing needs re-tuning if the figure is resized.
FIG_W, FIG_H = 11.6, 10.6
LX, LY = 58.0, 102.0          # left panel extent
RX, RY = 52.0, 102.0          # right panel extent


def box(ax, xc, yc, w, h, text, color="plain", fs=7.2, weight="normal", lw=1.1,
        style="round", ls="-", zorder=3):
    fc, ec = C[color]
    r = 0.9 if style == "round" else 0.0
    p = FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, weight=weight,
            color="#111111", zorder=zorder + 1, linespacing=1.35)
    return dict(x=xc, y=yc, w=w, h=h, top=yc + h / 2, bot=yc - h / 2,
                left=xc - w / 2, right=xc + w / 2)


def line(ax, pts, color="#333333", lw=1.2, ls="-", arrow=True, zorder=2):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, zorder=zorder,
            solid_capstyle="round")
    if arrow:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=lw, linestyle=ls, zorder=zorder,
                                     shrinkA=0, shrinkB=0))


def plus(ax, x, y, r=0.95):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="#333333", lw=1.1, zorder=4))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color="#333333", lw=1.0, zorder=5)
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color="#333333", lw=1.0, zorder=5)
    return dict(x=x, y=y, top=y + r, bot=y - r, left=x - r, right=x + r)


# ============================================================================ left: the stack
def draw_stack(ax):
    ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.axis("off")
    xc = 24.0
    W = 17.0

    ax.text(0.5, 100.0, "a", fontsize=15, weight="bold", va="top")

    # ---- legend (top-left) --------------------------------------------------
    ax.add_patch(Rectangle((0.4, 84.0), 11.2, 14.0, facecolor="white",
                           edgecolor="#CCCCCC", lw=0.9, zorder=1))
    box(ax, 3.4, 95.4, 4.0, 2.6, "", "salmon", lw=1.0)
    ax.text(6.0, 95.4, "new vs\nGPT-2", fontsize=6.2, va="center", linespacing=1.3)
    box(ax, 3.4, 90.0, 4.0, 2.6, "", "purple", lw=1.0)
    ax.text(6.0, 90.0, "new in this\nrepo (opt-in)", fontsize=6.2, va="center", linespacing=1.3)
    ax.text(6.0, 86.0, "dashed = weight tying", fontsize=5.8, ha="center", color="#555555")

    # ---- inputs ------------------------------------------------------------
    ax.text(xc, 0.8, "Times (age in days)  ·  tokens in model space 0–110",
            ha="center", fontsize=6.0, color="#555555")
    b_other = box(ax, 17.0, 4.2, 12.5, 4.6, "Other information\n(sex, lifestyle, APOE)",
                  "plain", fs=6.1)
    b_dis = box(ax, 31.0, 4.2, 11.5, 4.6, "Disease events\n+ No-event", "plain", fs=6.1)

    # ---- token embedding ---------------------------------------------------
    b_tok = box(ax, xc, 11.0, W, 5.0, "Token embedding\n111 × 120", "pink", fs=7.0)
    line(ax, [(b_other["x"], b_other["top"]), (b_other["x"], 8.0), (xc - 3.0, 8.0),
              (xc - 3.0, b_tok["bot"])])
    line(ax, [(b_dis["x"], b_dis["top"]), (b_dis["x"], 8.0), (xc + 3.0, 8.0),
              (xc + 3.0, b_tok["bot"])])

    # ---- age encoding ⊕ ----------------------------------------------------
    p1 = plus(ax, xc, 17.8)
    line(ax, [(xc, b_tok["top"]), (xc, p1["bot"])])
    b_age = box(ax, 10.6, 17.8, 11.0, 6.4, "", "salmon")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(10.6 + np.linspace(-3.1, 3.1, 200), 18.9 + 1.0 * np.sin(t),
            color="#7B241C", lw=1.0, zorder=5)
    ax.text(10.6, 15.9, "Age encoding\nsin/cos → Linear", ha="center", va="center",
            fontsize=5.9, zorder=5, linespacing=1.3)
    line(ax, [(b_age["right"], 17.8), (p1["left"], 17.8)])

    # ---- the N× block ------------------------------------------------------
    ax.add_patch(FancyBboxPatch((13.2, 22.4), 21.6, 44.4,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=C["grey"][0], edgecolor=C["grey"][1], lw=1.0,
                                zorder=1))
    ax.text(10.9, 44.6, "N ×\n8", ha="center", va="center", fontsize=8.5, style="italic")

    b_ln1 = box(ax, xc, 25.6, 15.0, 4.0, "LayerNorm", "yellow")
    b_msk = box(ax, xc, 31.6, 15.0, 5.6, "Causal continuous\ntime mask", "salmon", fs=6.8)
    b_att = box(ax, xc, 38.8, 15.0, 6.4, "Multi-head causal\nself-attention", "orange", fs=6.8)
    p2 = plus(ax, xc, 45.4)
    b_ln2 = box(ax, xc, 50.4, 15.0, 4.0, "LayerNorm", "yellow")
    b_ff = box(ax, xc, 56.8, 15.0, 6.4, "Feed forward\n120 → 480 → 120", "blue", fs=6.8)
    p3 = plus(ax, xc, 63.6)

    line(ax, [(xc, p1["top"]), (xc, b_ln1["bot"])])
    line(ax, [(xc, b_ln1["top"]), (xc, b_msk["bot"])], arrow=False)
    line(ax, [(xc, b_msk["top"]), (xc, b_att["bot"])], arrow=False)
    line(ax, [(xc, b_att["top"]), (xc, p2["bot"])])
    line(ax, [(xc, p2["top"]), (xc, b_ln2["bot"])])
    line(ax, [(xc, b_ln2["top"]), (xc, b_ff["bot"])], arrow=False)
    line(ax, [(xc, b_ff["top"]), (xc, p3["bot"])])
    # the two residual bypasses
    for y0, y1, node in ((23.6, 45.4, p2), (48.0, 63.6, p3)):
        line(ax, [(xc, y0), (xc - 9.4, y0), (xc - 9.4, y1), (node["left"], y1)],
             color="#555555", lw=1.0)

    # what builds the mask -- it is made once, outside the blocks, and shared
    ax.annotate("causal ∧ not-padding\n∧ same-visit (mask_ties)\nbuilt once from the ages,\n"
                "shared by all 8 blocks",
                xy=(b_msk["right"], 31.6), xytext=(37.0, 31.6),
                fontsize=5.7, va="center", ha="left", color="#7B241C", linespacing=1.35,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#B03A2E",
                                shrinkA=1, shrinkB=1))

    # ---- final norm + the tied projection ----------------------------------
    b_lnf = box(ax, xc, 70.2, W, 4.0, "LayerNorm", "yellow")
    b_lin = box(ax, xc, 77.0, W, 5.0, "Linear  120 → 111", "blue", fs=7.0)
    line(ax, [(xc, p3["top"]), (xc, b_lnf["bot"])])
    line(ax, [(xc, b_lnf["top"]), (xc, b_lin["bot"])])
    ax.text(25.4, 73.4, "hidden state  x", ha="left", va="center", fontsize=5.9,
            color="#555555", style="italic")

    # Weight tying. On the LEFT, not the right as in the paper: the time head taps the same
    # hidden state the Linear does, so any right-hand routing crosses that tap. Mirroring it
    # costs nothing and keeps every wire crossing-free.
    line(ax, [(b_lin["left"], 77.0), (2.6, 77.0), (2.6, 11.0), (b_tok["left"], 11.0)],
         color="#555555", lw=1.0, ls=(0, (3, 2)))
    ax.text(4.1, 50.0, "Weight tying", rotation=90, va="center", ha="center", fontsize=7.0,
            color="#555555")

    # ---- THE NEW HEAD ------------------------------------------------------
    b_th = box(ax, 40.0, 82.0, 14.0, 6.6, "Time head\nLinear 120 → 1\n+121 params",
               "purple", fs=6.4)
    line(ax, [(b_lnf["right"], 70.2), (40.0, 70.2), (40.0, b_th["bot"])],
         color=C["purple"][1], lw=1.3)

    # ---- the two outputs ---------------------------------------------------
    b_soft = box(ax, 18.0, 89.4, 13.0, 4.4, "Softmax", "green", fs=7.2)
    b_lam = box(ax, 38.0, 89.4, 14.5, 4.4, "log λ", "salmon", fs=8.0)
    b_next = box(ax, 18.0, 96.4, 15.5, 5.0, "Next disease\nevent", "green", fs=7.2)
    b_time = box(ax, 38.0, 96.4, 16.5, 5.0, "Time to event\nΔt ~ Exp(λ)", "red", fs=7.2)

    line(ax, [(18.0, b_lin["top"]), (18.0, b_soft["bot"])])
    line(ax, [(18.0, b_soft["top"]), (18.0, b_next["bot"])])
    line(ax, [(38.0, b_lam["top"]), (38.0, b_time["bot"])])
    # default: lambda comes from the token logits themselves
    # routed OVER the time head box (y 86 clears its top edge at 85.3) and entering log λ
    # left of it, so the dashed default path cannot be misread as coming out of the new head
    line(ax, [(29.0, b_lin["top"]), (29.0, 86.0), (32.0, 86.0), (32.0, b_lam["bot"])],
         color="#8A6D3B", lw=1.1, ls=(0, (3, 2)))
    ax.text(28.4, 83.4, "logsumexp\n(default)", fontsize=5.5, color="#8A6D3B",
            ha="right", va="center", linespacing=1.3)
    # opt-in: lambda comes from the new head
    line(ax, [(41.5, b_th["top"]), (41.5, b_lam["bot"])], color=C["purple"][1], lw=1.3)
    ax.text(46.0, 86.6, "time_head\n= True", fontsize=5.5, color=C["purple"][1],
            ha="left", va="center", linespacing=1.3)

    ax.text(29.0, 100.6, "one stack, two heads — the switch is which arrow feeds  log λ",
            ha="center", fontsize=6.4, color="#444444", style="italic")


# ============================================================================ right: the maths
def draw_maths(ax):
    ax.set_xlim(0, RX); ax.set_ylim(0, RY); ax.axis("off")
    ax.text(0.0, 100.0, "b", fontsize=15, weight="bold", va="top")

    MONO = dict(family="DejaVu Sans Mono", fontsize=6.5, color="#111111")
    # Line heights in tenths of an inch, matched to what the text actually occupies:
    # fontsize 6.5 pt at linespacing 1.5 is 9.75 pt = 0.135 in = 1.35 units. The first draft
    # used 2.4 here, which both padded every panel and pushed the last two off the figure.
    H = dict(t=1.45, m=1.45, h=2.40, gap=1.10)
    TITLE_H, PAD_TOP, PAD_BOT = 2.60, 1.40, 1.40

    def measure(lines):
        """Panel height from its content. Hand-placed offsets are how the first draft
        overflowed every box; this makes the rectangle follow the text instead."""
        h = TITLE_H + PAD_TOP + PAD_BOT
        for txt, kind in lines:
            h += H[kind] * (txt.count("\n") + 1) if kind in ("t", "m") else H[kind]
        return h

    def panel(y_top, title, lines, edge="#CCCCCC", face="#FFFFFF", tcol="#111111"):
        h = measure(lines)
        y0 = y_top - h
        ax.add_patch(Rectangle((0.0, y0), RX - 0.5, h, facecolor=face, edgecolor=edge,
                               lw=1.0, zorder=1))
        y = y_top - PAD_TOP
        ax.text(1.4, y, title, fontsize=8.0, weight="bold", va="top", color=tcol, zorder=2)
        y -= TITLE_H
        for txt, kind in lines:
            if kind == "gap":
                y -= H["gap"]
            elif kind == "m":
                ax.text(2.2, y, txt, va="top", zorder=2, linespacing=1.5, **MONO)
                y -= H["m"] * (txt.count("\n") + 1)
            elif kind == "h":
                ax.text(1.4, y, txt, fontsize=7.0, weight="bold", va="top", color="#444444",
                        zorder=2)
                y -= H["h"]
            else:
                ax.text(1.4, y, txt, fontsize=6.5, va="top", color="#333333", zorder=2,
                        linespacing=1.5)
                y -= H["t"] * (txt.count("\n") + 1)
        return y0

    GAP = 2.0
    y = 98.4

    y = panel(y, "What the top of the stack computes",
              [("Both objectives are read off the SAME hidden state x. The only question\n"
                "is where the rate λ comes from.", "t"),
               ("", "gap"),
               ("time_head = False   —   every delivered checkpoint", "h"),
               ("logits = W·x                (W tied to the token embedding)\n"
                "log λ  = logsumexp(logits)  ← one scalar, no parameters of its own", "m"),
               ("", "gap"),
               ("time_head = True    —   this repo, opt-in (+121 params)", "h"),
               ("log λ  = w·x + b            ← its own projection\n"
                "logits = log_softmax(W·x) + log λ", "m")]) - GAP

    y = panel(y, "Why that is a drop-in",
              [("A per-position constant is invisible to softmax and exactly recoverable\n"
                "by logsumexp, so nothing downstream changes:", "t"),
               ("", "gap"),
               ("softmax(logits)     unchanged     max |Δ| = 4.7e-09\n"
                "logsumexp(logits) == log λ        max |Δ| = 2.2e-07\n"
                "loss_ce             unchanged     Δ = 0, exactly", "m"),
               ("", "gap"),
               ("generate(), ad_engine, predict_adapter and figure2 pick the head up with\n"
                "no code change, and the gradients separate:\n"
                "∂loss_ce/∂(head) = 2e-07 ≈ 0,   ∂loss_dt/∂(head) = O(20).", "t")]) - GAP

    y = panel(y, "The floor, the losses, the sampling",
              [("t_min floor — both modes, guards λ → ∞:", "h"),
               ("E[Δt] = e^(−log λ) + t_min ,   t_min = 365.25/12 ≈ 30.4 d", "m"),
               ("", "gap"),
               ("Objective — a plain sum of two terms:", "h"),
               ("loss = CE(logits, next token)  +  ( λ·Δt − log λ )\n"
                "       └─ loss_ce ≈ 27% ─┘        └─ loss_dt ≈ 73% ─┘", "m"),
               ("", "gap"),
               ("Sampling in generate() — competing exponentials:", "h"),
               ("Δt_k = −e^(−logit_k)·ln U ,  min over k → (when, which)", "m")]) - GAP

    y = panel(y, "The delivered model, exactly",
              [("8 layers · 6 heads (head_dim 20) · 120 d · vocab 111 · block 96\n"
                "1,412,160 params  (1,412,281 with the time head)\n"
                "bias = False everywhere · dropout 0 · token_dropout 0\n"
                "tokens 0–21 (padding, No-event, sex, BMI, smoking, alcohol, education,\n"
                "APOE) never appear as targets and are excluded from both losses", "t")]) - GAP

    panel(y, "Where this differs from the paper's Fig. 1c",
          [("· the age encoding is sin/cos followed by a LEARNED Linear\n"
            "· the time mask is built once outside the blocks and shared by all 8; its\n"
            "  same-visit (mask_ties) part is active only when targets are passed — in\n"
            "  training and scoring, not in free-running generation\n"
            "· in validation_loss_mode the loss reads log λ directly; logsumexp of the\n"
            "  masked logits would deflate it by the content probability mass\n"
            "· λ no longer has to come from the token logits — the purple box", "t")],
          face="#FBF7FF", edge=C["purple"][1], tcol=C["purple"][1])


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax1 = fig.add_axes([0.005, 0.005, FIG_W and 5.8 / FIG_W, 10.2 / FIG_H])
    ax2 = fig.add_axes([6.05 / FIG_W, 0.005, 5.2 / FIG_W, 10.2 / FIG_H])
    draw_stack(ax1)
    draw_maths(ax2)
    png = os.path.join(OUT, "architecture.png")
    pdf = os.path.join(OUT, "architecture.pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
