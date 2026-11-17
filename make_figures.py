# -*- coding: utf-8 -*-
"""Main figures 1-8 of the manuscript (183 mm width, two-column layout). Run after run_all.py: python make_figures.py"""
import os, sys, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "figures")
os.makedirs(OUT, exist_ok=True)
R6 = R7 = os.path.join(H, "data", "empirical")      # outputs of the empirical scripts in empirics/

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
DOM = {"Wissenschaft": ("Science", BLUE), "Wirtschaft": ("Economy", ORANGE)}
MM = 1 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.dpi": 300})


def style(ax, grid=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def layout(A, seed=0):
    N = A.shape[0]; deg = A.sum(1)
    con, iso = np.flatnonzero(deg > 0), np.flatnonzero(deg == 0)
    pos = np.zeros((N, 2))
    if len(con):
        rng = np.random.default_rng(seed); Ac = A[np.ix_(con, con)].astype(float); n = len(con)
        p = rng.random((n, 2)); k = np.sqrt(1 / n); t = 0.1
        for _ in range(300):
            d = p[:, None] - p[None]; dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
            f = ((k * k / dist ** 2 - Ac * dist / k)[:, :, None] * d).sum(1) - 0.05 * (p - p.mean(0))
            ln = np.linalg.norm(f, axis=1) + 1e-9; p += f / ln[:, None] * np.minimum(ln, t)[:, None]; t *= 0.985
        p = p - np.median(p, 0); rad = np.sqrt((p ** 2).sum(1))
        p = p / (np.percentile(rad, 92) + 1e-9); rad = np.sqrt((p ** 2).sum(1))
        p = p * (np.minimum(rad, 1.05) / np.maximum(rad, 1e-9))[:, None]; pos[con] = p
    if len(iso):
        a = np.linspace(0, 2 * np.pi, len(iso), endpoint=False); pos[iso] = 1.22 * np.c_[np.cos(a), np.sin(a)]
    return pos


def draw_net(ax, A, colors, sizes, edgecol="#c3c2b7"):
    pos = layout(A)
    iu, ju = np.nonzero(np.triu(A, 1))
    for i, j in zip(iu, ju):
        ax.plot(pos[[i, j], 0], pos[[i, j], 1], color=edgecol, lw=0.3, alpha=0.7, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors, edgecolors="white", linewidths=0.25, zorder=2)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")


# =====================================================================
# Figure 1 - model schematic
# =====================================================================
def fig1():
    fig = plt.figure(figsize=(183 * MM, 62 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc="white", ec=col, lw=1.0))
        ax.text(x + w / 2, y + h - 3.5, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=col)
        ax.text(x + w / 2, y + h - 9, body, ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.35)

    def arrow(x1, y1, x2, y2, col=INK2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col))

    ax.text(2, 59, "a", fontsize=9, fontweight="bold"); ax.text(64, 59, "b", fontsize=9, fontweight="bold")
    ax.text(134, 59, "c", fontsize=9, fontweight="bold")
    # a: agent with skills
    box(6, 8, 52, 46, "Agent i", "idea quality $a_i$, holds n of K skills\nmissing skills come from partners\n(cost rises with degree)\noutput = $a_i$ \u00d7 geometric mean\nof skill coverage", INK)
    for k, (cov, lab) in enumerate([(0.9, "own"), (0.9, "own"), (0.05, "gap"), (0.05, "gap"), (0.05, "gap"), (0.05, "gap")]):
        x = 12 + k * 7.0
        ax.add_patch(plt.Rectangle((x, 10), 5, 8, fc="#f0efec", ec=AXIS, lw=0.5))
        ax.add_patch(plt.Rectangle((x, 10), 5, 8 * cov, fc=GREY if cov > 0.5 else "#e1e0d9", ec="none"))
    ax.text(32, 20.5, "own skill coverage (K = 6): 2 strong, 4 gaps", ha="center", fontsize=6, color=INK2)
    # b: three channels
    box(68, 40, 58, 15, "Substitution  α", "AI fills a share of each skill gap\n→ partners become less valuable", BLUE)
    box(68, 23, 58, 15, "Amplification  λ, η", "AI multiplies output, more so for\ngood ideas when η > 0", ORANGE)
    box(68, 6, 58, 15, "Coordination  κ", "AI lowers the cost of keeping a\npartner → links become cheaper", AQUA)
    arrow(58, 31, 66, 47); arrow(58, 31, 66, 31); arrow(58, 31, 66, 13)
    # c: dynamics
    box(136, 8, 44, 46, "Link dynamics", "each period, agents\n• drop the least valuable link\n  if value < marginal cost\n• propose a link to a friend\n  of a friend\n\nrule 1: both must gain\nrule 2: joint surplus > joint\n  cost, split by transfers", INK)
    arrow(126, 31, 134, 31)
    save(fig, "Fig1_model")



def letters(fig, axes_letters):
    """Panel letters placed after layout, left of each panel's full extent, at title height."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, l in axes_letters:
        tb = ax.get_tightbbox(r)
        x0, y1 = inv.transform((tb.x0, tb.y1))
        fig.text(max(x0 - 0.004, 0.002), y1, l, fontsize=9, fontweight="bold", ha="right" if x0 > 0.02 else "left", va="top")


def below_legend(ax, ncol=2, y=-0.28, **kw):
    return ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=6, **kw)


def heat_labels(ax, M, fmt, thresh, size=5.8):
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=size, color="white" if v > thresh else INK)



# =====================================================================
# All simulation figures come from one result set (results/)
# =====================================================================
R8 = R9 = os.path.join(H, "results")              # output of run_all.py
from matplotlib.lines import Line2D


def q5_cmap(n=5):
    cm = plt.get_cmap("Blues")
    return [cm(0.35 + 0.6 * i / (n - 1)) for i in range(n)]


def sd_band(ax, g, col, f=1.0, **kw):
    m, sd = g.mean() * f, g.std().fillna(0) * f
    ax.plot(m.index, m, "-o", ms=2.3, color=col, **kw)
    ax.fill_between(m.index, m - sd, m + sd, color=col, alpha=0.14, lw=0)
    return m


# ---------------------------------------------------------------------
# Figure 2 - dissolution
# ---------------------------------------------------------------------
def fig2():
    p1 = pd.read_csv(R8 + "/P1_aufloesung.csv"); K = pd.read_csv(R8 + "/P1_kappa_alpha.csv")
    nd = pd.read_csv(R8 + "/P1_snapshot_knoten.csv"); ed = pd.read_csv(R8 + "/P1_snapshot_kanten.csv")
    b = p1[~p1.transfers.astype(bool)]
    fig = plt.figure(figsize=(183 * MM, 130 * MM), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1, 1.1], hspace=0.06)
    axa, axb, axc = sf[0].subplots(1, 3)
    for dom, (name, col) in DOM.items():
        d = b[b.domaene == dom].groupby("alpha_param")
        sd_band(axa, d.mittlerer_grad, col, label=name)
        sd_band(axb, d.anteil_isoliert, col, 100, label=name)
    g0 = b[b.domaene == "Wissenschaft"].groupby("alpha_param").mittlerer_grad.mean()
    axa.set_title(f"Degree falls by {100 * (1 - g0[8.0] / g0[0.0]):.0f}% (science)", loc="left")
    axa.set_ylim(0, 9); axa.legend(frameon=False, loc="upper right")
    axa.set_xlabel("AI substitution capability α"); axa.set_ylabel("Mean degree"); style(axa)
    i8 = b.groupby(["domaene", "alpha_param"]).anteil_isoliert.mean()
    axb.set_title(f"Up to {100 * i8.xs(8.0, level=1).max():.0f}% work without partners", loc="left")
    axb.set_xlabel("AI substitution capability α"); axb.set_ylabel("Isolated agents (%)"); style(axb)
    M = K.groupby(["kappa", "alpha_param"]).mittlerer_grad.mean().unstack()
    axc.imshow(M.values, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=9.5)
    heat_labels(axc, M.values, lambda v: f"{v:.1f}", 5.5, 5.3)
    axc.set_xticks(range(len(M.columns))); axc.set_xticklabels([f"{c:g}" for c in M.columns])
    axc.set_yticks(range(len(M.index))); axc.set_yticklabels([f"{c:g}" for c in M.index])
    axc.set_xlabel("Substitution α"); axc.set_ylabel("Coordination κ"); axc.set_title("Mean degree by α and κ (science)", loc="left")
    for sp in axc.spines.values(): sp.set_visible(False)
    axs = sf[1].subplots(1, 3)
    titles = ["Before AI", "Mid-diffusion", "End of diffusion"]; cmap = plt.get_cmap("Blues")
    for c, t in enumerate(sorted(nd.t.unique())):
        n = nd[nd.t == t].sort_values("id"); e = ed[ed.t == t]
        N = len(n); A = np.zeros((N, N), bool); A[e.i.values, e.j.values] = True; A |= A.T
        u = n.ki_nutzung.values; Y = n.output.values; k = A.sum(1)
        draw_net(axs[c], A, cmap(0.35 + 0.65 * np.clip(u, 0, 1)), 3 + 22 * Y / Y.max())
        axs[c].set_title(f"{titles[c]}: degree {k.mean():.1f}, isolated {np.mean(k == 0):.0%}", fontsize=6.6)
    sf[1].supxlabel("Science, logistic diffusion to α = 6. Colour: AI uptake (darker = more); size: output; isolated agents on the outer ring.",
                    fontsize=6, color=INK2)
    letters(fig, [(axa, "a"), (axb, "b"), (axc, "c"), (axs[0], "d")])
    save(fig, "Fig2_dissolution")


# ---------------------------------------------------------------------
# Figure 3 - independence and selection (new)
# ---------------------------------------------------------------------
def fig3():
    p1 = pd.read_csv(R8 + "/P1_aufloesung.csv"); sm = pd.read_csv(R8 + "/P3_solo_nach_quintilen.csv")
    b = p1[~p1.transfers.astype(bool)]
    fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 138 * MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.1, wspace=0.08)
    axa, axb, axc, axd = axs.ravel()
    cols = q5_cmap()
    s = b[b.domaene == "Wissenschaft"].groupby("alpha_param")
    for q in range(1, 6):
        m = s[f"abhaengigkeit_ki_q{q}"].mean() * 100
        axa.plot(m.index, m, "-o", ms=2.3, color=cols[q - 1], label=f"Q{q}")
    axa.set_ylim(0, 100); axa.set_xlabel("AI substitution capability α"); axa.set_ylabel("Mean individual dependence (%)")
    axa.set_title("Independence follows AI use (science)", loc="left"); style(axa)
    axa.legend(frameon=False, loc="upper right", title="AI-uptake quintile", title_fontsize=6, ncol=1)
    for dom, (name, col) in list(DOM.items())[:2]:
        d = b[b.domaene == dom].groupby("alpha_param")
        axb.plot(d.isoliert_typ_ideen_q1.mean().index, d.isoliert_typ_ideen_q1.mean() * 100, "-o", ms=2.3, color=col, label=f"{name}: weakest ideas (Q1)")
        axb.plot(d.isoliert_typ_ideen_q5.mean().index, d.isoliert_typ_ideen_q5.mean() * 100, "--o", ms=2.3, color=col, mfc="white", label=f"{name}: strongest ideas (Q5)")
    axb.set_ylim(0, 85); axb.set_xlabel("AI substitution capability α"); axb.set_ylabel("Without any partner (%)")
    axb.set_title("Exit selects on idea quality (fixed costs)", loc="left"); style(axb)
    axb.legend(frameon=False, loc="upper left", fontsize=6)
    S = sm.pivot(index="qa", columns="qu", values="solo") * 100
    I = sm.pivot(index="qa", columns="qu", values="isoliert") * 100
    axc.imshow(S.values, origin="lower", cmap="Blues", vmin=0, vmax=130)
    for (i, j), v in np.ndenumerate(S.values):
        iso = I.values[i, j]
        colr = "white" if v >= 80 else INK
        if iso >= 10:
            axc.text(j, i + 0.13, f"{v:.0f}%", ha="center", va="center", fontsize=5.6, color=colr)
            axc.text(j, i - 0.2, f"iso {iso:.0f}%", ha="center", va="center", fontsize=4.9, color="#ffd2d1" if v >= 80 else RED, fontweight="bold")
        else:
            axc.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=5.6, color=colr)
    axc.set_xticks(range(5)); axc.set_xticklabels([f"Q{i}" for i in range(1, 6)])
    axc.set_yticks(range(5)); axc.set_yticklabels([f"Q{i}" for i in range(1, 6)])
    axc.set_xlabel("AI uptake quintile"); axc.set_ylabel("Idea quality quintile")
    axc.set_title("Publishing alone after AI diffusion (science)", loc="left")
    for sp in axc.spines.values(): sp.set_visible(False)
    q1 = pd.read_csv(R9 + "/Q1_selektion.csv")
    q = q1[(q1.domaene == "Wissenschaft") & (q1.rho == 0.0)]
    for tr, ls, mk in [(False, "-", "o"), (True, "--", "s")]:
        g = q[q.transfers == tr].groupby("kosten_skalierung")
        axd.plot(g.pkorr_ideen.mean().index, g.pkorr_ideen.mean(), ls, marker=mk, ms=2.8, color=RED)
        axd.plot(g.pkorr_ki.mean().index, g.pkorr_ki.mean(), ls, marker=mk, ms=2.8, color=BLUE)
    axd.axhline(0, color=INK, lw=0.6)
    hd = [Line2D([], [], color=RED, lw=1.5, label="idea quality"), Line2D([], [], color=BLUE, lw=1.5, label="AI uptake"),
          Line2D([], [], color=INK2, ls="-", marker="o", ms=2.8, label="no transfers"), Line2D([], [], color=INK2, ls="--", marker="s", ms=2.8, label="paid collaboration")]
    axd.legend(handles=hd, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6)
    axd.set_xticks([0, 0.25, 0.5, 0.75, 1]); axd.set_xticklabels(["0\nfixed", "0.25", "0.5", "0.75", "1\n∝ productivity"])
    axd.set_ylim(-0.6, 1.0); axd.set_xlabel("Scaling of collaboration cost with own productivity ω")
    axd.set_ylabel("Partial rank correlation with\nbeing without partner (α = 6)")
    axd.set_title("Selection shifts from ideas to AI use", loc="left"); style(axd)
    letters(fig, [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")])
    save(fig, "Fig3_independence_selection")


# ---------------------------------------------------------------------
# Figure 4 - inequality
# ---------------------------------------------------------------------
def fig4():
    G = pd.read_csv(R8 + "/P2_gewinnmatrix.csv", index_col=0); R = pd.read_csv(R8 + "/P2_rho_eta.csv")
    D = pd.read_csv(R8 + "/P2_atlas.csv")
    fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 135 * MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.08, wspace=0.08)
    axs = axs.ravel()
    ax = axs[0]; M = G.values
    ax.imshow(np.log2(M), origin="lower", cmap="Blues", vmin=0, vmax=4.2)
    heat_labels(ax, M, lambda v: f"{v:.1f}×", 6)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i}" for i in range(1, 6)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"Q{i}" for i in range(1, 6)])
    ax.set_xlabel("AI uptake quintile"); ax.set_ylabel("Idea quality quintile")
    ax.set_title("Median output gain, economy (η = 1, α = 6)", loc="left")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax = axs[1]
    for dom, (name, col) in list(DOM.items())[:2]:
        for eta, ls in [(0.0, "--"), (1.0, "-")]:
            d = R[(R.domaene == dom) & (R.eta == eta)].groupby("rho").gini_output.mean()
            ax.plot(d.index, d, ls, marker="o", ms=2.5, color=col, label=f"{name}, η = {eta:g}")
    ax.set_xlabel("Correlation of ideas and AI uptake ρ"); ax.set_ylabel("Gini of output")
    ax.set_title("Inequality rises when the best ideas use AI most", loc="left"); style(ax)
    below_legend(ax, ncol=2, y=-0.2)
    ax = axs[2]; etas = [-1.0, 0.0, 1.0, 2.0]; x = np.arange(4)
    for k, (dom, (name, col)) in enumerate(list(DOM.items())[:2]):
        q1 = R[(R.domaene == dom) & (R.rho == 0.3)].groupby("eta").isoliert_ideen_q1.mean().reindex(etas) * 100
        ax.bar(x + (k - 0.5) * 0.38, q1.values, 0.36, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels([f"{e:g}" for e in etas]); ax.set_ylim(0, 100)
    ax.set_xlabel("Complementarity of AI and ideas η"); ax.set_ylabel("Isolated in weakest idea quintile (%)")
    ax.set_title("Complementarity sharpens the selection", loc="left"); style(ax)
    ax.legend(frameon=False, loc="upper left")
    ax = axs[3]
    for dom, (name, col) in DOM.items():
        d = D[D.domaene == dom].groupby("verteilung").agg(gu=("gini_ki", "first"), gy=("gini_output", "mean"))
        r_ = np.corrcoef(d.gu, d.gy)[0, 1]
        ax.scatter(d.gu, d.gy, s=12, color=col, edgecolors="white", linewidths=0.3)
        ax.text(0.8, d.gy.iloc[np.argmax(d.gu.values)] + 0.035, f"{name}, r = {r_:.2f}", color=col, fontsize=6.3, va="center")
    ax.set_xlim(-0.02, 1.1); ax.set_ylim(0.15, 0.82)
    ax.set_xlabel("Gini of AI uptake (20 distributions, equal means)"); ax.set_ylabel("Gini of output")
    ax.set_title("Unequal access to AI → unequal output", loc="left"); style(ax)
    letters(fig, [(axs[0], "a"), (axs[1], "b"), (axs[2], "c"), (axs[3], "d")])
    save(fig, "Fig4_inequality")


# ---------------------------------------------------------------------
# Figure 5 - publications
# ---------------------------------------------------------------------
def fig5():
    ts = pd.read_csv(R8 + "/P3_publikationen_zeit.csv"); ag = pd.read_csv(R8 + "/P3_publikationen_agenten.csv")
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 66 * MM), layout="constrained")
    ax = axs[0]
    g = ts[ts.t >= -20].groupby("t")
    m, sd = g.einzelautor_anteil.mean() * 100, g.einzelautor_anteil.std() * 100; m2 = g.autoren_pro_paper.mean()
    ax.plot(m.index, m, color=BLUE); ax.fill_between(m.index, m - sd, m + sd, color=BLUE, alpha=0.15, lw=0)
    ax.axvline(0, color=AXIS, lw=0.8, ls="--")
    pre = m.loc[0]
    ax.annotate(f"before AI:\n{pre:.0f}%", (-12, pre), (-19, 52), fontsize=6.3, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.text(m.index[-1], m.iloc[-1] - 9, f"{m.iloc[-1]:.0f}%", fontsize=6.3, color=BLUE, ha="right", va="top")
    ax.set_ylim(0, 105); ax.set_xlabel("Period (0 = AI diffusion starts)"); ax.set_ylabel("Single-authored papers (%)")
    ax.set_title(f"Authors per paper: {m2.loc[0]:.1f} → {m2.iloc[-1]:.1f}", loc="left"); style(ax)
    ax = axs[1]
    for col, lab, c in [("papers_vor", "before AI", GREY), ("papers", "after AI", BLUE)]:
        xs = np.sort(np.clip(ag[col].values, 0, None)); cum = np.r_[0, np.cumsum(xs)] / xs.sum()
        gi = (2 * np.arange(1, len(xs) + 1) - len(xs) - 1) @ xs / (len(xs) * xs.sum())
        ax.plot(np.linspace(0, 1, len(cum)), cum, color=c, label=f"{lab} (Gini {gi:.2f})")
    ax.plot([0, 1], [0, 1], ":", color=AXIS, lw=0.8)
    ax.set_xlabel("Share of researchers"); ax.set_ylabel("Share of papers"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("Papers concentrate", loc="left"); style(ax)
    ax = axs[2]
    ag["qa"] = ag.groupby("seed").ideen.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    q = ag.groupby("qa")[["papers_vor", "erstautor", "koautor"]].mean(); xq = np.arange(5)
    ax.bar(xq - 0.2, q.papers_vor, 0.38, color=GREY, label="before AI")
    ax.bar(xq + 0.2, q.erstautor, 0.38, color=BLUE, label="after AI: led")
    ax.bar(xq + 0.2, q.koautor, 0.38, bottom=q.erstautor, color="#86b6ef", label="after AI: co-authored")
    ax.set_yscale("log"); ax.set_ylim(0.1, 40)
    ax.set_xticks(xq); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("Idea quality quintile (Q1 = weakest)"); ax.set_ylabel("Papers per person (log scale)")
    ax.set_title("Co-authorships disappear", loc="left"); style(ax)
    below_legend(ax, ncol=2, y=-0.25)
    letters(fig, [(axs[0], "a"), (axs[1], "b"), (axs[2], "c")])
    save(fig, "Fig5_publications")


# ---------------------------------------------------------------------
# Figure 6 - bottlenecks and value
# ---------------------------------------------------------------------
def fig6():
    g1 = pd.read_csv(R8 + "/P4_nadeloehr_alpha.csv"); g3 = pd.read_csv(R8 + "/P4_nadeloehr_knappheit.csv")
    k1 = pd.read_csv(R8 + "/P5_myerson.csv")
    kn = pd.read_csv(R8 + "/P4_snapshot_knoten.csv"); ke = pd.read_csv(R8 + "/P4_snapshot_kanten.csv")
    fig = plt.figure(figsize=(183 * MM, 140 * MM), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)
    axa, axb, axc = sf[0].subplots(1, 3)
    N = len(kn); A = np.zeros((N, N), bool); A[ke.i.values, ke.j.values] = True; A |= A.T
    spec, need = kn.spezialist.values.astype(bool), kn.beduerftig.values.astype(bool)
    draw_net(axa, A, np.where(spec, RED, np.where(need, ORANGE, BLUE)), np.where(spec, 28, 5))
    axa.set_title("Hubs around specialists (α = 6)", fontsize=6.6)
    for c, l in [(RED, "specialist"), (ORANGE, "needs bottleneck"), (BLUE, "other")]:
        axa.scatter([], [], c=c, s=12, label=l)
    axa.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, handletextpad=0.2, columnspacing=0.8)
    nb = g1[g1.domaene == "Wissenschaft"].groupby("alpha_param")
    axb.plot(nb.grad_spezialisten.mean(), "-o", ms=2.5, color=RED, label="specialists")
    axb.plot(nb.grad_andere.mean(), "-o", ms=2.5, color=GREY, label="others")
    axb.set_xlim(0, 8.3); axb.set_ylim(0, 20); axb.legend(frameon=False, loc="upper left", ncol=2)
    axb.set_xlabel("AI substitution capability α"); axb.set_ylabel("Mean degree"); axb.set_title("Specialists gain links (science)", loc="left"); style(axb)
    h = g3[(g3.domaene == "Wissenschaft") & g3.nb_phi.isna()] if "nb_phi" in g3 else g3[g3.domaene == "Wissenschaft"]
    M = h.groupby(["f_need", "f_spec"]).rente_spezialisten.mean().unstack()
    axc.imshow(np.log10(M.values), origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1.9)
    heat_labels(axc, M.values, lambda v: f"{v:.0f}×" if v >= 9.5 else f"{v:.1f}×", 10)
    axc.set_xticks(range(len(M.columns))); axc.set_xticklabels([f"{c:.0%}" for c in M.columns])
    axc.set_yticks(range(len(M.index))); axc.set_yticklabels([f"{c:.0%}" for c in M.index])
    axc.set_xlabel("Specialists (share of agents)"); axc.set_ylabel("Agents needing the bottleneck")
    axc.set_title("Scarcity rent (median income ratio)", loc="left")
    for sp in axc.spines.values(): sp.set_visible(False)
    axd, axe = sf[1].subplots(1, 2)
    for dom, (name, col) in list(DOM.items())[:2]:
        for sc, ls in [("ohne Nadeloehr", "--"), ("mit Nadeloehr", "-")]:
            g = k1[(k1.domaene == dom) & (k1.szenario == sc)].groupby("alpha_param").wert_von_anderen_brutto
            axd.errorbar(g.mean().index, g.mean() * 100, yerr=g.std() * 100, fmt=ls, marker="o", ms=2.5, color=col, capsize=1.5, lw=1.2)
    axd.set_ylim(0, 45)
    axd.set_xlabel("AI substitution capability α"); axd.set_ylabel("Output contributed by others (%)")
    axd.set_title("Value created through others collapses (Myerson value)", loc="left"); style(axd)
    hd = [Line2D([], [], color=BLUE, ls="--", marker="o", ms=2.5, label="Science"), Line2D([], [], color=BLUE, ls="-", marker="o", ms=2.5, label="Science + bottleneck"),
          Line2D([], [], color=ORANGE, ls="--", marker="o", ms=2.5, label="Economy"), Line2D([], [], color=ORANGE, ls="-", marker="o", ms=2.5, label="Economy + bottleneck")]
    axd.legend(handles=hd, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, handlelength=2.6)
    k = k1[k1.szenario == "mit Nadeloehr"]; x = np.arange(2); w = 0.25
    for i, (cn, lab, c) in enumerate([("shapley_anteil_spezialisten", "Shapley value", RED),
                                      ("einkommen_anteil_spezialisten", "bargained income", "#f2a3a2"),
                                      ("output_anteil_spezialisten", "own output", GREY)]):
        vals = [k[(k.domaene == dm) & (k.alpha_param == 8.0)][cn].mean() * 100 for dm in ["Wissenschaft", "Wirtschaft"]]
        axe.bar(x + (i - 1) * w, vals, w * 0.92, color=c, label=lab)
        for xi, v in zip(x + (i - 1) * w, vals):
            axe.text(xi, v + 0.3, f"{v:.0f}%", ha="center", fontsize=6)
    axe.set_xticks(x); axe.set_xticklabels(["Science", "Economy"]); axe.set_ylim(0, 14)
    axe.set_ylabel("Specialists' share at α = 8 (%)"); axe.set_title("5% of agents, about 10% of the value", loc="left"); style(axe)
    below_legend(axe, ncol=3, y=-0.2)
    letters(fig, [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e")])
    save(fig, "Fig6_bottlenecks_value")


# ---------------------------------------------------------------------
# Figure 7 - theory and sensitivity
# ---------------------------------------------------------------------
def fig7():
    V = pd.read_csv(R8 + "/P6_theorie_vs_simulation.csv"); HI = pd.read_csv(R8 + "/P6_hohes_alpha.csv")
    TR = pd.read_csv(R8 + "/P6_verlaeufe_personen.csv"); PH = pd.read_csv(R8 + "/P6_phasendiagramme.csv")
    PR = pd.read_csv(R8 + "/P6_prcc.csv", index_col=0)
    fig = plt.figure(figsize=(183 * MM, 128 * MM), layout="constrained")
    st, sb = fig.subfigures(2, 1, height_ratios=[1.1, 0.9], hspace=0.05)
    axa, axb, axc = st.subplots(1, 3)
    v = V[V.domaene == "Wissenschaft"]; h = HI[HI.domaene == "Wissenschaft"].groupby("alpha_param")
    xs = np.r_[v.alpha_param.values, h.theorie_isoliert.mean().index.values[1:]]
    th = np.r_[v.theorie_isoliert.values, h.theorie_isoliert.mean().values[1:]] * 100
    axa.plot(xs, th, color=INK, label="mean-field theory")
    sim = np.r_[v.sim_ohne_transfers.values, h.anteil_isoliert.mean().values[1:]] * 100
    axa.plot(xs, sim, "o", ms=3, color=BLUE, label="simulation")
    axa.plot(v.alpha_param, v.sim_mit_transfers * 100, "s", ms=2.6, color=GREY, label="simulation, transfers")
    axa.set_xscale("symlog", linthresh=8); axa.set_xticks([0, 2, 4, 6, 8, 16, 32]); axa.set_xticklabels(["0", "2", "4", "6", "8", "16", "32"])
    axa.set_ylim(0, 25); axa.set_xlabel("AI substitution capability α"); axa.set_ylabel("Isolated agents (%)")
    r_ = np.corrcoef(v.theorie_isoliert, v.sim_ohne_transfers)[0, 1]
    axa.set_title(f"Theory vs simulation (r = {r_:.3f})", loc="left"); axa.legend(frameon=False, fontsize=5.8, loc="upper left"); style(axa)
    q3 = pd.read_csv(R9 + "/Q3_rueckkehr_simulation.csv")
    for dom, (name, col) in list(DOM.items())[:2]:
        g = q3[q3.domaene == dom].groupby("alpha")
        axb.plot(g.sim_zurueck.mean().index, g.sim_zurueck.mean() * 100, "o-", ms=2.5, color=col, label=f"{name}: returned (simulation)")
        axb.plot(g.theorie_wieder_willig.mean().index, g.theorie_wieder_willig.mean() * 100, "--", color=col, label=f"{name}: willing again (theory)")
    axb.set_xscale("symlog", linthresh=8); axb.set_xticks([0, 4, 8, 16, 32]); axb.set_xticklabels(["0", "4", "8", "16", "32"])
    axb.set_ylim(0, 22); axb.set_xlabel("AI substitution capability α"); axb.set_ylabel("Agents back after isolation (%)")
    axb.set_title("Isolation is a window", loc="left"); axb.legend(frameon=False, fontsize=5.4, loc="upper left"); style(axb)
    ph = PH[(PH.domaene == "Wissenschaft") & (PH.achse == "kosten")]
    M = ph.pivot(index="wert", columns="alpha", values="isoliert") * 100
    Rr = ph.pivot(index="wert", columns="alpha", values="zurueckgekehrt") * 100
    al, cf = M.columns.values.astype(float), M.index.values.astype(float)
    axc.pcolormesh(al, cf, M.values, cmap="Blues", vmin=0, vmax=100, shading="auto")
    cs = axc.contour(al, cf, M.values, levels=[10, 50], colors=[INK, "white"], linewidths=0.8); axc.clabel(cs, fmt="%d%%", fontsize=6)
    if Rr.values.max() >= 5:
        axc.contour(al, cf, Rr.values, levels=[5], colors=[RED], linewidths=0.9, linestyles="--")
        axc.legend(handles=[Line2D([], [], color=RED, ls="--", lw=0.9, label="≥5% have returned")], frameon=False, loc="lower left", fontsize=5.8)
    axc.set_yscale("log"); axc.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axc.set_yticks([0.25, 0.5, 1, 2, 4]); axc.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    axc.set_xlabel("AI substitution capability α"); axc.set_ylabel("Collaboration cost (× baseline)")
    axc.set_title("Phase diagram (share isolated)", loc="left")
    for sp in axc.spines.values(): sp.set_visible(False)
    axd = sb.subplots(1, 1)
    q2 = pd.read_csv(R9 + "/Q2_kanaele.csv"); q2 = q2[q2.domaene == "Wissenschaft"]
    combos = [(0, 0, 0, "none"), (1, 0, 0, "S"), (0, 1, 0, "A"), (0, 0, 1, "C"), (1, 1, 0, "S + A\n(baseline)"), (1, 0, 1, "S + C"), (0, 1, 1, "A + C"), (1, 1, 1, "S + A + C")]
    outs = [("abhaengigkeit", "BASIS", 100, "mean individual\ndependence (%)"), ("grad_verhaeltnis", "BASIS", 1, "degree\n(× pre-AI)"),
            ("isoliert", "BASIS", 100, "without\npartner (%)"), ("gini_zuwachs", "BASIS", 1, "Gini\nincrease"),
            ("kantenanteil_spez", "NADELOEHR", 100, "links involving\nspecialists (%)")]
    Mv = np.zeros((len(outs), len(combos))); Mc = np.zeros_like(Mv)
    for j, (S_, A_, C_, _) in enumerate(combos):
        for i, (c, cfg, f, _) in enumerate(outs):
            d = q2[(q2.konfiguration == cfg) & (q2.substitution == S_) & (q2.amplifikation == A_) & (q2.koordination == C_)]
            Mv[i, j] = d[c].mean() * f
    ref = Mv[:, 0:1]
    rng_ = np.abs(Mv - ref).max(axis=1, keepdims=True) + 1e-9
    Mc = np.clip((Mv - ref) / rng_, -1, 1)
    im = axd.imshow(Mc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(Mv):
        fmt = f"{v:.2f}" if outs[i][0] in ("grad_verhaeltnis", "gini_zuwachs") else f"{v:.0f}"
        axd.text(j, i, fmt, ha="center", va="center", fontsize=5.8, color="white" if abs(Mc[i, j]) > 0.6 else INK)
    axd.set_xticks(range(len(combos))); axd.set_xticklabels([c[3] for c in combos], fontsize=6)
    axd.set_yticks(range(len(outs))); axd.set_yticklabels([o[3] for o in outs], fontsize=6)
    axd.set_title("Which channel does what: AI channels switched on (S, substitution; A, amplification; C, coordination), science, α = 6", loc="left")
    for sp in axd.spines.values(): sp.set_visible(False)
    cb = sb.colorbar(im, ax=axd, shrink=0.8, aspect=12, pad=0.01, ticks=[-1, 0, 1]); cb.outline.set_visible(False)
    cb.ax.set_yticklabels(["lower", "as without AI", "higher"], fontsize=5.5)
    letters(fig, [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")])
    save(fig, "Fig7_theory_channels")


# ---------------------------------------------------------------------
# Figure 8 - empirical evidence
# ---------------------------------------------------------------------
def fig8():
    oa = pd.read_csv(R6 + "/A_openalex_teamgroessen.csv"); dv = pd.read_csv(R7 + "/V7_abweichungen_teilgebiete.csv")
    te = pd.read_csv(R7 + "/V7_testergebnisse.csv")
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 72 * MM), layout="constrained", width_ratios=[1, 1, 1.1])
    ax = axs[0]
    a = oa[(oa.feld_id == "ALLE") & (oa.jahr <= 2025)].set_index("jahr")
    ax.plot(a.index, a.anteil_a1 * 100, "-o", ms=2.5, color=INK, label="all fields")
    for cl, col, lab in [("rechnerisch", BLUE, "computational fields"), ("physisch", ORANGE, "physical fields")]:
        g = oa[(oa.klasse == cl) & (oa.jahr <= 2025)].groupby("jahr")[["a1", "gesamt"]].sum()
        ax.plot(g.index, g.a1 / g.gesamt * 100, "-", color=col, lw=1.2, label=lab)
    pre = a.loc[2012:2022]; b = np.polyfit(pre.index, np.log(pre.anteil_a1 / (1 - pre.anteil_a1)), 1)
    yy = np.arange(2012, 2026); ax.plot(yy, 100 / (1 + np.exp(-np.polyval(b, yy))), ":", color=INK2, lw=1, label="pre-2023 trend (all)")
    ax.axvspan(2022.5, 2025.5, color="#f0efec", zorder=0)
    ax.set_ylim(0, 75); ax.set_xlabel("Publication year"); ax.set_ylabel("Single-authored articles (%)")
    ax.set_title("The decline halts after 2022", loc="left"); style(ax)
    below_legend(ax, ncol=2, y=-0.22)
    ax = axs[1]
    d = dv[dv.eingeschlossen]; bins = np.linspace(-1.2, 1.4, 27)
    ax.hist(d.placebo_a1, bins=bins, color=GREY, alpha=0.6, label=f"placebo break 2019 ({(d.placebo_a1 > 0).mean():.0%} > 0)")
    ax.hist(d.abw_a1, bins=bins, histtype="step", color=BLUE, lw=1.5, label=f"2023–2025 ({(d.abw_a1 > 0).mean():.0%} > 0)")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_xlabel("Deviation from pre-trend (logit)"); ax.set_ylabel("Subfields (n = 239)")
    ax.set_title("Above trend in most subfields", loc="left"); style(ax)
    below_legend(ax, ncol=1, y=-0.22)
    ax = axs[2]
    lab = ["H1 within fields", "H2 across fields", "H3 large teams", "placebo 2019", "weighted", "quadratic trend", "rule-classified"]
    est = te.koef.fillna(te["diff"]).values; y = np.arange(len(est))[::-1]
    for yi, e, lo, hi, i in zip(y, est, te.ki_lo, te.ki_hi, range(len(est))):
        c = BLUE if i < 3 else GREY
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.2); ax.plot(e, yi, "o", color=c, ms=3.5)
    ax.axvline(0, color=INK, lw=0.7); ax.set_ylim(-0.6, len(est) - 0.4)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.2)
    ax.set_xlabel("Computational − physical\n(logit, 95% CI)")
    ax.set_title("Preregistered tests", loc="left"); style(ax)
    letters(fig, [(axs[0], "a"), (axs[1], "b"), (axs[2], "c")])
    save(fig, "Fig8_empirics")




if __name__ == "__main__":
    import glob
    for f in glob.glob(os.path.join(OUT, "Fig*")):
        os.remove(f)
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        f(); print("done", f.__name__)
