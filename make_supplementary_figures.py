# -*- coding: utf-8 -*-
"""Supplementary Figs 1-10. Run after run_all.py and make_figures.py: python make_supplementary_figures.py"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_figures.py"), encoding="utf-8").read().split("# Figure 2 - dissolution")[0]
exec(src.rsplit("# ---------------------------------------------------------------------", 1)[0])
R8 = R9 = RS = os.path.join(H, "results")
OUT = os.path.join(H, "figures_supplementary"); os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=250); plt.close(fig)


def band(ax, g, col, f=1, ls="-", label=None):
    m, sd = g.mean() * f, g.std().fillna(0) * f
    ax.plot(m.index, m, ls, color=col, label=label, lw=1.3)
    ax.fill_between(m.index, m - sd, m + sd, color=col, alpha=0.13, lw=0)


def S1():
    A = pd.read_csv(RS + "/SA_zeitreihen.csv"); A = A[A.t >= -20]
    fig, axs = plt.subplots(1, 5, figsize=(183 * MM, 52 * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08)
    specs = [("mittlerer_grad", 1, "Mean degree"), ("anteil_isoliert", 100, "Isolated (%)"),
             ("abhaengigkeit", 100, "Mean dependence (%)"), ("gini_output", 1, "Gini of output")]
    for ax, (c, f, lab) in zip(axs, specs):
        for dom, (name, col) in DOM.items():
            band(ax, A[A.domaene == dom].groupby("t")[c], col, f, label=name)
        ax.axvline(0, color=AXIS, lw=0.7, ls="--"); ax.set_ylabel(lab); ax.set_xlabel("Period"); style(ax)
    fig.legend(handles=[Line2D([], [], color=c, label=n) for n, c in DOM.values()], loc="outside lower center", ncol=3, frameon=False, fontsize=6)
    s = A[A.domaene == "Wissenschaft"].groupby("t")
    band(axs[4], s.einzelautor_anteil, BLUE, 100); axs[4].set_ylabel("Solo papers (%)")
    axs[4].set_xlabel("Period"); axs[4].axvline(0, color=AXIS, lw=0.7, ls="--"); style(axs[4])
    letters(fig, [(a, l) for a, l in zip(axs, "abcde")])
    save(fig, "FigS1_time_series")


def S2():
    D = pd.read_csv(R8 + "/P2_atlas.csv")
    g = D.groupby(["verteilung", "domaene"]).agg(gu=("gini_ki", "first"), gy=("gini_output", "mean"),
                                                  iso=("anteil_isoliert", "mean")).reset_index()
    order = g.groupby("verteilung").gu.first().sort_values().index
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 95 * MM), layout="constrained", width_ratios=[0.6, 1, 1])
    y = np.arange(len(order))
    axs[0].barh(y, g.groupby("verteilung").gu.first().reindex(order), color=GREY)
    EN = {"konstant": "constant", "gleich": "uniform", "normal_eng": "normal, narrow", "normal_breit": "normal, wide", "zugangsluecke_40": "access gap (40% non-users)", "bimodal_30/70": "bimodal 30/70", "weibull_1(exp)": "exponential", "beta_U(0.5,0.5)": "beta(0.5, 0.5)", "beta_(2,5)": "beta(2, 5)", "beta_(5,2)": "beta(5, 2)"}
    lab = [EN.get(o, o.replace("_", " ")) for o in order]
    axs[0].set_yticks(y); axs[0].set_yticklabels(lab, fontsize=5.8); axs[0].set_xlabel("Gini of AI uptake"); style(axs[0])
    for k, (c, f, lab) in enumerate([("gy", 1, "Gini of output (α = 6)"), ("iso", 100, "Without partner at α = 6 (%)")]):
        ax = axs[k + 1]
        for dom, (name, col) in DOM.items():
            v = g[g.domaene == dom].set_index("verteilung").reindex(order)[c] * f
            ax.plot(v.values, y, "o", ms=3, color=col, label=name)
        ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel(lab); style(ax)
    axs[1].legend(frameon=False, fontsize=6, loc="lower right")
    letters(fig, [(axs[0], "a"), (axs[1], "b"), (axs[2], "c")])
    save(fig, "FigS2_uptake_atlas")


def S3():
    B = pd.read_csv(RS + "/SB_topologien.csv")
    topo = ["ER", "Ring", "WS", "BA", "HK", "SBM", "CP", "RGG", "ChungLu"]
    names = ["Erdős–Rényi", "Ring", "Watts–Strogatz", "Barabási–Albert", "Holme–Kim", "Block model", "Core–periphery", "Geometric", "Chung–Lu"]
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 70 * MM), layout="constrained")
    x = np.arange(len(topo))
    for ax, (c, f, al, lab) in zip(axs, [("mittlerer_grad", 1, 0.0, "Mean degree before AI"), ("mittlerer_grad", 1, 6.0, "Mean degree at α = 6"),
                                         ("anteil_isoliert", 100, 6.0, "Without partner at α = 6 (%)")]):
        for dom, (name, col) in list(DOM.items())[:2]:
            for net, mk, fill in [("flexibel", "o", col), ("starr", "s", "white")]:
                v = B[(B.domaene == dom) & (B.netz == net) & (B.alpha_param == al)].groupby("topologie")[c].mean().reindex(topo) * f
                ax.plot(x, v.values, mk, ms=3.5, color=col, mfc=fill, mew=0.9)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=50, ha="right", fontsize=5.8); ax.set_ylabel(lab); style(ax)
    axs[0].set_ylim(0, 6); axs[1].set_ylim(0, 6)
    hd = [Line2D([], [], color=BLUE, marker="o", ls="", label="Science"), Line2D([], [], color=ORANGE, marker="o", ls="", label="Economy"),
          Line2D([], [], color=INK2, marker="o", ls="", label="flexible (50% review/period)"), Line2D([], [], color=INK2, marker="s", mfc="white", ls="", label="rigid (5%)")]
    fig.legend(handles=hd, loc="outside lower center", ncol=4, frameon=False, fontsize=6)
    letters(fig, [(a, l) for a, l in zip(axs, "abc")])
    save(fig, "FigS3_topologies")


def S4():
    P = pd.read_csv(R8 + "/P1_aufloesung.csv")
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 60 * MM), layout="constrained")
    for ax, (c, f, lab) in zip(axs, [("mittlerer_grad", 1, "Mean degree"), ("anteil_isoliert", 100, "Isolated (%)"),
                                     ("abhaengigkeit_mittel", 100, "Mean dependence (%)")]):
        for dom, (name, col) in list(DOM.items())[:2]:
            for tr, ls in [(False, "-"), (True, "--")]:
                g = P[(P.domaene == dom) & (P.transfers == tr)].groupby("alpha_param")[c]
                band(ax, g, col, f, ls)
        ax.set_xlabel("AI substitution capability α"); ax.set_ylabel(lab); style(ax)
    hd = [Line2D([], [], color=BLUE, label="Science"), Line2D([], [], color=ORANGE, label="Economy"),
          Line2D([], [], color=INK2, ls="-", label="mutual consent"), Line2D([], [], color=INK2, ls="--", label="transfers")]
    fig.legend(handles=hd, loc="outside lower center", ncol=4, frameon=False, fontsize=6)
    letters(fig, [(a, l) for a, l in zip(axs, "abc")])
    save(fig, "FigS4_link_rule")


def S5():
    T = pd.read_csv(R8 + "/P3_theta_autorschaft.csv").groupby("theta")
    fig, axs = plt.subplots(1, 2, figsize=(120 * MM, 55 * MM), layout="constrained")
    for ax, (a, b, lab) in zip(axs, [("einzelautor_vor", "einzelautor_nach", "Single-authored papers (%)"), ("autoren_vor", "autoren_nach", "Authors per paper")]):
        f = 100 if "einzel" in a else 1
        ax.plot(T[a].mean().index, T[a].mean() * f, "o-", color=GREY, ms=3, label="before AI")
        ax.plot(T[b].mean().index, T[b].mean() * f, "o-", color=BLUE, ms=3, label="after AI diffusion")
        ax.set_xscale("log"); ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2]); ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%"])
        ax.set_xlabel("Co-authorship threshold θ"); ax.set_ylabel(lab); style(ax)
    axs[0].legend(frameon=False, fontsize=6, loc="lower right")
    letters(fig, [(axs[0], "a"), (axs[1], "b")])
    save(fig, "FigS5_coauthor_threshold")


def S6():
    W = pd.read_csv(R8 + "/P4_arbeit.csv")
    fig, axs = plt.subplots(1, 4, figsize=(183 * MM, 55 * MM), layout="constrained")
    for ax, (c, f, lab) in zip(axs, [("transfervolumen_anteil", 100, "Transfers (% of revenue)"), ("anteil_solo", 100, "Working alone (%)"),
                                     ("anteil_netto_empfaenger", 100, "Net recipients of transfers (%)"), ("armutsquote", 100, "Income < 60% of median (%)")]):
        for dom, (name, col) in list(DOM.items())[:2]:
            for eps, ls in [("inf", "-"), ("1.5", "--")]:
                g = W[(W.domaene == dom) & (W.eps_mkt.astype(str) == eps)].groupby("alpha_param")[c]
                ax.plot(g.mean().index, g.mean() * f, ls, marker="o", ms=2.3, color=col)
        ax.set_xlabel("AI substitution capability α"); ax.set_ylabel(lab); style(ax)
    hd = [Line2D([], [], color=BLUE, label="Science"), Line2D([], [], color=ORANGE, label="Economy"),
          Line2D([], [], color=INK2, ls="-", label="demand absorbs output"), Line2D([], [], color=INK2, ls="--", label="saturating demand (ε = 1.5)")]
    fig.legend(handles=hd, loc="outside lower center", ncol=4, frameon=False, fontsize=6)
    letters(fig, [(a, l) for a, l in zip(axs, "abcd")])
    save(fig, "FigS6_labour")


def S7():
    K = pd.read_csv(R8 + "/P5_myerson.csv")
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), layout="constrained")
    for ax, c, lab in [(axs[0], "wert_von_anderen_brutto", "By others, gross (%)"), (axs[1], "wert_von_anderen_netto", "Net measure (%)")]:
        for dom, (name, col) in list(DOM.items())[:2]:
            for sc, ls in [("ohne Nadeloehr", "--"), ("mit Nadeloehr", "-")]:
                g = K[(K.domaene == dom) & (K.szenario == sc)].groupby("alpha_param")[c]
                ax.plot(g.mean().index, g.mean() * 100, ls, marker="o", ms=2.5, color=col)
        ax.set_xlabel("AI substitution capability α"); ax.set_ylabel(lab); style(ax)
    ax = axs[2]
    for dom, (name, col) in list(DOM.items())[:2]:
        for sc, mk in [("ohne Nadeloehr", "o"), ("mit Nadeloehr", "s")]:
            g = K[(K.domaene == dom) & (K.szenario == sc)].groupby("alpha_param")
            ax.plot(g.rangstabilitaet_shapley.mean(), "-", marker=mk, ms=2.5, color=col)
            ax.plot(g.rangstabilitaet_grad.mean(), ":", marker=mk, ms=2.5, color=col)
    ax.set_xlabel("AI substitution capability α"); ax.set_ylabel("Rank stability (Spearman)"); style(ax)
    hd = [Line2D([], [], color=BLUE, label="Science"), Line2D([], [], color=ORANGE, label="Economy"),
          Line2D([], [], color=INK2, ls="--", label="a, b: without bottleneck"), Line2D([], [], color=INK2, ls="-", label="a, b: with bottleneck"),
          Line2D([], [], color=INK2, ls="-", marker="o", ms=2.5, label="c: Shapley value"), Line2D([], [], color=INK2, ls=":", label="c: degree")]
    fig.legend(handles=hd, loc="outside lower center", ncol=6, frameon=False, fontsize=5.8)
    letters(fig, [(a, l) for a, l in zip(axs, "abc")])
    save(fig, "FigS7_myerson_measures")


def S8():
    P = pd.read_csv(R8 + "/P1_aufloesung.csv"); P = P[~P.transfers.astype(bool) & P.domaene.isin(["Wissenschaft", "Wirtschaft"])]
    Q = pd.read_csv(R9 + "/Q1_selektion.csv")
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 64 * MM), layout="constrained")
    ax = axs[0]
    for dom, (name, col) in list(DOM.items())[:2]:
        g = P[P.domaene == dom].groupby("alpha_param")
        ax.plot(g.anteil_abgehaengt.mean() * 100, "-o", ms=2.3, color=col, label=f"{name}: selected out")
        ax.plot((g.anteil_unabhaengig.mean() + g.anteil_ausgeschlossen.mean() + g.anteil_suchend.mean()) * 100, ":o", ms=2.3, color=col, mfc="white", label=f"{name}: other isolated")
    ax.set_ylim(0, 16); ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.set_xlabel("AI substitution capability α"); ax.set_ylabel("Share of all agents (%)"); style(ax)
    ax = axs[1]
    q = Q[(Q.domaene == "Wirtschaft") & (Q.rho == 0.0)]
    for tr, ls, mk in [(False, "-", "o"), (True, "--", "s")]:
        g = q[q.transfers == tr].groupby("kosten_skalierung")
        ax.plot(g.pkorr_ideen.mean(), ls, marker=mk, ms=2.8, color=RED); ax.plot(g.pkorr_ki.mean(), ls, marker=mk, ms=2.8, color=BLUE)
    ax.axhline(0, color=INK, lw=0.6); ax.set_ylim(-0.7, 1)
    ax.set_xlabel("Cost scaling ω (economy, ρ = 0)"); ax.set_ylabel("Partial rank correlation\nwith being without partner"); style(ax)
    ax = axs[2]
    q = Q[(Q.domaene == "Wissenschaft") & (Q.rho == 0.0) & (~Q.transfers.astype(bool))]
    x = np.arange(3); w = 0.2
    for k, (om, q_, col, lab) in enumerate([(0.0, 1, RED, "ω = 0, weakest ideas"), (0.0, 5, "#f2a3a2", "ω = 0, strongest ideas"),
                                             (1.0, 1, BLUE, "ω = 1, weakest ideas"), (1.0, 5, "#86b6ef", "ω = 1, strongest ideas")]):
        v = [q[q.kosten_skalierung == om][f"iso_ideen_q{q_}_ki_t{t}"].mean() * 100 for t in (1, 2, 3)]
        ax.bar(x + (k - 1.5) * w, v, w * 0.92, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["low", "middle", "high"]); ax.set_xlabel("AI-uptake tercile (science, ρ = 0)")
    ax.set_ylabel("Without partner at α = 6 (%)"); ax.set_ylim(0, 105); style(ax)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    hd = [Line2D([], [], color=RED, label="idea quality (b)"), Line2D([], [], color=BLUE, label="AI uptake (b)"),
          Line2D([], [], color=INK2, ls="-", marker="o", ms=2.5, label="no transfers (b)"), Line2D([], [], color=INK2, ls="--", marker="s", ms=2.5, label="paid collaboration (b)")]
    fig.legend(handles=hd, loc="outside lower center", ncol=4, frameon=False, fontsize=5.8)
    letters(fig, [(a, l) for a, l in zip(axs, "abc")])
    save(fig, "FigS8_selection_robustness")


def S10():
    PR = pd.read_csv(R8 + "/P6_prcc.csv", index_col=0)
    names = {"c0": "link cost c₀", "gamma": "cost convexity γ", "q": "partner contribution q", "phi_cap": "substitution ceiling φ",
             "lam": "amplification λ", "eta": "AI–idea complementarity η", "kappa": "coordination κ", "sigma_a": "idea dispersion σ",
             "K": "number of skills K", "n_strong": "own skills n", "rho": "idea–uptake correlation ρ", "transfers": "transfers"}
    outs = [("grad_verhaeltnis", "degree\nchange"), ("isoliert_4", "share\nisolated"), ("gini_zuwachs", "Gini\nincrease")]
    P = PR[[o for o, _ in outs]].rename(index=names)
    P = P.loc[P.abs().max(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(110 * MM, 95 * MM), layout="constrained")
    im = ax.imshow(P.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(P.values):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in outs]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index, fontsize=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, aspect=20); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
    save(fig, "FigS10_sensitivity")


def S9():
    V = pd.read_csv(R8 + "/P6_theorie_vs_simulation.csv"); HI = pd.read_csv(R8 + "/P6_hohes_alpha.csv")
    PH = pd.read_csv(R8 + "/P6_phasendiagramme.csv"); VT = pd.read_csv(R8 + "/P6_verlaufstypen.csv")
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), layout="constrained", width_ratios=[1, 1, 1.15])
    ax = axs[0]
    for dom, (name, col) in list(DOM.items())[:2]:
        v = V[V.domaene == dom]; h = HI[HI.domaene == dom].groupby("alpha_param")
        xs = np.r_[v.alpha_param.values, h.theorie_isoliert.mean().index.values[1:]]
        ax.plot(xs, np.r_[v.theorie_isoliert.values, h.theorie_isoliert.mean().values[1:]] * 100, "-", color=col)
        ax.plot(xs, np.r_[v.sim_ohne_transfers.values, h.anteil_isoliert.mean().values[1:]] * 100, "o", ms=2.5, color=col)
    ax.set_xscale("symlog", linthresh=8); ax.set_xticks([0, 4, 8, 16, 32]); ax.set_xticklabels(["0", "4", "8", "16", "32"])
    ax.set_xlabel("AI substitution capability α"); ax.set_ylabel("Without partner (%)"); style(ax)
    ax.legend(handles=[Line2D([], [], color=INK2, label="theory"), Line2D([], [], color=INK2, marker="o", ls="", ms=2.5, label="simulation")], frameon=False, fontsize=5.8, loc="upper left")
    ax = axs[1]
    ph = PH[(PH.domaene == "Wissenschaft") & (PH.achse == "amplifikation")]
    for lam, c in zip(sorted(ph.wert.unique()), plt.get_cmap("Blues")(np.linspace(0.35, 1, ph.wert.nunique()))):
        d = ph[ph.wert == lam]; ax.plot(d.alpha, d.isoliert * 100, color=c, label=f"λ = {lam:g}")
    ax.set_xlabel("AI substitution capability α"); ax.set_ylabel("Without partner, theory (%)"); style(ax)
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax = axs[2]
    types = ["nie isoliert", "ein Kipppunkt (kein Wiedereintritt bis alpha_max)", "Isolationsfenster innerhalb [0, 8]", "Wiedereintritt erst oberhalb 8", "schon ohne KI isoliert"]
    labs = ["never isolated", "single tipping point", "window within α ≤ 8", "returns above α = 8", "isolated before AI"]
    cols = [GREY, RED, AQUA, BLUE, INK]
    rows = [("Wissenschaft", "Basis"), ("Wissenschaft", "ohne Amplifikation (lam=0)"), ("Wirtschaft", "Basis"), ("Wirtschaft", "ohne Amplifikation (lam=0)")]
    rl = ["Science", "Science, λ = 0", "Economy", "Economy, λ = 0"]
    left = np.zeros(4)
    for t, l, c in zip(types, labs, cols):
        v = np.array([VT[(VT.domaene == d) & (VT.variante == va) & (VT.typ == t)].anteil.sum() for d, va in rows]) * 100
        ax.barh(np.arange(4), v, left=left, color=c, label=l); left += v
    ax.set_yticks(np.arange(4)); ax.set_yticklabels(rl); ax.invert_yaxis(); ax.set_xlabel("Agents (%)"); ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2); style(ax, grid=False)
    letters(fig, [(a, l) for a, l in zip(axs, "abc")])
    save(fig, "FigS9_theory_extensions")


if __name__ == "__main__":
    for f in [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10]:
        f(); print("done", f.__name__)
