# -*- coding: utf-8 -*-
"""
AI-driven independence in collaboration networks -- simulation and analysis code
=================================================================================

Companion code for:
  C. R. Klein, "AI can substitute for collaborators and reshape networks in
  science and the economy".

Requirements: Python >= 3.7 with numpy, pandas and matplotlib only.
Usage:        python run_all.py            (full analysis, ~10-20 min)
              set QUICK = True below for a 2-minute smoke test.
Output:       results/  (CSV files for every figure and table, SCHLUESSELZAHLEN.csv
              = table of all numbers reported in the paper with source and configuration)

Structure of this file
  1-7  Model: agents, skills, AI channels, network formation, outcomes, simulation loop
  8    Mean-field theory (isolation condition, trajectories)
  9    Analyses reported in the paper
       P1  network dissolution, independence and exit types        (Figs 2, 3a-c)
       P2  inequality, uptake atlas                                 (Fig. 4, Suppl. Fig. 2)
       P3  publications and co-authorship                           (Fig. 5, Suppl. Fig. 5)
       P4  bottleneck skills and labour                             (Fig. 6a-c, Suppl. Fig. 6)
       P5  Myerson value, gross and net decomposition               (Fig. 6d,e, Suppl. Fig. 7)
       P6  mean-field theory vs simulation, phase diagrams, LHS/PRCC (Fig. 7a,c, Suppl. Figs 9, 10)
       P8  cost scaling and paid collaboration                      (Suppl. Fig. 8)
       Q1  selection on ideas vs AI use                             (Fig. 3d, Suppl. Fig. 8)
       Q2  factorial ablation of the three AI channels              (Fig. 7d)
       Q3  willingness vs reconnection                             (Fig. 7b)
       S_A-S_C time series, initial topologies, reciprocity         (Suppl. Figs 1, 3; Note 4)
       P7  table of all reported numbers                            (Supplementary Data 1)

Naming: internal labels are German (the code grew in a German-speaking project).
  Wissenschaft = science, Wirtschaft = economy, Nadeloehr = bottleneck,
  Spezialisten = specialists, isoliert = isolated, Abhaengigkeit = dependence,
  Ideen = idea quality, KI_Nutzung = AI uptake, Kanaele = channels,
  Kosten_Skalierung = cost scaling (omega), Transfers = side payments.
  See README.md for a full glossary of output columns.
"""

import os
import sys
import json
import time
import shutil
import zipfile

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

VERSION = "release"


# ---------------------------------------------------------------
# EINSTELLUNGEN
# ---------------------------------------------------------------
QUICK = False                         # True = Schnelltest (~1 Minute)
N_AGENTS = 150 if QUICK else 300
T_BURN = 40 if QUICK else 60
T_AI = 50 if QUICK else 80
REPS = 2 if QUICK else 5
MASTER_SEED = 20261117

DOMAINS = {
    "Wissenschaft": dict(K=6, n_strong=2, s_hi=0.90, s_lo=0.05, q=0.60,
                         c0=0.020, gamma=0.10, phi_cap=0.95, lam=0.5,
                         kappa=0.0, sigma_a=0.6, eta=0.0, eps_mkt=np.inf,
                         transfers=False),
    "Wirtschaft":   dict(K=8, n_strong=2, s_hi=0.90, s_lo=0.05, q=0.60,
                         c0=0.025, gamma=0.08, phi_cap=0.90, lam=0.8,
                         kappa=0.0, sigma_a=0.8, eta=0.0, eps_mkt=np.inf,
                         transfers=False),
    "Sozial":       dict(K=4, n_strong=1, s_hi=0.60, s_lo=0.02, q=0.50,
                         c0=0.015, gamma=0.15, phi_cap=0.70, lam=0.0,
                         kappa=0.0, sigma_a=0.3, eta=0.0, eps_mkt=np.inf,
                         transfers=False),
}

# Nadeloehr-Kompetenz (Standardwerte; nb_on=True schaltet sie ein)
NB_DEFAULT = dict(nb_on=False,      # Nadeloehr aktiv?
                  nb_f_spec=0.05,   # Anteil Spezialisten (z.B. Kliniker mit Studienzentrum)
                  nb_f_need=0.40,   # Anteil Personen, deren Projekte das Nadeloehr brauchen
                  nb_s_spec=0.95,   # Kompetenz der Spezialisten
                  nb_weight=3.0,    # Gewicht im Output (3 = fast unverzichtbar, z.B. Zulassung)
                  nb_phi=0.05)      # wie viel davon KI ersetzen kann (5%)

DYN = dict(p_rev=0.5, p_tri=0.7, eps=0.003, k_init=6)
U_CAP = 10.0            # Obergrenze der KI-Nutzung (20x Mittelwert)
THETA_AUTHOR = 0.05     # Partner wird Ko-Autor, wenn ohne ihn >5% Output fehlen

COLORS = {"Wissenschaft": "#2a6fdb", "Wirtschaft": "#e07b00", "Sozial": "#1a9e5c"}
SEQ = ["#9ecae1", "#4292c6", "#08519c", "#08306b", "#000000"]


# =====================================================================
#  1. STARTNETZWERKE
# =====================================================================
def initial_network(rng, kind, N, k=6):
    A = np.zeros((N, N), dtype=bool)
    if kind == "ER":
        A = np.triu(rng.random((N, N)) < k / (N - 1), 1)
    elif kind in ("WS", "Ring"):
        beta = 0.1 if kind == "WS" else 0.0
        for i in range(N):
            for s in range(1, k // 2 + 1):
                j = (i + s) % N
                if rng.random() < beta:
                    j = rng.integers(N)
                    while j == i or A[min(i, j), max(i, j)]:
                        j = rng.integers(N)
                A[min(i, j), max(i, j)] = True
    elif kind in ("BA", "HK"):                    # HK = Holme-Kim (BA + Dreiecke)
        m = k // 2
        p_triad = 0.8 if kind == "HK" else 0.0
        S = A
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                S[i, j] = S[j, i] = True
        deg = S.sum(1).astype(float)
        for new in range(m + 1, N):
            p = deg[:new] / deg[:new].sum()
            t = rng.choice(new, p=p)
            targets = {t}
            while len(targets) < m:
                if rng.random() < p_triad:
                    nb = [x for x in np.flatnonzero(S[t, :new]) if x not in targets]
                    if nb:
                        targets.add(nb[rng.integers(len(nb))])
                        continue
                targets.add(rng.choice(new, p=p))
            for t2 in targets:
                S[t2, new] = S[new, t2] = True
                deg[t2] += 1
            deg[new] = m
        A = np.triu(S, 1)
    elif kind == "SBM":                           # Gemeinschaften
        g, p_in = 10, 0.4
        grp = np.arange(N) // g
        p_out = (k - (g - 1) * p_in) / (N - g)
        P = np.where(grp[:, None] == grp[None, :], p_in, p_out)
        A = np.triu(rng.random((N, N)) < P, 1)
    elif kind == "CP":                            # Kern-Peripherie
        core = np.arange(N) < int(0.15 * N)
        nc = core.sum()
        p_cc = 0.5
        p_cp = (k * N - p_cc * nc * (nc - 1)) / (2 * nc * (N - nc)) * 0.9
        P = np.where(core[:, None] & core[None, :], p_cc,
                     np.where(core[:, None] | core[None, :], p_cp, 0.002))
        A = np.triu(rng.random((N, N)) < P, 1)
    elif kind == "RGG":                           # raeumlich (Nachbarschaft)
        x = rng.random((N, 2))
        r = np.sqrt(k / (np.pi * N))
        D = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
        A = np.triu(D < r, 1)
    elif kind == "ChungLu":                       # Potenzgesetz-Gradverteilung
        w = rng.pareto(1.5, N) + 1
        w = w / w.mean() * k
        P = np.minimum(1.0, np.outer(w, w) / w.sum())
        A = np.triu(rng.random((N, N)) < P, 1)
    else:
        raise ValueError(kind)
    A = A | A.T
    np.fill_diagonal(A, False)
    return A


TOPOLOGIES = ["ER", "Ring", "WS", "BA", "HK", "SBM", "CP", "RGG", "ChungLu"]
TOPO_NAMES = {"ER": "Zufall (ER)", "Ring": "Gitter/Ring", "WS": "Small-World",
              "BA": "Skalenfrei (BA)", "HK": "Skalenfrei+Cluster (HK)",
              "SBM": "Gemeinschaften", "CP": "Kern-Peripherie",
              "RGG": "Raeumlich", "ChungLu": "Potenzgesetz (Chung-Lu)"}


# =====================================================================
#  2. VERTEILUNGSATLAS DER KI-NUTZUNG (roh; wird auf Mittel 0.5 skaliert)
# =====================================================================
UPTAKE = {
    "konstant":        lambda r, N: np.ones(N),
    "gleich":          lambda r, N: r.random(N),
    "normal_eng":      lambda r, N: np.clip(r.normal(1, 0.25, N), 0, None),
    "normal_breit":    lambda r, N: np.clip(r.normal(1, 0.8, N), 0, None),
    "lognormal_0.5":   lambda r, N: r.lognormal(0, 0.5, N),
    "lognormal_1.0":   lambda r, N: r.lognormal(0, 1.0, N),
    "lognormal_1.5":   lambda r, N: r.lognormal(0, 1.5, N),
    "weibull_0.5":     lambda r, N: r.weibull(0.5, N),
    "weibull_1(exp)":  lambda r, N: r.weibull(1.0, N),
    "weibull_2":       lambda r, N: r.weibull(2.0, N),
    "weibull_5":       lambda r, N: r.weibull(5.0, N),
    "gamma_0.5":       lambda r, N: r.gamma(0.5, 1, N),
    "gamma_3":         lambda r, N: r.gamma(3.0, 1, N),
    "beta_U(0.5,0.5)": lambda r, N: r.beta(0.5, 0.5, N),
    "beta_(2,5)":      lambda r, N: r.beta(2, 5, N),
    "beta_(5,2)":      lambda r, N: r.beta(5, 2, N),
    "pareto_1.5":      lambda r, N: r.pareto(1.5, N) + 1,
    "pareto_3":        lambda r, N: r.pareto(3.0, N) + 1,
    "bimodal_30/70":   lambda r, N: np.where(r.random(N) < 0.3, r.normal(1, 0.1, N),
                                             r.normal(0.15, 0.05, N)).clip(0),
    "zugangsluecke_40": lambda r, N: np.where(r.random(N) < 0.4, 0.0, r.random(N)),
}


def draw_uptake(rng, kind, N):
    u = UPTAKE[kind](rng, N).astype(float)
    u = 0.5 * u / u.mean()
    u = np.minimum(u, U_CAP)
    return u


def draw_agents(rng, N, p, uptake, rho):
    """Ideenqualitaet a (lognormal) und KI-Nutzung u, gekoppelt ueber
    eine Gauss-Copula mit Rangkorrelation ~rho."""
    z1 = rng.standard_normal(N)
    z2 = rho * z1 + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal(N)
    a = np.exp(p["sigma_a"] * z1)
    a = a / a.mean()
    u_raw = draw_uptake(rng, uptake, N)
    u = np.empty(N)
    u[np.argsort(z2)] = np.sort(u_raw)
    return a, u



def make_skills(rng, N, p):
    """Faehigkeitsmatrix S (N x K'), Gewichte W (N x K'), KI-Ersetzbarkeit je Faehigkeit.
    Optional: eine zusaetzliche Nadeloehr-Faehigkeit (z.B. klinische Studien,
    Core Facility, Rohstoffzugang), die nur wenige besitzen und KI kaum ersetzt."""
    K = p["K"]
    S = np.full((N, K), p["s_lo"])
    for i in range(N):
        S[i, rng.choice(K, p["n_strong"], replace=False)] = p["s_hi"]
    W = np.ones((N, K))
    phi_vec = np.full(K, p["phi_cap"])
    spec = np.zeros(N, bool)
    need = np.zeros(N, bool)
    if p.get("nb_on", False):
        n_spec = max(1, int(round(p["nb_f_spec"] * N)))
        spec[rng.choice(N, n_spec, replace=False)] = True
        need = rng.random(N) < p["nb_f_need"]
        need |= spec                                  # Spezialisten brauchen es selbst
        s_b = np.where(spec, p["nb_s_spec"], 0.0)
        S = np.c_[S, s_b]
        W = np.c_[W, np.where(need, p["nb_weight"], 0.0)]
        phi_vec = np.r_[phi_vec, p["nb_phi"]]
    return S, W, phi_vec, spec, need


# =====================================================================
#  3. KERNMODELL
# =====================================================================
def evaluate(A, a, u, S, W, phi_vec, L, alpha, p, price=1.0):
    ai = 1.0 - np.exp(-alpha * u)                             # KI-Faehigkeit je Person
    Phi = phi_vec[None, :] * ai[:, None]                      # Ersetzbarkeit je Faehigkeit
    Uown = (1.0 - S) * (1.0 - Phi)
    SL = A.astype(float) @ L
    cov = np.maximum(1.0 - Uown * np.exp(SL), 1e-4)
    Wn = W / W.sum(axis=1, keepdims=True)
    amp = 1.0 + p["lam"] * alpha * u * a ** p["eta"]
    base = a * amp
    Y = base * np.exp((Wn * np.log(cov)).sum(axis=1))
    Y_alone = base * np.exp((Wn * np.log(np.maximum(1.0 - Uown, 1e-4))).sum(axis=1))
    deg = A.sum(axis=1)
    ceff = p["c0"] / (1.0 + p["kappa"] * alpha * u)
    # Robustheit: Kooperationskosten steigen mit der eigenen Produktivitaet
    # (Opportunitaetskosten der Zeit); cost_scale = 0 im Basismodell
    om = p.get("cost_scale", 0.0)
    if om:
        ceff = ceff * base ** om
    cost = ceff * (deg + p["gamma"] * deg * (deg - 1) / 2.0)
    return dict(phi=Phi[:, :p["K"]].mean(axis=1), Uown=Uown, SL=SL, Wn=Wn, base=base,
                Y=Y, Y_alone=Y_alone, deg=deg, ceff=ceff, cost=cost, price=price,
                income=price * Y - cost)


def _Y_with(st, idx, delta_log):
    cov = np.maximum(1.0 - st["Uown"][idx] * np.exp(st["SL"][idx] + delta_log), 1e-4)
    return st["base"][idx] * np.exp((st["Wn"][idx] * np.log(cov)).sum(axis=1))


def step(A, st, L, rng, p, p_rev):
    """Strategische Netzwerkdynamik.
    transfers=False: Kante nur bei beiderseitigem Vorteil, Aufloesung einseitig.
    transfers=True : Kante, wenn der GEMEINSAME Nutzen die gemeinsamen Kosten
                     uebersteigt (Ausgleichszahlungen / Honorar / Drittmittel)."""
    N = A.shape[0]
    gamma, P = p["gamma"], st["price"]
    tr = p.get("transfers", False)
    rev = np.flatnonzero(rng.random(N) < p_rev)
    if len(rev) == 0:
        return
    mc_drop = st["ceff"] * (1.0 + gamma * (st["deg"] - 1))
    r_idx, jj = np.nonzero(A[rev])
    if len(jj):
        ii = rev[r_idx]
        net = P * (st["Y"][ii] - _Y_with(st, ii, -L[jj])) - mc_drop[ii]
        if tr:
            net = net + P * (st["Y"][jj] - _Y_with(st, jj, -L[ii])) - mc_drop[jj]
        order = np.lexsort((net, ii))
        ii_s, jj_s, net_s = ii[order], jj[order], net[order]
        first = np.unique(ii_s, return_index=True)[1]
        sel = first[net_s[first] < 0]
        A[ii_s[sel], jj_s[sel]] = False
        A[jj_s[sel], ii_s[sel]] = False
    ci, cj = [], []
    for i in rev:
        j = -1
        if rng.random() < DYN["p_tri"]:
            nb = np.flatnonzero(A[i])
            if len(nb):
                k = nb[rng.integers(len(nb))]
                nb2 = np.flatnonzero(A[k])
                j = nb2[rng.integers(len(nb2))]
        if j < 0:
            j = rng.integers(N)
        if j != i and not A[i, j]:
            ci.append(i)
            cj.append(j)
    if ci:
        ci, cj = np.array(ci), np.array(cj)
        deg = A.sum(axis=1)
        net_i = P * (_Y_with(st, ci, L[cj]) - st["Y"][ci]) - st["ceff"][ci] * (1 + gamma * deg[ci])
        net_j = P * (_Y_with(st, cj, L[ci]) - st["Y"][cj]) - st["ceff"][cj] * (1 + gamma * deg[cj])
        ok = (net_i + net_j > 0) if tr else ((net_i > 0) & (net_j > 0))
        A[ci[ok], cj[ok]] = True
        A[cj[ok], ci[ok]] = True
    if DYN["eps"] > 0:
        E = np.triu(A, 1) & (rng.random((N, N)) < DYN["eps"])
        A[E] = False
        A[E.T] = False


def income_with_transfers(A, st, L):
    """Einkommen inkl. Ausgleichszahlungen: Der Netto-Ueberschuss jeder Kante
    wird haelftig geteilt (Nash-Verhandlungsloesung)."""
    N = A.shape[0]
    P = st["price"]
    inc = P * st["Y"] - st["cost"]
    ii, jj = np.nonzero(np.triu(A, 1))
    if len(ii) == 0:
        return inc, np.zeros(N)
    avg_c = st["cost"] / np.maximum(st["deg"], 1)
    v_i = P * (st["Y"][ii] - _Y_with(st, ii, -L[jj])) - avg_c[ii]
    v_j = P * (st["Y"][jj] - _Y_with(st, jj, -L[ii])) - avg_c[jj]
    tau = 0.5 * (v_j - v_i)                    # Zahlung von j an i
    T = np.bincount(ii, weights=tau, minlength=N) - np.bincount(jj, weights=tau, minlength=N)
    return inc + T, T


def paper_stats(A, st, L):
    N = A.shape[0]
    Y = st["Y"]
    led = Y.copy()
    ii, jj = np.nonzero(A)
    co_n = np.zeros(N)
    coauth = np.zeros(N)
    if len(ii):
        rel_loss = (Y[ii] - _Y_with(st, ii, -L[jj])) / Y[ii]
        sub = rel_loss > THETA_AUTHOR
        co_n = np.bincount(ii[sub], minlength=N).astype(float)
        coauth = np.bincount(jj[sub], weights=led[ii[sub]], minlength=N)
    total = led + coauth
    return dict(led=led, coauth=coauth, total=total, co_n=co_n,
                single_share=float(led[co_n == 0].sum() / led.sum()),
                authors_per_paper=float((led * (1 + co_n)).sum() / led.sum()))


# =====================================================================
#  5. KENNZAHLEN
# =====================================================================
def iso_mask(deg):
    return deg == 0


def gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None))
    n = len(x)
    if x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def topshare(x, frac):
    x = np.sort(np.clip(x, 0, None))[::-1]
    return float(x[:max(1, int(round(frac * len(x))))].sum() / max(x.sum(), 1e-12))


def component_sizes(A):
    N = A.shape[0]
    seen = np.zeros(N, bool)
    sizes = []
    for s in range(N):
        if seen[s]:
            continue
        stack, cnt = [s], 0
        seen[s] = True
        while stack:
            v = stack.pop()
            cnt += 1
            for w in np.flatnonzero(A[v] & ~seen):
                seen[w] = True
                stack.append(w)
        sizes.append(cnt)
    return np.array(sizes)


def metrics(A, st, a, u, L, spec, need, papers=False, tr=True):
    deg = st["deg"].astype(float)
    Y = st["Y"]
    N = len(Y)
    Af = A.astype(float)
    triples = (deg * (deg - 1)).sum()
    clust = float(np.trace(Af @ Af @ Af) / triples) if triples > 0 else 0.0
    sizes = component_sizes(A)
    iso = deg == 0
    qa = np.argsort(np.argsort(a)) * 5 // N
    inc, T = income_with_transfers(A, st, L)
    if not tr:                                   # ohne Ausgleichszahlungen keine Transfers
        inc, T = inc - T, np.zeros(N)
    n_edges = max(A.sum() / 2, 1)
    conn = ~iso_mask(deg)
    pov = inc < 0.6 * np.median(inc)             # relative Armut (60% des Medians)
    rev_tot = max((st["price"] * Y).sum(), 1e-12)
    m = dict(
        mittlerer_grad=deg.mean(),
        max_grad=deg.max(),
        gini_grad=gini(deg),
        anteil_isoliert=iso.mean(),
        clustering=clust,
        groesste_komponente=sizes.max() / N,
        gesamtoutput=Y.sum(),
        preis=st["price"],
        gini_output=gini(Y),
        gini_einkommen=gini(inc),
        top1_anteil=topshare(Y, 0.01),
        top10_anteil=topshare(Y, 0.10),
        abhaengigkeit=float(np.mean(1.0 - st["Y_alone"] / Y)),
        isoliert_ideen_q1=float(iso[qa == 0].mean()),
        isoliert_ideen_q5=float(iso[qa == 4].mean()),
        # Arbeitsmarkt-Linse: Rollen nach Einkommensquelle
        anteil_netto_empfaenger=float((conn & (T > 0)).mean()),   # "Angestellte/Dienstleister"
        anteil_netto_zahler=float((conn & (T <= 0)).mean()),      # "Auftraggeber/Gruender mit Team"
        anteil_solo=float(iso.mean()),                            # "Solo-Selbstaendige"
        transfervolumen_anteil=float(np.abs(T).sum() / 2 / rev_tot),  # ~ Lohnquote
        armutsquote=float(pov.mean()),
        abgehaengt=float((iso & pov).mean()),
        armut_ideen_q1=float(pov[qa == 0].mean()),
    )
    if spec.any():
        has_spec_nb = (A[:, spec].sum(axis=1) > 0) | spec
        nn = need & ~spec
        oth = ~spec
        m.update(
            grad_spezialisten=float(deg[spec].mean()),
            grad_andere=float(deg[oth].mean()),
            kantenanteil_spezialisten=float((A[spec].sum() - A[np.ix_(spec, spec)].sum() / 2) / n_edges),
            zugang_nadeloehr=float(has_spec_nb[nn].mean()) if nn.any() else np.nan,
            isoliert_beduerftige=float(iso[nn].mean()) if nn.any() else np.nan,
            isoliert_nicht_beduerftige=float(iso[~need].mean()) if (~need).any() else np.nan,
            output_beduerftige=float(Y[nn].mean()) if nn.any() else np.nan,
            output_nicht_beduerftige=float(Y[~need].mean()) if (~need).any() else np.nan,
            einkommen_spezialisten=float(inc[spec].mean()),
            einkommen_andere=float(inc[oth].mean()),
            transfer_an_spezialisten=float(T[spec].mean()),
            zugang_ideen_q1=float(has_spec_nb[nn & (qa == 0)].mean()) if (nn & (qa == 0)).any() else np.nan,
            zugang_ideen_q5=float(has_spec_nb[nn & (qa == 4)].mean()) if (nn & (qa == 4)).any() else np.nan,
        )
        # Nadeloehr-Rente: Median-Einkommen Spezialisten / Median-Einkommen andere (robust)
        m["rente_spezialisten"] = float(np.median(inc[spec]) / max(np.median(inc[oth]), 1e-9))
        # Anteil des Spezialisten-Einkommens, der aus Ausgleichszahlungen stammt
        m["transferanteil_spezialisten"] = float(T[spec].sum() / max(np.abs(inc[spec]).sum(), 1e-9))
    if papers:
        ps = paper_stats(A, st, L)
        m.update(einzelautor_anteil=ps["single_share"], autoren_pro_paper=ps["authors_per_paper"],
                 gini_papers=gini(ps["total"]))
    return m


def logistic_schedule(alpha_max, t_mid=None, tau=None):
    t_mid = T_AI * 0.35 if t_mid is None else t_mid
    tau = T_AI / 12 if tau is None else tau
    return lambda t: alpha_max / (1.0 + np.exp(-(t - t_mid) / tau))


def constant_schedule(alpha):
    return lambda t: alpha


def simulate(domain, alpha_sched, topology="WS", uptake="lognormal_1.0", rho=0.3,
             seed=0, overrides=None, record="final", record_every=1,
             papers=False, snapshot_times=()):
    N, T_burn, T_ai = N_AGENTS, T_BURN, T_AI
    p = dict(DOMAINS[domain])
    p.update(NB_DEFAULT)
    if overrides:
        p.update(overrides)
    rng = np.random.default_rng(seed)
    a, u = draw_agents(rng, N, p, uptake, rho)
    S, W, phi_vec, spec, need = make_skills(rng, N, p)
    L = np.log(1.0 - p["q"] * S)
    A = initial_network(rng, topology, N, DYN["k_init"])
    T = T_burn + T_ai
    rows, snaps = [], {}
    Yref, price, pre = None, 1.0, None
    for t in range(T):
        alpha = 0.0 if t < T_burn else float(alpha_sched(t - T_burn))
        st = evaluate(A, a, u, S, W, phi_vec, L, alpha, p, price)
        if t == T_burn - 1:
            Yref = st["Y"].sum()
            pre = dict(Y=st["Y"].copy(), deg=st["deg"].copy(),
                       income=income_with_transfers(A, st, L)[0])
            if papers:
                pre["papers"] = paper_stats(A, st, L)
        if (record == "all" and (t % record_every == 0 or t == T - 1)) or \
           (record == "final" and t >= T - 10):
            m = metrics(A, st, a, u, L, spec, need, papers, p.get("transfers", False))
            m.update(t=t - T_burn, alpha=alpha)
            rows.append(m)
        if t in snapshot_times:
            snaps[t - T_burn] = dict(A=A.copy(), Y=st["Y"].copy(), spec=spec.copy(), need=need.copy())
        step(A, st, L, rng, p, DYN["p_rev"])
        if Yref is not None and np.isfinite(p["eps_mkt"]):
            price = (st["Y"].sum() / Yref) ** (-1.0 / p["eps_mkt"])
    st = evaluate(A, a, u, S, W, phi_vec, L, alpha, p, price)
    inc, T_ = income_with_transfers(A, st, L)
    agents = pd.DataFrame(dict(ideen=a, ki_nutzung=u, spezialist=spec, beduerftig=need,
                               grad_vor=pre["deg"], grad=st["deg"], output_vor=pre["Y"],
                               output=st["Y"], einkommen_vor=pre["income"], einkommen=inc,
                               transfer=T_))
    if papers:
        ps = paper_stats(A, st, L)
        agents["papers_vor"] = pre["papers"]["total"]
        agents["papers"] = ps["total"]
        agents["erstautor_vor"] = pre["papers"]["led"]
        agents["erstautor"] = ps["led"]
        agents["koautor_vor"] = pre["papers"]["coauth"]
        agents["koautor"] = ps["coauth"]
        agents["koautoren_n_vor"] = pre["papers"]["co_n"]
        agents["koautoren_n"] = ps["co_n"]      # 0 = publiziert allein
    snaps["final"] = dict(A=A.copy(), alpha=alpha, price=price)
    df = pd.DataFrame(rows)
    if record == "final":
        df = df.drop(columns=["t"]).mean().to_frame().T
    return df, agents, snaps


# =====================================================================
#  7. GRAFIK-HILFEN
# =====================================================================
def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lorenz(x):
    x = np.sort(np.clip(x, 0, None))
    c = np.r_[0, np.cumsum(x)] / max(x.sum(), 1e-12)
    return np.linspace(0, 1, len(c)), c


def gic(pre, post, n=20):
    """Anonyme Wachstums-Inzidenz-Kurve: Verhaeltnis der Quantile post/pre."""
    qs = np.linspace(0.025, 0.975, n)
    return qs * 100, np.quantile(post, qs) / np.maximum(np.quantile(pre, qs), 1e-9)


def seeds(n, offset):
    return [MASTER_SEED + 1000 * offset + r for r in range(n)]



def spring_layout(A, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    pos = rng.random((N, 2))
    k = np.sqrt(1.0 / N)
    temp = 0.1
    Af = A.astype(float)
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
        f = ((k * k / dist ** 2 - Af * dist / k)[:, :, None] * d).sum(1)
        f -= 0.02 * (pos - 0.5)
        ln = np.linalg.norm(f, axis=1) + 1e-9
        pos += f / ln[:, None] * np.minimum(ln, temp)[:, None]
        temp *= 0.985
    return pos





# =====================================================================

# =====================================================================
#  8. THEORIE: MEAN-FIELD
# =====================================================================
from math import factorial


def comb(n, k):                     # (math.comb gibt es erst ab Python 3.8)
    return factorial(n) // (factorial(k) * factorial(n - k))


def mf_struct_logY(p, phi, d):
    """Erwarteter log-Strukturfaktor (geometrisches Mittel der Abdeckungen) einer
    Person mit KI-Substitution phi und d zufaelligen Partnern.
    Jeder Partner ist in einer gegebenen Faehigkeit mit Wahrscheinlichkeit
    pi = n_strong/K stark (s_hi), sonst schwach (s_lo)."""
    K, ns = p["K"], p["n_strong"]
    pi = ns / K
    r_hi, r_lo = 1 - p["q"] * p["s_hi"], 1 - p["q"] * p["s_lo"]
    U_s = (1 - p["s_hi"]) * (1 - phi)
    U_w = (1 - p["s_lo"]) * (1 - phi)
    Es = np.zeros_like(phi)
    Ew = np.zeros_like(phi)
    for b in range(d + 1):
        w = comb(d, b) * pi ** b * (1 - pi) ** (d - b)
        f = r_hi ** b * r_lo ** (d - b)
        Es += w * np.log(np.maximum(1 - U_s * f, 1e-4))
        Ew += w * np.log(np.maximum(1 - U_w * f, 1e-4))
    return (ns * Es + (K - ns) * Ew) / K


def mf_predict(p, a, u, alpha, Dmax=25):
    """Gewuenschter Grad d_i* je Person: groesstes d, fuer das jeder weitere Partner
    mehr bringt als er kostet (ohne Ausgleichszahlungen)."""
    phi = p["phi_cap"] * (1 - np.exp(-alpha * u))
    base = a * (1 + p["lam"] * alpha * u * a ** p["eta"])
    ceff = p["c0"] / (1 + p["kappa"] * alpha * u)
    Yprev = base * np.exp(mf_struct_logY(p, phi, 0))
    dstar = np.zeros(len(a), int)
    alive = np.ones(len(a), bool)
    for d in range(1, Dmax + 1):
        Yd = base * np.exp(mf_struct_logY(p, phi, d))
        ok = alive & ((Yd - Yprev) > ceff * (1 + p["gamma"] * (d - 1)))
        dstar[ok] = d
        alive = ok
        Yprev = Yd
        if not alive.any():
            break
    return dstar


def mf_isolation_closed_form(p, a, u, alpha):
    """Isolationsbedingung (erster Partner lohnt sich nicht):
       a_i * amp_i * Y0(phi_i) * (exp(Delta(phi_i)) - 1) < c_i"""
    phi = p["phi_cap"] * (1 - np.exp(-alpha * u))
    base = a * (1 + p["lam"] * alpha * u * a ** p["eta"])
    ceff = p["c0"] / (1 + p["kappa"] * alpha * u)
    l0 = mf_struct_logY(p, phi, 0)
    l1 = mf_struct_logY(p, phi, 1)
    return base * np.exp(l0) * (np.exp(l1 - l0) - 1) < ceff


LHS_RANGES = {            # Parameter: (min, max, typ)
    "c0": (0.01, 0.05, "f"), "gamma": (0.02, 0.30, "f"), "q": (0.3, 0.9, "f"),
    "phi_cap": (0.5, 1.0, "f"), "lam": (0.0, 1.5, "f"), "eta": (-1.0, 2.0, "f"),
    "kappa": (0.0, 2.0, "f"), "sigma_a": (0.2, 1.2, "f"), "K": (4, 10, "i"),
    "n_strong": (1, 3, "i"), "rho": (-0.3, 0.9, "f"), "transfers": (0, 1, "b"),
}
LHS_OUT = {"grad_verhaeltnis": "Grad(alpha=4) / Grad(0)", "isoliert_4": "Anteil Isolierter (alpha=4)",
           "gini_zuwachs": "Gini-Zuwachs Output", "transfer_verhaeltnis": "Transfervolumen(4) / (0)"}


def lhs(rng, n, ranges):
    X = {}
    for k, (lo, hi, typ) in ranges.items():
        v = (rng.permutation(n) + rng.random(n)) / n
        if typ == "f":
            X[k] = lo + v * (hi - lo)
        elif typ == "i":
            X[k] = np.floor(lo + v * (hi - lo + 1)).astype(int).clip(lo, hi)
        else:
            X[k] = v > 0.5
    return pd.DataFrame(X)


def prcc(X, y):
    """Partielle Rangkorrelation (PRCC) jedes Parameters mit dem Ergebnis."""
    R = X.rank().values.astype(float)
    ry = pd.Series(y).rank().values
    out = {}
    for j, name in enumerate(X.columns):
        others = np.c_[np.ones(len(R)), np.delete(R, j, axis=1)]
        bx = np.linalg.lstsq(others, R[:, j], rcond=None)[0]
        by = np.linalg.lstsq(others, ry, rcond=None)[0]
        ex, ey = R[:, j] - others @ bx, ry - others @ by
        out[name] = float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else 0.0
    return pd.Series(out)

# =====================================================================
#  9. CONFIGURATIONS USED IN THE PAPER
# =====================================================================
# Jede Zahl im Manuskript stammt aus genau einer dieser Konfigurationen.
CONFIGS = {
    "BASIS": dict(),                                   # eta=0, gegenseitige Zustimmung
    "WERT": dict(eta=1.0, transfers=True),             # Wert-, Arbeits- und Myerson-Analysen
    "NADELOEHR": dict(eta=1.0, transfers=True, nb_on=True),
}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
A_END = 6.0                    # Endpunkt der logistischen Diffusion und Standard-alpha
DOMS3 = ["Wissenschaft", "Wirtschaft"]   # the social domain is not reported in the paper
DOMS2 = ["Wissenschaft", "Wirtschaft"]


def tag(df, **kw):
    for k, v in kw.items():
        df[k] = v
    return df


def agent_setup(dom, seed, ov, uptake="lognormal_1.0", rho=0.3):
    """Erzeugt exakt dieselben Personen wie simulate() mit gleichem Seed."""
    p = dict(DOMAINS[dom]); p.update(NB_DEFAULT); p.update(ov)
    rng = np.random.default_rng(seed)
    a, u = draw_agents(rng, N_AGENTS, p, uptake, rho)
    S, W, phi_vec, spec, need = make_skills(rng, N_AGENTS, p)
    L = np.log(1.0 - p["q"] * S)
    return p, a, u, S, W, phi_vec, spec, need, L


def quint(x):
    return np.argsort(np.argsort(x)) * 5 // len(x)


# ---------------------------------------------------------------------
# Unabhaengigkeit vs. Ausschluss
# ---------------------------------------------------------------------
def classify_isolation(A, st, p, L, a, u, alpha):
    """Ordnet jede isolierte Person einem von vier Typen zu:
       'unabhaengig'    : KI-Selbstversorgung - kein Partner wuerde ihr mehr bringen als er kostet,
                          und das gaelte auch mit mittlerer Ideenqualitaet (gewaehlte Unabhaengigkeit);
       'abgehaengt'     : mit ihrer eigenen Ideenqualitaet lohnt sich kein Partner, mit mittlerer
                          Ideenqualitaet (gleiche KI-Nutzung) wuerde sie kooperieren (schwache Ideen
                          tragen die Kosten einer Kooperation nicht mehr);
       'ausgeschlossen' : sie wuerde von Partnern profitieren, aber keine Kante waere fuer beide
                          Seiten (bzw. gemeinsam, mit Transfers) lohnend;
       'suchend'        : eine lohnende Kante existiert, wurde aber noch nicht gefunden."""
    N = A.shape[0]
    P = st["price"]
    deg = A.sum(1)
    typ = np.full(N, "vernetzt", dtype=object)
    iso = np.flatnonzero(deg == 0)
    tr = p.get("transfers", False)
    a_med = float(np.median(a))
    for i in iso:
        js = np.delete(np.arange(N), i)
        ii = np.full(len(js), i)
        gross_i = P * (_Y_with(st, ii, L[js]) - st["Y"][i])
        gi = gross_i - st["ceff"][i]
        gj = P * (_Y_with(st, js, L[ii]) - st["Y"][js]) - st["ceff"][js] * (1 + p["gamma"] * deg[js])
        feas = (gi + gj > 0) if tr else ((gi > 0) & (gj > 0))
        if feas.any():
            typ[i] = "suchend"
        elif (gi > 0).any():
            typ[i] = "ausgeschlossen"
        else:
            base_cf = a_med * (1 + p["lam"] * alpha * u[i] * a_med ** p["eta"])
            gi_cf = gross_i * base_cf / st["base"][i] - st["ceff"][i]
            typ[i] = "abgehaengt" if (a[i] < a_med and (gi_cf > 0).any()) else "unabhaengig"
    dep = np.where(deg > 0, 1.0 - st["Y_alone"] / st["Y"], 0.0)
    return typ, dep


def decomposition_metrics(typ, dep, a, u):
    qa, qu = quint(a), quint(u)
    m = dict(anteil_unabhaengig=float((typ == "unabhaengig").mean()),
             anteil_abgehaengt=float((typ == "abgehaengt").mean()),
             anteil_ausgeschlossen=float((typ == "ausgeschlossen").mean()),
             anteil_suchend=float((typ == "suchend").mean()),
             abhaengigkeit_mittel=float(dep.mean()))
    for q in range(5):
        m[f"unabhaengig_ideen_q{q+1}"] = float((typ[qa == q] == "unabhaengig").mean())
        m[f"ausgeschlossen_ideen_q{q+1}"] = float((typ[qa == q] == "ausgeschlossen").mean())
        m[f"abgehaengt_ideen_q{q+1}"] = float((typ[qa == q] == "abgehaengt").mean())
        m[f"abhaengigkeit_ideen_q{q+1}"] = float(dep[qa == q].mean())
        m[f"unabhaengig_ki_q{q+1}"] = float((typ[qu == q] == "unabhaengig").mean())
        m[f"abhaengigkeit_ki_q{q+1}"] = float(dep[qu == q].mean())
        m[f"isoliert_typ_ideen_q{q+1}"] = float((typ[qa == q] != "vernetzt").mean())
        m[f"ausgeschlossen_ki_q{q+1}"] = float((typ[qu == q] == "ausgeschlossen").mean())
        m[f"abgehaengt_ki_q{q+1}"] = float((typ[qu == q] == "abgehaengt").mean())
    return m


def run_and_classify(dom, al, seed, ov, keep_agents=False):
    df, ag, sn = simulate(dom, constant_schedule(al), seed=seed, overrides=ov)
    p, a, u, S, W, phi_vec, spec, need, L = agent_setup(dom, seed, ov)
    A = sn["final"]["A"]
    st = evaluate(A, a, u, S, W, phi_vec, L, al, p, sn["final"]["price"])
    typ, dep = classify_isolation(A, st, p, L, a, u, al)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame([decomposition_metrics(typ, dep, a, u)])], axis=1)
    if keep_agents:
        ag = ag.copy()
        ag["typ"] = typ
        ag["abhaengigkeit"] = dep
    return df, ag


# =====================================================================
# P1  NETZWERKAUFLOESUNG + UNABHAENGIGKEIT/AUSSCHLUSS (Abb. 2, neue Abb.)
# =====================================================================
def P1_aufloesung(out):
    rows, agent_rows = [], []
    for d_i, dom in enumerate(DOMS3):
        for tr in ([False, True] if dom != "Sozial" else [False]):
            ov = dict(transfers=tr)
            for al in ALPHAS:
                for s in seeds(REPS, 10 + d_i):          # gleiche Seeds fuer alle alpha (gepaart)
                    keep = (al == A_END and s == seeds(1, 10 + d_i)[0])
                    df, ag = run_and_classify(dom, al, s, ov, keep_agents=True)
                    rows.append(tag(df, domaene=dom, transfers=tr, alpha_param=al, seed=s))
                    if al == A_END:
                        agent_rows.append(tag(ag, domaene=dom, transfers=tr, alpha_param=al, seed=s))
            print(f"   P1 {dom} (Transfers={tr}): fertig")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(out, "P1_aufloesung.csv"), index=False)
    pd.concat(agent_rows, ignore_index=True).to_csv(os.path.join(out, "P1_agenten_alpha6.csv"), index=False)

    # Koordination kappa x Substitution alpha (Wissenschaft)
    krows = []
    for kap in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        for al in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
            for s in seeds(max(3, REPS - 2), 13):
                df, _, _ = simulate("Wissenschaft", constant_schedule(al), seed=s, overrides=dict(kappa=kap))
                krows.append(tag(df, kappa=kap, alpha_param=al, seed=s))
    K = pd.concat(krows, ignore_index=True)
    K.to_csv(os.path.join(out, "P1_kappa_alpha.csv"), index=False)

    # Netzwerk-Schnappschuesse (Wissenschaft, logistische Diffusion bis alpha=6)
    s = seeds(1, 15)[0]
    times = (T_BURN - 1, T_BURN + int(round(0.35 * T_AI)), T_BURN + T_AI - 1)
    _, ag, sn = simulate("Wissenschaft", logistic_schedule(A_END), seed=s, snapshot_times=times)
    p, a, u, *_ = agent_setup("Wissenschaft", s, {})
    nodes, edges = [], []
    for t, snap in sorted((k, v) for k, v in sn.items() if k != "final"):
        A = snap["A"]
        nodes.append(pd.DataFrame(dict(t=t, id=np.arange(N_AGENTS), ki_nutzung=u, ideen=a,
                                       output=snap["Y"], grad=A.sum(1))))
        ii, jj = np.nonzero(np.triu(A, 1))
        edges.append(pd.DataFrame(dict(t=t, i=ii, j=jj)))
    pd.concat(nodes).to_csv(os.path.join(out, "P1_snapshot_knoten.csv"), index=False)
    pd.concat(edges).to_csv(os.path.join(out, "P1_snapshot_kanten.csv"), index=False)

    if HAS_MPL:
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
        b = res[~res.transfers.astype(bool) | (res.domaene == "Sozial")]
        for dom in DOMS3:
            g = b[b.domaene == dom].groupby("alpha_param")
            axs[0].plot(g.mittlerer_grad.mean(), "o-", color=COLORS[dom], label=dom)
            axs[1].plot(g.anteil_isoliert.mean() * 100, "o-", color=COLORS[dom], label=dom)
        for dom, ls in [("Wissenschaft", "-"), ("Wirtschaft", "--")]:
            g = b[b.domaene == dom].groupby("alpha_param")
            axs[2].plot(g.anteil_unabhaengig.mean() * 100, ls, color="#1a9e5c", label=f"{dom}: unabhaengig")
            axs[2].plot(g.anteil_abgehaengt.mean() * 100, ls, color="#e07b00", label=f"{dom}: abgehaengt")
            axs[2].plot(g.anteil_ausgeschlossen.mean() * 100, ls, color="#b3001b", label=f"{dom}: ausgeschlossen")
        agd = pd.read_csv(os.path.join(out, "P1_agenten_alpha6.csv"))
        agd = agd[(agd.domaene == "Wirtschaft") & (~agd.transfers.astype(bool))]
        agd["q"] = agd.groupby("seed").ideen.transform(lambda x: quint(x.values))
        tab = agd.groupby("q").typ.value_counts(normalize=True).unstack().fillna(0) * 100
        bottom = np.zeros(len(tab))
        for col, c in [("unabhaengig", "#1a9e5c"), ("abgehaengt", "#e07b00"), ("ausgeschlossen", "#b3001b"), ("suchend", "#999999")]:
            if col in tab:
                axs[3].bar(tab.index + 1, tab[col], bottom=bottom, color=c, label=col)
                bottom += tab[col].values
        axs[0].set_title("Mittlerer Grad", fontsize=9); axs[1].set_title("Isolierte (%)", fontsize=9)
        axs[2].set_title("Isolierte nach Typ (%)", fontsize=9)
        axs[3].set_title("Wirtschaft, alpha=6: Typ der Isolierten nach Ideenquintil", fontsize=9)
        for ax in axs:
            ax.legend(fontsize=7, frameon=False); style(ax)
        savefig(fig, os.path.join(out, "P1_aufloesung.png"))
    return res, K


# =====================================================================
# P2  UNGLEICHHEIT (Abb. 3)
# =====================================================================
def P2_ungleichheit(out):
    # a) Gewinnmatrix Ideen x KI-Nutzung (Wirtschaft, eta=1)
    ags = []
    for s in seeds(REPS, 20):
        _, ag, _ = simulate("Wirtschaft", constant_schedule(A_END), seed=s, overrides=dict(eta=1.0))
        ag["qa"], ag["qu"] = quint(ag.ideen.values) + 1, quint(ag.ki_nutzung.values) + 1
        ags.append(ag)
    ag = pd.concat(ags)
    ag["gewinn"] = ag.output / ag.output_vor
    G = ag.groupby(["qa", "qu"]).gewinn.median().unstack()
    G.to_csv(os.path.join(out, "P2_gewinnmatrix.csv"))
    # b,c) rho und eta
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for eta in [-1.0, 0.0, 1.0, 2.0]:
            for rho in [-0.5, 0.0, 0.3, 0.6, 0.9]:
                if eta in (-1.0, 2.0) and rho != 0.3:
                    continue
                for s in seeds(REPS, 21 + d_i):
                    df, _, _ = simulate(dom, constant_schedule(A_END), seed=s, rho=rho, overrides=dict(eta=eta))
                    rows.append(tag(df, domaene=dom, eta=eta, rho=rho, seed=s))
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(out, "P2_rho_eta.csv"), index=False)
    # d) Atlas der KI-Nutzung: 20 Verteilungen
    rows = []
    rg = np.random.default_rng(MASTER_SEED + 5)
    ginis = {k: gini(draw_uptake(rg, k, 100000)) for k in UPTAKE}
    for d_i, dom in enumerate(DOMS3):
        for k in UPTAKE:
            for s in seeds(max(3, REPS - 2), 24 + d_i):
                df, _, _ = simulate(dom, constant_schedule(A_END), seed=s, uptake=k)
                rows.append(tag(df, domaene=dom, verteilung=k, gini_ki=ginis[k], seed=s))
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(os.path.join(out, "P2_atlas.csv"), index=False)
    print("   P2: fertig")
    if HAS_MPL:
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
        axs[0].imshow(G.values, origin="lower", cmap="Blues")
        for (i, j), v in np.ndenumerate(G.values):
            axs[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
        axs[0].set_title("Median-Gewinn (Zeile Ideen, Spalte KI)", fontsize=9)
        for dom in DOMS2:
            for eta, ls in [(0.0, "--"), (1.0, "-")]:
                g = R[(R.domaene == dom) & (R.eta == eta)].groupby("rho").gini_output.mean()
                axs[1].plot(g, ls, marker="o", color=COLORS[dom], label=f"{dom}, eta={eta:g}")
            g = R[(R.domaene == dom) & (R.rho == 0.3)].groupby("eta").isoliert_ideen_q1.mean() * 100
            axs[2].plot(g, "o-", color=COLORS[dom], label=dom)
        for dom in DOMS3:
            g = D[D.domaene == dom].groupby("verteilung")[["gini_ki", "gini_output"]].mean()
            axs[3].scatter(g.gini_ki, g.gini_output, color=COLORS[dom], s=12, label=dom)
        axs[1].set_title("Gini Output vs rho", fontsize=9)
        axs[2].set_title("Isoliert im schwaechsten Ideenquintil (%) vs eta", fontsize=9)
        axs[3].set_title("Gini KI-Nutzung vs Gini Output", fontsize=9)
        for ax in axs[1:]:
            ax.legend(fontsize=7, frameon=False); style(ax)
        savefig(fig, os.path.join(out, "P2_ungleichheit.png"))
    return G, R, D


# =====================================================================
# P3  PUBLIKATIONEN (Abb. 4)
# =====================================================================
def P3_publikationen(out):
    global THETA_AUTHOR
    rows, ags = [], []
    for s in seeds(REPS, 30):
        df, ag, _ = simulate("Wissenschaft", logistic_schedule(A_END), seed=s, record="all", papers=True)
        rows.append(tag(df, seed=s))
        ags.append(tag(ag, seed=s))
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(out, "P3_publikationen_zeit.csv"), index=False)
    AG = pd.concat(ags, ignore_index=True)
    AG.to_csv(os.path.join(out, "P3_publikationen_agenten.csv"), index=False)
    # Wer publiziert allein? Anteil Solo-Autoren nach Ideen- x KI-Quintil (vor/nach)
    AG["qa"] = AG.groupby("seed").ideen.transform(lambda x: quint(x.values)) + 1
    AG["qu"] = AG.groupby("seed").ki_nutzung.transform(lambda x: quint(x.values)) + 1
    AG["solo_vor"] = (AG.koautoren_n_vor == 0).astype(float)
    AG["solo"] = (AG.koautoren_n == 0).astype(float)
    AG["isoliert"] = (AG.grad == 0).astype(float)
    AG["vernetzt_aber_solo"] = ((AG.grad > 0) & (AG.koautoren_n == 0)).astype(float)
    SM = AG.groupby(["qa", "qu"])[["solo_vor", "solo", "isoliert", "vernetzt_aber_solo", "grad"]].mean()
    SM.to_csv(os.path.join(out, "P3_solo_nach_quintilen.csv"))
    th_rows = []
    keep = THETA_AUTHOR
    for th in [0.01, 0.02, 0.05, 0.10, 0.20]:
        THETA_AUTHOR = th
        for s in seeds(max(3, REPS - 2), 31):
            df, _, _ = simulate("Wissenschaft", logistic_schedule(A_END), seed=s, record="all",
                                record_every=10, papers=True)
            b, e = df[df.t == 0].iloc[0], df.iloc[-1]
            th_rows.append(dict(theta=th, seed=s, einzelautor_vor=b.einzelautor_anteil,
                                einzelautor_nach=e.einzelautor_anteil, autoren_vor=b.autoren_pro_paper,
                                autoren_nach=e.autoren_pro_paper))
    THETA_AUTHOR = keep
    TH = pd.DataFrame(th_rows)
    TH.to_csv(os.path.join(out, "P3_theta_autorschaft.csv"), index=False)
    print("   P3: fertig")
    if HAS_MPL:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        g = R.groupby("t")
        axs[0].plot(g.einzelautor_anteil.mean() * 100, color="#2a6fdb")
        axs[0].set_title("Anteil Einzelautor-Papers (%)", fontsize=9)
        for col, c, lab in [("papers_vor", "#999999", "vor KI"), ("papers", "#2a6fdb", "nach KI")]:
            x, y = lorenz(AG[col].values)
            axs[1].plot(x, y, color=c, label=lab)
        axs[1].plot([0, 1], [0, 1], "k:", lw=0.6)
        axs[1].set_title("Lorenzkurven der Papers", fontsize=9)
        AG["q"] = AG.groupby("seed").ideen.transform(lambda x: quint(x.values)) + 1
        t = AG.groupby("q")[["erstautor_vor", "koautor_vor", "erstautor", "koautor"]].mean()
        w = 0.2
        for k, (col, c) in enumerate([("erstautor_vor", "#bbbbbb"), ("koautor_vor", "#777777"),
                                      ("erstautor", "#9ecae1"), ("koautor", "#2a6fdb")]):
            axs[2].bar(t.index + (k - 1.5) * w, t[col].clip(lower=1e-3), w, color=c, label=col)
        axs[2].set_yscale("log")
        axs[2].set_title("Papers pro Ideenquintil", fontsize=9)
        for ax in axs:
            ax.legend(fontsize=7, frameon=False); style(ax)
        savefig(fig, os.path.join(out, "P3_publikationen.png"))
    return R, AG, TH


# =====================================================================
# P4  NADELOEHR UND ARBEIT (Abb. 5a-c, Arbeits-Abschnitt)
# =====================================================================
def P4_nadeloehr(out):
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for al in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
            for s in seeds(REPS, 40 + d_i):
                df, _, _ = simulate(dom, constant_schedule(al), seed=s, overrides=CONFIGS["NADELOEHR"])
                rows.append(tag(df, domaene=dom, alpha_param=al, seed=s))
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(out, "P4_nadeloehr_alpha.csv"), index=False)
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for fs in [0.01, 0.02, 0.05, 0.10, 0.20]:
            for fn in [0.20, 0.40, 0.70]:
                for s in seeds(max(3, REPS - 2), 42 + d_i):
                    ov = dict(CONFIGS["NADELOEHR"], nb_f_spec=fs, nb_f_need=fn)
                    df, _, _ = simulate(dom, constant_schedule(A_END), seed=s, overrides=ov)
                    rows.append(tag(df, domaene=dom, f_spec=fs, f_need=fn, seed=s))
        for nbphi in [0.05, 0.5, 0.95]:
            for s in seeds(max(3, REPS - 2), 44 + d_i):
                ov = dict(CONFIGS["NADELOEHR"], nb_phi=nbphi)
                df, _, _ = simulate(dom, constant_schedule(A_END), seed=s, overrides=ov)
                rows.append(tag(df, domaene=dom, f_spec=0.05, f_need=0.40, nb_phi=nbphi, seed=s))
    H = pd.concat(rows, ignore_index=True)
    H.to_csv(os.path.join(out, "P4_nadeloehr_knappheit.csv"), index=False)
    # Schnappschuss alpha=6 (Wissenschaft)
    s = seeds(1, 46)[0]
    _, ag, sn = simulate("Wissenschaft", constant_schedule(A_END), seed=s, overrides=CONFIGS["NADELOEHR"])
    A = sn["final"]["A"]
    ii, jj = np.nonzero(np.triu(A, 1))
    pd.DataFrame(dict(i=ii, j=jj)).to_csv(os.path.join(out, "P4_snapshot_kanten.csv"), index=False)
    ag.assign(id=np.arange(len(ag)), grad_final=A.sum(1)).to_csv(
        os.path.join(out, "P4_snapshot_knoten.csv"), index=False)
    # Arbeit: Transfervolumen mit/ohne Marktsaettigung
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for eps in [np.inf, 1.5]:
            for al in [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]:
                for s in seeds(max(3, REPS - 2), 47 + d_i):
                    df, _, _ = simulate(dom, constant_schedule(al), seed=s,
                                        overrides=dict(CONFIGS["WERT"], eps_mkt=eps))
                    rows.append(tag(df, domaene=dom, eps_mkt=str(eps), alpha_param=al, seed=s))
    W = pd.concat(rows, ignore_index=True)
    W.to_csv(os.path.join(out, "P4_arbeit.csv"), index=False)
    print("   P4: fertig")
    if HAS_MPL:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for dom in DOMS2:
            g = R[R.domaene == dom].groupby("alpha_param")
            axs[0].plot(g.grad_spezialisten.mean(), "o-", color=COLORS[dom], label=f"{dom}: Spezialisten")
            axs[0].plot(g.grad_andere.mean(), "o--", color=COLORS[dom], label=f"{dom}: andere")
            for eps, ls in [("inf", "-"), ("1.5", "--")]:
                g2 = W[(W.domaene == dom) & (W.eps_mkt == eps)].groupby("alpha_param")
                axs[2].plot(g2.transfervolumen_anteil.mean() * 100, ls, marker="o", color=COLORS[dom],
                            label=f"{dom}, eps={eps}")
        h = H[(H.domaene == "Wissenschaft") & H.get("nb_phi", pd.Series(np.nan, index=H.index)).isna()]
        M = h.groupby(["f_need", "f_spec"]).rente_spezialisten.mean().unstack()
        axs[1].imshow(M.values, origin="lower", cmap="Blues")
        for (i, j), v in np.ndenumerate(M.values):
            axs[1].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
        axs[0].set_title("Grad: Spezialisten vs andere", fontsize=9)
        axs[1].set_title("Knappheitsrente Wissenschaft (Zeile f_need, Spalte f_spec)", fontsize=9)
        axs[2].set_title("Transfervolumen (% des Umsatzes)", fontsize=9)
        for ax in (axs[0], axs[2]):
            ax.legend(fontsize=7, frameon=False); style(ax)
        savefig(fig, os.path.join(out, "P4_nadeloehr.png"))
    return R, H, W


# =====================================================================
# P5  MYERSON-WERT: NETTO UND BRUTTO (Abb. 5d,e)
# =====================================================================
M_PERM = 40 if QUICK else 120


def myerson_value(A, a, u, S, W, phi_vec, L, alpha, p, M, rng):
    """Monte-Carlo-Myerson-Wert mit Zerlegung nach Empfaengern.
    Spiel v(C) = sum_{i in C} Y_i(A[C]). Weil der Shapley-Wert linear ist, zerfaellt er
    in die Spiele v_i(C) = 1[i in C] * Y_i(A[C]) je Empfaenger i. Beim Eintritt von j
    in eine Reihenfolge ist sein Grenzbeitrag = Y_j (eigener Output, Spiel v_j)
    + Zuwachs bei bereits anwesenden Nachbarn (Beitrag zu deren Spielen v_i).
    Rueckgabe: Shapley-Wert sh, Haelften h1/h2, Beitrag an andere (je Geber),
    von anderen erhalten (je Empfaenger)."""
    N = len(a)
    Phi = phi_vec[None, :] * (1 - np.exp(-alpha * u))[:, None]
    Uown = (1 - S) * (1 - Phi)
    Wn = W / W.sum(1, keepdims=True)
    base = a * (1 + p["lam"] * alpha * u * a ** p["eta"])

    def Yf(idx, SLr):
        cov = np.maximum(1 - Uown[idx] * np.exp(SLr), 1e-4)
        return base[idx] * np.exp((Wn[idx] * np.log(cov)).sum(-1))

    nbrs = [np.flatnonzero(A[i]) for i in range(N)]
    halves = [np.zeros(N), np.zeros(N)]
    given, received = np.zeros(N), np.zeros(N)
    for m in range(M):
        perm = rng.permutation(N)
        inS = np.zeros(N, bool)
        SL = np.zeros_like(S)
        Yc = np.zeros(N)
        val = halves[m % 2]
        for i in perm:
            nb = nbrs[i]
            SL[nb] += L[i]
            nbS = nb[inS[nb]]
            yi = Yf(i, SL[i])
            d = yi
            if len(nbS):
                yn = Yf(nbS, SL[nbS])
                inc = yn - Yc[nbS]
                d += inc.sum()
                given[i] += inc.sum()
                received[nbS] += inc
                Yc[nbS] = yn
            Yc[i] = yi
            inS[i] = True
            val[i] += d
    h1, h2 = halves[0] / ((M + 1) // 2), halves[1] / (M // 2)
    return (h1 + h2) / 2, h1, h2, given / M, received / M


def rank_corr(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def myerson_selftest():
    """Zwei symmetrische Partner: allein je 10, zusammen je 20 (Beispiel aus dem Review).
    Netto-Mass = 0, Brutto-Mass muss > 0 sein (je 5 von 20 stammen vom Partner)."""
    p = dict(lam=0.0, eta=0.0)
    S = np.array([[0.0, 0.9], [0.9, 0.0]])
    W = np.ones((2, 2))
    L = np.log(1 - 1.0 * S)
    phi = np.zeros(2)
    A = np.array([[False, True], [True, False]])
    sh, _, _, giv, rec = myerson_value(A, np.ones(2), np.zeros(2), S, W, phi, L, 0.0, p, 200,
                                       np.random.default_rng(0))
    Y_alone = np.exp(0.5 * np.log(np.maximum(1 - (1 - S), 1e-4)).sum(1))
    Y_full = np.exp(0.5 * (np.log(np.maximum(1 - (1 - S) * np.exp(A.astype(float) @ L), 1e-4))).sum(1))
    net = np.maximum(sh - Y_full, 0).sum() / Y_full.sum()
    gross = rec.sum() / Y_full.sum()
    return dict(Y_allein=Y_alone.round(4).tolist(), Y_gemeinsam=Y_full.round(4).tolist(),
                shapley=sh.round(4).tolist(), netto=round(float(net), 4), brutto=round(float(gross), 4))


def P5_myerson(out):
    alphas = [0.0, 2.0, 4.0, 6.0, 8.0]
    scen = {"ohne Nadeloehr": CONFIGS["WERT"], "mit Nadeloehr": CONFIGS["NADELOEHR"]}
    T = T_BURN + T_AI
    rows, agent_rows = [], []
    for d_i, dom in enumerate(DOMS2):
        for s_i, (sname, ov) in enumerate(scen.items()):
            for seed in seeds(max(3, REPS - 2), 50 + 10 * d_i + s_i):
                p, a, u, S, W, phi_vec, spec, need, L = agent_setup(dom, seed, ov)
                rng = np.random.default_rng(seed + 1)
                pre = None
                for al in alphas:
                    _, ag, sn = simulate(dom, constant_schedule(al), seed=seed, overrides=ov,
                                         snapshot_times=(T_BURN - 1,))
                    s0 = sn[-1]
                    A1 = sn["final"]["A"]
                    st1 = evaluate(A1, a, u, S, W, phi_vec, L, al, p)
                    if pre is None:
                        sh0, _, _, g0, r0 = myerson_value(s0["A"], a, u, S, W, phi_vec, L, 0.0, p, M_PERM, rng)
                        pre = dict(sh=sh0, Y=s0["Y"], k=s0["A"].sum(1), rec=r0)
                    sh, h1, h2, giv, rec = myerson_value(A1, a, u, S, W, phi_vec, L, al, p, M_PERM, rng)
                    Y, k = st1["Y"], A1.sum(1)
                    rel = sh / Y
                    inc = ag.einkommen.values
                    r = dict(domaene=dom, szenario=sname, alpha_param=al, seed=seed,
                             zuverlaessigkeit=rank_corr(h1, h2),
                             wert_von_anderen_netto=float(np.maximum(sh - Y, 0).sum() / Y.sum()),
                             wert_von_anderen_brutto=float(rec.sum() / Y.sum()),
                             wert_von_anderen_brutto_vor=float(pre["rec"].sum() / pre["Y"].sum()),
                             wert_von_anderen_netto_vor=float(np.maximum(pre["sh"] - pre["Y"], 0).sum() / pre["Y"].sum()),
                             unverzichtbarkeit_andere=float(np.median(rel[~spec])),
                             gini_shapley=gini(sh), gini_output=gini(Y),
                             korr_shapley_einkommen=rank_corr(sh, inc),
                             rangstabilitaet_shapley=rank_corr(pre["sh"], sh),
                             rangstabilitaet_grad=rank_corr(pre["k"], k),
                             rangstabilitaet_output=rank_corr(pre["Y"], Y))
                    if spec.any():
                        pct = pd.Series(sh).rank(pct=True).values
                        r.update(unverzichtbarkeit_spezialisten=float(np.median(rel[spec])),
                                 shapley_perzentil_spezialisten=float(np.median(pct[spec])),
                                 shapley_anteil_spezialisten=float(sh[spec].sum() / sh.sum()),
                                 beitrag_an_andere_anteil_spezialisten=float(giv[spec].sum() / max(giv.sum(), 1e-12)),
                                 output_anteil_spezialisten=float(Y[spec].sum() / Y.sum()),
                                 einkommen_anteil_spezialisten=float(inc[spec].sum() / inc.sum()))
                    rows.append(r)
                    if seed == seeds(1, 50 + 10 * d_i + s_i)[0] and al in (0.0, 8.0):
                        agent_rows.append(pd.DataFrame(dict(domaene=dom, szenario=sname, alpha_param=al,
                                                            shapley=sh, gegeben=giv, erhalten=rec,
                                                            output=Y, grad=k, spezialist=spec,
                                                            einkommen=inc, ideen=a, ki_nutzung=u)))
            print(f"   P5 {dom} / {sname}: fertig")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "P5_myerson.csv"), index=False)
    pd.concat(agent_rows, ignore_index=True).to_csv(os.path.join(out, "P5_myerson_agenten.csv"), index=False)
    if HAS_MPL:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for dom in DOMS2:
            for sname, ls in [("ohne Nadeloehr", "--"), ("mit Nadeloehr", "-")]:
                g = res[(res.domaene == dom) & (res.szenario == sname)].groupby("alpha_param")
                axs[0].plot(g.wert_von_anderen_netto.mean() * 100, ls, marker="o", color=COLORS[dom],
                            label=f"{dom}, {sname}")
                axs[1].plot(g.wert_von_anderen_brutto.mean() * 100, ls, marker="o", color=COLORS[dom],
                            label=f"{dom}, {sname}")
            g = res[(res.domaene == dom) & (res.szenario == "mit Nadeloehr")].groupby("alpha_param")
            axs[2].plot(g.shapley_anteil_spezialisten.mean() * 100, "o-", color=COLORS[dom], label=f"{dom}: Shapley")
            axs[2].plot(g.einkommen_anteil_spezialisten.mean() * 100, "s--", color=COLORS[dom], label=f"{dom}: Einkommen")
        axs[0].set_title("Wert von anderen: NETTO (%)", fontsize=9)
        axs[1].set_title("Wert von anderen: BRUTTO (%)", fontsize=9)
        axs[2].set_title("Spezialisten: Anteile (%)", fontsize=9)
        for ax in axs:
            ax.legend(fontsize=7, frameon=False); style(ax)
        savefig(fig, os.path.join(out, "P5_myerson.png"))
    return res


# =====================================================================
# P6  THEORIE: ISOLATIONSFENSTER, PHASENDIAGRAMME, SENSITIVITAET (Abb. 6)
# =====================================================================
def mf_first_gain(p, a, u, alpha):
    """Netto-Zugewinn des ersten Partners (Mean-Field), absolute Skala."""
    phi = p["phi_cap"] * (1 - np.exp(-alpha * u))
    base = a * (1 + p["lam"] * alpha * u * a ** p["eta"])
    ceff = p["c0"] / (1 + p["kappa"] * alpha * u)
    l0 = mf_struct_logY(p, phi, 0)
    l1 = mf_struct_logY(p, phi, 1)
    return base * np.exp(l0) * (np.exp(l1 - l0) - 1) - ceff


def trajectory_types(G, grid, a_max=8.0):
    """G: (Personen x alpha-Gitter) Netto-Zugewinn. Klassifiziert jede Person."""
    neg = G < 0
    n = G.shape[0]
    typ = np.empty(n, dtype=object)
    a_in = np.full(n, np.nan)
    a_out = np.full(n, np.nan)
    for i in range(n):
        idx = np.flatnonzero(neg[i])
        if len(idx) == 0:
            typ[i] = "nie isoliert"
            continue
        a_in[i] = grid[idx[0]]
        later = np.flatnonzero(~neg[i, idx[0]:])
        if len(later):
            a_out[i] = grid[idx[0] + later[0]]
        if idx[0] == 0:
            typ[i] = "schon ohne KI isoliert"
        elif np.isnan(a_out[i]):
            typ[i] = "ein Kipppunkt (kein Wiedereintritt bis alpha_max)"
        elif a_out[i] <= a_max:
            typ[i] = "Isolationsfenster innerhalb [0, 8]"
        else:
            typ[i] = "Wiedereintritt erst oberhalb 8"
    return typ, a_in, a_out


def P6_theorie(out, P1res):
    # (1) Theorie vs Simulation, mit Abstand in Prozentpunkten
    trows = []
    for d_i, dom in enumerate(DOMS2):
        for al in ALPHAS:
            for s in seeds(REPS, 10 + d_i):
                p, a, u, *_ = agent_setup(dom, s, {})
                ds = mf_predict(p, a, u, al)
                trows.append(dict(domaene=dom, alpha_param=al, seed=s,
                                  theorie_isoliert=float((ds == 0).mean()),
                                  theorie_grad_gegenseitig=float(np.minimum(ds[:, None], ds[None, :]).mean())))
    TH = pd.DataFrame(trows)
    sim = P1res[P1res.domaene.isin(DOMS2)].groupby(["domaene", "transfers", "alpha_param"]).anteil_isoliert.mean()
    sim = sim.unstack("transfers")
    sim.columns = ["sim_ohne_transfers", "sim_mit_transfers"]
    th = TH.groupby(["domaene", "alpha_param"]).theorie_isoliert.mean()
    V = pd.concat([th, sim], axis=1).reset_index()
    V["abstand_pp_ohne"] = (V.theorie_isoliert - V.sim_ohne_transfers) * 100
    V["abstand_pp_mit"] = (V.theorie_isoliert - V.sim_mit_transfers) * 100
    V.to_csv(os.path.join(out, "P6_theorie_vs_simulation.csv"), index=False)

    # (2) Verlaeufe je Person ueber ein weites alpha-Gitter: Isolationsfenster
    grid = np.round(np.arange(0, 40.001, 0.1), 2)
    trajs, summ = [], []
    for d_i, dom in enumerate(DOMS2):
        for lam_mult in [None, 0.0]:
            Gs, As, Us = [], [], []
            for s in seeds(REPS, 10 + d_i):
                ov = {} if lam_mult is None else dict(lam=0.0)
                p, a, u, *_ = agent_setup(dom, s, ov)
                Gs.append(np.stack([mf_first_gain(p, a, u, al) for al in grid], axis=1))
                As.append(a); Us.append(u)
            G, a, u = np.vstack(Gs), np.concatenate(As), np.concatenate(Us)
            typ, a_in, a_out = trajectory_types(G, grid)
            lab = "Basis" if lam_mult is None else "ohne Amplifikation (lam=0)"
            vc = pd.Series(typ).value_counts(normalize=True)
            for k, v in vc.items():
                summ.append(dict(domaene=dom, variante=lab, typ=k, anteil=float(v)))
            if lam_mult is None:
                trajs.append(pd.DataFrame(dict(domaene=dom, ideen=a, ki_nutzung=u, typ=typ,
                                               alpha_ein=a_in, alpha_aus=a_out)))
    pd.DataFrame(summ).to_csv(os.path.join(out, "P6_verlaufstypen.csv"), index=False)
    TR = pd.concat(trajs, ignore_index=True)
    TR.to_csv(os.path.join(out, "P6_verlaeufe_personen.csv"), index=False)

    # (3) Phasendiagramme: alpha x Kosten und alpha x Amplifikation
    prow = []
    agr = np.linspace(0, 8, 41)
    for d_i, dom in enumerate(DOMS2):
        p0, a, u, *_ = agent_setup(dom, seeds(1, 10 + d_i)[0], {})
        for cm in np.geomspace(0.25, 4, 33):
            p = dict(p0, c0=p0["c0"] * cm)
            G = np.stack([mf_first_gain(p, a, u, al) for al in agr], axis=1)
            iso = G < 0
            ever = np.maximum.accumulate(iso, axis=1)
            for k, al in enumerate(agr):
                prow.append(dict(domaene=dom, achse="kosten", wert=float(cm), alpha=float(al),
                                 isoliert=float(iso[:, k].mean()),
                                 zurueckgekehrt=float((ever[:, k] & ~iso[:, k]).mean())))
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            p = dict(p0, lam=lam)
            G = np.stack([mf_first_gain(p, a, u, al) for al in agr], axis=1)
            iso = G < 0
            ever = np.maximum.accumulate(iso, axis=1)
            for k, al in enumerate(agr):
                prow.append(dict(domaene=dom, achse="amplifikation", wert=float(lam), alpha=float(al),
                                 isoliert=float(iso[:, k].mean()),
                                 zurueckgekehrt=float((ever[:, k] & ~iso[:, k]).mean())))
    PH = pd.DataFrame(prow)
    PH.to_csv(os.path.join(out, "P6_phasendiagramme.csv"), index=False)

    # (4) Simulation jenseits alpha=8: kehrt die Isolation um, wie die Theorie sagt?
    hrows = []
    for d_i, dom in enumerate(DOMS2):
        for al in [8.0, 12.0, 16.0, 24.0, 32.0]:
            for s in seeds(max(3, REPS - 2), 10 + d_i):
                df, _, _ = simulate(dom, constant_schedule(al), seed=s)
                p, a, u, *_ = agent_setup(dom, s, {})
                ds = mf_predict(p, a, u, al)
                hrows.append(tag(df, domaene=dom, alpha_param=al, seed=s,
                                 theorie_isoliert=float((ds == 0).mean())))
    HI = pd.concat(hrows, ignore_index=True)
    HI.to_csv(os.path.join(out, "P6_hohes_alpha.csv"), index=False)

    # (5) Globale Sensitivitaet (Latin Hypercube, PRCC), Basiskonfiguration
    n = 30 if QUICK else 160
    rng = np.random.default_rng(MASTER_SEED + 7)
    X = lhs(rng, n, LHS_RANGES)
    X["n_strong"] = np.minimum(X.n_strong, X.K - 1)
    rows = []
    for i, row in X.iterrows():
        ov = {k: (bool(v) if k == "transfers" else (int(v) if k in ("K", "n_strong") else float(v)))
              for k, v in row.items() if k != "rho"}
        res = {}
        for al in [0.0, 4.0]:
            df, _, _ = simulate("Wissenschaft", constant_schedule(al), seed=MASTER_SEED + 9000 + i,
                                overrides=ov, rho=float(row.rho))
            res[al] = df.iloc[0]
        rows.append(dict(grad_verhaeltnis=res[4.0].mittlerer_grad / max(res[0.0].mittlerer_grad, 1e-9),
                         isoliert_4=res[4.0].anteil_isoliert,
                         gini_zuwachs=res[4.0].gini_output - res[0.0].gini_output))
        if (i + 1) % 40 == 0:
            print(f"   P6 Sensitivitaet: {i+1}/{n}")
    Y = pd.DataFrame(rows)
    pd.concat([X, Y], axis=1).to_csv(os.path.join(out, "P6_lhs_stichprobe.csv"), index=False)
    PR = pd.DataFrame({k: prcc(X.astype(float), Y[k].values) for k in Y.columns})
    PR.to_csv(os.path.join(out, "P6_prcc.csv"))
    print("   P6: fertig")
    if HAS_MPL:
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
        for dom in DOMS2:
            v = V[V.domaene == dom]
            axs[0].plot(v.alpha_param, v.theorie_isoliert * 100, "-", color=COLORS[dom], label=f"{dom}: Theorie")
            axs[0].plot(v.alpha_param, v.sim_ohne_transfers * 100, "o", color=COLORS[dom], label=f"{dom}: Sim.")
            h = HI[HI.domaene == dom].groupby("alpha_param")
            axs[1].plot(h.anteil_isoliert.mean() * 100, "o-", color=COLORS[dom], label=f"{dom}: Sim.")
            axs[1].plot(h.theorie_isoliert.mean() * 100, "--", color=COLORS[dom], label=f"{dom}: Theorie")
        ph = PH[(PH.domaene == "Wissenschaft") & (PH.achse == "kosten")]
        M = ph.pivot(index="wert", columns="alpha", values="isoliert")
        axs[2].imshow(M.values, origin="lower", aspect="auto", cmap="Blues",
                      extent=[0, 8, np.log2(0.25), np.log2(4)])
        R2 = ph.pivot(index="wert", columns="alpha", values="zurueckgekehrt")
        axs[2].contour(np.linspace(0, 8, R2.shape[1]), np.log2(R2.index.values), R2.values,
                       levels=[0.02, 0.1], colors="#b3001b", linewidths=0.8)
        axs[2].set_title("Wiss.: isoliert (blau), zurueckgekehrt (rot: 2%, 10%)", fontsize=9)
        axs[2].set_ylabel("log2 Kostenfaktor", fontsize=8)
        s = PR["isoliert_4"].sort_values()
        axs[3].barh(range(len(s)), s.values, color=np.where(s.values > 0, "#b3001b", "#2a6fdb"))
        axs[3].set_yticks(range(len(s))); axs[3].set_yticklabels(s.index, fontsize=7)
        axs[3].set_title("PRCC: Anteil Isolierter (alpha=4)", fontsize=9)
        axs[0].set_title("Theorie vs Simulation (%)", fontsize=9)
        axs[1].set_title("Hohe alpha: Rueckkehr ins Netzwerk? (%)", fontsize=9)
        for ax in axs[:2]:
            ax.legend(fontsize=7, frameon=False)
        for ax in axs:
            style(ax)
        savefig(fig, os.path.join(out, "P6_theorie.png"))
    return V, TR, PH, HI, PR


# =====================================================================
# P8  ROBUSTHEIT DER SELEKTION + POLITIKHEBEL BEZAHLTE KOOPERATION
# =====================================================================
def P8_robustheit(out):
    """Haelt die Selektion nach Ideenqualitaet, wenn Kooperationskosten mit der eigenen
    Produktivitaet steigen (omega = 0, 0.5, 1)? Und wie stark verhindern Ausgleichszahlungen
    (bezahlte Zusammenarbeit: Honorare, Drittmittel, Dienstleistungsvertraege) die Isolation?"""
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for om in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for tr in [False, True]:
                for al in [0.0, 2.0, 4.0, 6.0, 8.0]:
                    for s in seeds(max(3, REPS - 2), 80 + d_i):
                        ov = dict(cost_scale=om, transfers=tr)
                        df, ag = run_and_classify(dom, al, s, ov, keep_agents=True)
                        qa, qu = quint(ag.ideen.values), quint(ag.ki_nutzung.values)
                        iso = (ag.grad.values == 0)
                        extra = dict(iso_ideen_q1_ki_q5=float(iso[(qa == 0) & (qu == 4)].mean()) if ((qa == 0) & (qu == 4)).any() else np.nan,
                                     iso_ideen_q5_ki_q5=float(iso[(qa == 4) & (qu == 4)].mean()) if ((qa == 4) & (qu == 4)).any() else np.nan,
                                     iso_ideen_q1=float(iso[qa == 0].mean()), iso_ideen_q5=float(iso[qa == 4].mean()),
                                     korr_ideen_isoliert=float(np.corrcoef(pd.Series(ag.ideen).rank(), iso)[0, 1]) if iso.any() else 0.0)
                        df = pd.concat([df, pd.DataFrame([extra])], axis=1)
                        rows.append(tag(df, domaene=dom, kosten_skalierung=om, transfers=tr, alpha_param=al, seed=s))
            print(f"   P8 {dom}, omega={om}: fertig")
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(out, "P8_robustheit_selektion.csv"), index=False)
    if HAS_MPL:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for k, dom in enumerate(DOMS2):
            for om, ls in [(0.0, "-"), (0.5, "--"), (1.0, ":")]:
                for tr, c in [(False, "#b3001b"), (True, "#1a9e5c")]:
                    g = R[(R.domaene == dom) & (R.kosten_skalierung == om) & (R.transfers == tr)].groupby("alpha_param")
                    axs[k].plot(g.iso_ideen_q1.mean() * 100, ls, color=c, marker="o", ms=3,
                                label=f"omega={om:g}, {'mit' if tr else 'ohne'} Transfers")
            axs[k].set_title(f"{dom}: isoliert im schwaechsten Ideenquintil (%)", fontsize=9)
            axs[k].legend(fontsize=6, frameon=False); style(axs[k])
        savefig(fig, os.path.join(out, "P8_robustheit.png"))
    print("   P8: fertig")
    return R


# =====================================================================
# P7  SCHLUESSELZAHLEN: jede Zahl des Manuskripts mit Quelle
# =====================================================================
def P7_schluesselzahlen(out, P1res, K, G, R2, D, R3, AG3, TH3, R4, H4, W4, M5, V6, TR6, HI6, PR6, R8):
    K_ = []

    def add(name, value, quelle, konfig, notiz=""):
        try:
            value = float(value)
        except Exception:
            value = np.nan
        K_.append(dict(kennzahl=name, wert=value, quelle=quelle, konfiguration=konfig, notiz=notiz))

    b = P1res[~P1res.transfers.astype(bool)]
    for dom in DOMS3:
        g = b[b.domaene == dom].groupby("alpha_param").mean(numeric_only=True)
        add(f"{dom}: Grad alpha=0", g.mittlerer_grad[0.0], "P1_aufloesung.csv", "BASIS")
        add(f"{dom}: Grad alpha=8", g.mittlerer_grad[8.0], "P1_aufloesung.csv", "BASIS")
        add(f"{dom}: Grad alpha=4", g.mittlerer_grad[4.0], "P1_aufloesung.csv", "BASIS")
        add(f"{dom}: isoliert alpha=6 (%)", 100 * g.anteil_isoliert[6.0], "P1_aufloesung.csv", "BASIS")
        add(f"{dom}: isoliert alpha=8 (%)", 100 * g.anteil_isoliert[8.0], "P1_aufloesung.csv", "BASIS")
        add(f"{dom}: groesste Komponente alpha=8 (%)", 100 * g.groesste_komponente[8.0], "P1_aufloesung.csv", "BASIS")
        if dom != "Sozial":
            add(f"{dom}: davon unabhaengig alpha=6 (% aller)", 100 * g.anteil_unabhaengig[6.0], "P1_aufloesung.csv", "BASIS")
            add(f"{dom}: davon abgehaengt alpha=6 (% aller)", 100 * g.anteil_abgehaengt[6.0], "P1_aufloesung.csv", "BASIS")
            add(f"{dom}: davon ausgeschlossen alpha=6 (% aller)", 100 * g.anteil_ausgeschlossen[6.0], "P1_aufloesung.csv", "BASIS")
            add(f"{dom}: davon suchend alpha=6 (% aller)", 100 * g.anteil_suchend[6.0], "P1_aufloesung.csv", "BASIS")
            add(f"{dom}: Abhaengigkeit von Partnern alpha=0 (%)", 100 * g.abhaengigkeit_mittel[0.0], "P1_aufloesung.csv", "BASIS")
            add(f"{dom}: Abhaengigkeit von Partnern alpha=6 (%)", 100 * g.abhaengigkeit_mittel[6.0], "P1_aufloesung.csv", "BASIS")
            for q in (1, 5):
                add(f"{dom}: Abhaengigkeit KI-Q{q} alpha=0 (%)", 100 * g[f"abhaengigkeit_ki_q{q}"][0.0], "P1_aufloesung.csv", "BASIS")
                add(f"{dom}: Abhaengigkeit KI-Q{q} alpha=6 (%)", 100 * g[f"abhaengigkeit_ki_q{q}"][6.0], "P1_aufloesung.csv", "BASIS")
                add(f"{dom}: Abhaengigkeit Ideen-Q{q} alpha=6 (%)", 100 * g[f"abhaengigkeit_ideen_q{q}"][6.0], "P1_aufloesung.csv", "BASIS")
                add(f"{dom}: isoliert Ideen-Q{q} alpha=6 (%)", 100 * g[f"isoliert_typ_ideen_q{q}"][6.0], "P1_aufloesung.csv", "BASIS")
                add(f"{dom}: unabhaengig Ideen-Q{q} alpha=6 (%)", 100 * g[f"unabhaengig_ideen_q{q}"][6.0], "P1_aufloesung.csv", "BASIS")
                add(f"{dom}: abgehaengt Ideen-Q{q} alpha=6 (%)", 100 * g[f"abgehaengt_ideen_q{q}"][6.0], "P1_aufloesung.csv", "BASIS")
                add(f"{dom}: ausgeschlossen Ideen-Q{q} alpha=6 (%)", 100 * g[f"ausgeschlossen_ideen_q{q}"][6.0], "P1_aufloesung.csv", "BASIS")
            gt = P1res[P1res.transfers.astype(bool) & (P1res.domaene == dom)].groupby("alpha_param").mean(numeric_only=True)
            add(f"{dom}: isoliert alpha=6 mit Transfers (%)", 100 * gt.anteil_isoliert[6.0], "P1_aufloesung.csv", "BASIS+Transfers")
    k = K.groupby(["kappa", "alpha_param"]).mittlerer_grad.mean()
    add("Wiss.: Grad kappa=4, alpha=8", k[(4.0, 8.0)], "P1_kappa_alpha.csv", "BASIS")
    add("Wiss.: Grad kappa=0, alpha=8", k[(0.0, 8.0)], "P1_kappa_alpha.csv", "BASIS")
    add("Wirtschaft eta=1: Median-Gewinn Q5/Q5", G.loc[5, 5], "P2_gewinnmatrix.csv", "BASIS, eta=1")
    add("Wirtschaft eta=1: Median-Gewinn Q1/Q1", G.loc[1, 1], "P2_gewinnmatrix.csv", "BASIS, eta=1")
    for dom in DOMS2:
        r = R2[(R2.domaene == dom) & (R2.rho == 0.3)].groupby("eta").mean(numeric_only=True)
        for eta in [-1.0, 0.0, 1.0, 2.0]:
            add(f"{dom}: isoliert schwaechstes Ideenquintil, eta={eta:g} (%)", 100 * r.isoliert_ideen_q1[eta], "P2_rho_eta.csv", "BASIS")
            add(f"{dom}: Gini Output, eta={eta:g}", r.gini_output[eta], "P2_rho_eta.csv", "BASIS")
    for dom in DOMS3:
        d = D[D.domaene == dom].groupby("verteilung")[["gini_ki", "gini_output"]].mean()
        add(f"{dom}: Korrelation Gini KI vs Gini Output (20 Verteilungen)", np.corrcoef(d.gini_ki, d.gini_output)[0, 1], "P2_atlas.csv", "BASIS")
    g3 = R3.groupby("t").mean(numeric_only=True)
    add("Einzelautor-Anteil vor KI (%)", 100 * g3.einzelautor_anteil[0], "P3_publikationen_zeit.csv", "BASIS, logistisch bis 6")
    add("Einzelautor-Anteil Ende (%)", 100 * g3.einzelautor_anteil.iloc[-1], "P3_publikationen_zeit.csv", "BASIS, logistisch bis 6",
        "dieselbe Zahl in Text UND Abb. 4 verwenden")
    add("Autoren pro Paper vor KI", g3.autoren_pro_paper[0], "P3_publikationen_zeit.csv", "BASIS")
    add("Autoren pro Paper Ende", g3.autoren_pro_paper.iloc[-1], "P3_publikationen_zeit.csv", "BASIS")
    add("Gini Papers vor KI", g3.gini_papers[0], "P3_publikationen_zeit.csv", "BASIS")
    add("Gini Papers Ende", g3.gini_papers.iloc[-1], "P3_publikationen_zeit.csv", "BASIS")
    sm = pd.read_csv(os.path.join(out, "P3_solo_nach_quintilen.csv")).set_index(["qa", "qu"])
    for qa_, qu_ in [(5, 5), (1, 1), (5, 1), (1, 5)]:
        add(f"Ideen-Q{qa_}/KI-Q{qu_}: publiziert allein, vor KI (%)", 100 * sm.solo_vor[(qa_, qu_)], "P3_solo_nach_quintilen.csv", "BASIS")
        add(f"Ideen-Q{qa_}/KI-Q{qu_}: publiziert allein, Ende (%)", 100 * sm.solo[(qa_, qu_)], "P3_solo_nach_quintilen.csv", "BASIS")
        add(f"Ideen-Q{qa_}/KI-Q{qu_}: vernetzt, aber allein publizierend (%)", 100 * sm.vernetzt_aber_solo[(qa_, qu_)], "P3_solo_nach_quintilen.csv", "BASIS")
        add(f"Ideen-Q{qa_}/KI-Q{qu_}: isoliert (%)", 100 * sm.isoliert[(qa_, qu_)], "P3_solo_nach_quintilen.csv", "BASIS")
    t = TH3.groupby("theta").mean(numeric_only=True)
    add("Einzelautor Ende, theta=1% (%)", 100 * t.einzelautor_nach[0.01], "P3_theta_autorschaft.csv", "BASIS")
    add("Einzelautor Ende, theta=20% (%)", 100 * t.einzelautor_nach[0.20], "P3_theta_autorschaft.csv", "BASIS")
    for dom in DOMS2:
        r = R4[R4.domaene == dom].groupby("alpha_param").mean(numeric_only=True)
        add(f"{dom}: Grad Spezialisten alpha=0", r.grad_spezialisten[0.0], "P4_nadeloehr_alpha.csv", "NADELOEHR")
        add(f"{dom}: Grad Spezialisten alpha=8", r.grad_spezialisten[8.0], "P4_nadeloehr_alpha.csv", "NADELOEHR")
        add(f"{dom}: Grad andere alpha=0", r.grad_andere[0.0], "P4_nadeloehr_alpha.csv", "NADELOEHR")
        add(f"{dom}: Grad andere alpha=8", r.grad_andere[8.0], "P4_nadeloehr_alpha.csv", "NADELOEHR")
        add(f"{dom}: Kantenanteil Spezialisten alpha=0 (%)", 100 * r.kantenanteil_spezialisten[0.0], "P4_nadeloehr_alpha.csv", "NADELOEHR")
        add(f"{dom}: Kantenanteil Spezialisten alpha=8 (%)", 100 * r.kantenanteil_spezialisten[8.0], "P4_nadeloehr_alpha.csv", "NADELOEHR")
        h = H4[(H4.domaene == dom)]
        hh = h[h.get("nb_phi", pd.Series(np.nan, index=h.index)).isna()].groupby(["f_spec", "f_need"]).mean(numeric_only=True)
        add(f"{dom}: Rente f_spec=5%, f_need=40%", hh.rente_spezialisten[(0.05, 0.40)], "P4_nadeloehr_knappheit.csv", "NADELOEHR, alpha=6")
        add(f"{dom}: Rente f_spec=1%, f_need=70%", hh.rente_spezialisten[(0.01, 0.70)], "P4_nadeloehr_knappheit.csv", "NADELOEHR, alpha=6")
        add(f"{dom}: Zugang Beduerftige f_spec=1%, f_need=70% (%)", 100 * hh.zugang_nadeloehr[(0.01, 0.70)], "P4_nadeloehr_knappheit.csv", "NADELOEHR, alpha=6")
        if "nb_phi" in h:
            hp = h[h.nb_phi.notna()].groupby("nb_phi").mean(numeric_only=True)
            add(f"{dom}: Grad Spezialisten nb_phi=0.95", hp.grad_spezialisten[0.95], "P4_nadeloehr_knappheit.csv", "NADELOEHR, alpha=6")
            add(f"{dom}: Grad Spezialisten nb_phi=0.05", hp.grad_spezialisten[0.05], "P4_nadeloehr_knappheit.csv", "NADELOEHR, alpha=6")
        for eps in ["inf", "1.5"]:
            w = W4[(W4.domaene == dom) & (W4.eps_mkt == eps)].groupby("alpha_param").mean(numeric_only=True)
            add(f"{dom}: Transfervolumen alpha=0, eps={eps} (%)", 100 * w.transfervolumen_anteil[0.0], "P4_arbeit.csv", "WERT")
            add(f"{dom}: Transfervolumen alpha=8, eps={eps} (%)", 100 * w.transfervolumen_anteil[8.0], "P4_arbeit.csv", "WERT")
            add(f"{dom}: Solo-Anteil alpha=8, eps={eps} (%)", 100 * w.anteil_solo[8.0], "P4_arbeit.csv", "WERT")
    for dom in DOMS2:
        for sname in ["ohne Nadeloehr", "mit Nadeloehr"]:
            m = M5[(M5.domaene == dom) & (M5.szenario == sname)].groupby("alpha_param").mean(numeric_only=True)
            cfg = "WERT" if sname == "ohne Nadeloehr" else "NADELOEHR"
            add(f"{dom}/{sname}: Wert von anderen NETTO alpha=0 (%)", 100 * m.wert_von_anderen_netto[0.0], "P5_myerson.csv", cfg)
            add(f"{dom}/{sname}: Wert von anderen NETTO alpha=8 (%)", 100 * m.wert_von_anderen_netto[8.0], "P5_myerson.csv", cfg)
            add(f"{dom}/{sname}: Wert von anderen BRUTTO alpha=0 (%)", 100 * m.wert_von_anderen_brutto[0.0], "P5_myerson.csv", cfg)
            add(f"{dom}/{sname}: Wert von anderen BRUTTO alpha=8 (%)", 100 * m.wert_von_anderen_brutto[8.0], "P5_myerson.csv", cfg)
            add(f"{dom}/{sname}: Zuverlaessigkeit (split-half)", m.zuverlaessigkeit.mean(), "P5_myerson.csv", cfg)
            if sname == "mit Nadeloehr":
                add(f"{dom}: Shapley-Anteil Spezialisten alpha=8 (%)", 100 * m.shapley_anteil_spezialisten[8.0], "P5_myerson.csv", cfg)
                add(f"{dom}: Einkommensanteil Spezialisten alpha=8 (%)", 100 * m.einkommen_anteil_spezialisten[8.0], "P5_myerson.csv", cfg)
                add(f"{dom}: Output-Anteil Spezialisten alpha=8 (%)", 100 * m.output_anteil_spezialisten[8.0], "P5_myerson.csv", cfg)
                add(f"{dom}: Shapley-Perzentil Spezialisten alpha=0", m.shapley_perzentil_spezialisten[0.0], "P5_myerson.csv", cfg)
                add(f"{dom}: Shapley-Perzentil Spezialisten alpha=8", m.shapley_perzentil_spezialisten[8.0], "P5_myerson.csv", cfg)
                add(f"{dom}: Unverzichtbarkeit Spezialisten alpha=8", m.unverzichtbarkeit_spezialisten[8.0], "P5_myerson.csv", cfg)
                add(f"{dom}: Unverzichtbarkeit andere alpha=8", m.unverzichtbarkeit_andere[8.0], "P5_myerson.csv", cfg)
    for dom in DOMS2:
        v = V6[V6.domaene == dom]
        add(f"{dom}: Korrelation Theorie-Simulation", np.corrcoef(v.theorie_isoliert, v.sim_ohne_transfers)[0, 1], "P6_theorie_vs_simulation.csv", "BASIS")
        add(f"{dom}: mittlerer Abstand Theorie-Sim. (pp)", v.abstand_pp_ohne.mean(), "P6_theorie_vs_simulation.csv", "BASIS")
        add(f"{dom}: maximaler Abstand Theorie-Sim. (pp)", v.abstand_pp_ohne.max(), "P6_theorie_vs_simulation.csv", "BASIS")
        add(f"{dom}: Abstand bei alpha=8 (pp)", v[v.alpha_param == 8.0].abstand_pp_ohne.iloc[0], "P6_theorie_vs_simulation.csv", "BASIS")
        tr = TR6[TR6.domaene == dom]
        vc = tr.typ.value_counts(normalize=True)
        for typ_name, val in vc.items():
            add(f"{dom}: Verlaufstyp '{typ_name}' (%)", 100 * val, "P6_verlaeufe_personen.csv", "BASIS")
        w = tr[tr.alpha_aus.notna() & (tr.alpha_ein > 0)]
        if len(w):
            add(f"{dom}: Median alpha Eintritt Isolation", w.alpha_ein.median(), "P6_verlaeufe_personen.csv", "BASIS")
            add(f"{dom}: Median alpha Wiedereintritt", w.alpha_aus.median(), "P6_verlaeufe_personen.csv", "BASIS")
        h = HI6[HI6.domaene == dom].groupby("alpha_param").mean(numeric_only=True)
        for al in h.index:
            add(f"{dom}: Simulation isoliert alpha={al:g} (%)", 100 * h.anteil_isoliert[al], "P6_hohes_alpha.csv", "BASIS")
            add(f"{dom}: Theorie isoliert alpha={al:g} (%)", 100 * h.theorie_isoliert[al], "P6_hohes_alpha.csv", "BASIS")
    for out_k in PR6.columns:
        for par, val in PR6[out_k].items():
            add(f"PRCC {out_k} ~ {par}", val, "P6_prcc.csv", "BASIS (LHS)")
    for dom in DOMS2:
        for om in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for tr in [False, True]:
                r = R8[(R8.domaene == dom) & (R8.kosten_skalierung == om) & (R8.transfers == tr)].groupby("alpha_param").mean(numeric_only=True)
                lab = f"{dom}, omega={om:g}, {'mit' if tr else 'ohne'} Transfers"
                add(f"{lab}: isoliert alpha=6 (%)", 100 * r.anteil_isoliert[6.0], "P8_robustheit_selektion.csv", "BASIS + Kostenskalierung")
                add(f"{lab}: isoliert Ideen-Q1 alpha=6 (%)", 100 * r.iso_ideen_q1[6.0], "P8_robustheit_selektion.csv", "BASIS + Kostenskalierung")
                add(f"{lab}: isoliert Ideen-Q5 alpha=6 (%)", 100 * r.iso_ideen_q5[6.0], "P8_robustheit_selektion.csv", "BASIS + Kostenskalierung")
                add(f"{lab}: isoliert Ideen-Q1 & KI-Q5 alpha=6 (%)", 100 * r.iso_ideen_q1_ki_q5[6.0], "P8_robustheit_selektion.csv", "BASIS + Kostenskalierung")
                add(f"{lab}: Rangkorr. Ideen ~ isoliert alpha=6", r.korr_ideen_isoliert[6.0], "P8_robustheit_selektion.csv", "BASIS + Kostenskalierung")
                add(f"{lab}: Abhaengigkeit alpha=6 (%)", 100 * r.abhaengigkeit_mittel[6.0], "P8_robustheit_selektion.csv", "BASIS + Kostenskalierung")
    KZ = pd.DataFrame(K_)
    KZ.to_csv(os.path.join(out, "SCHLUESSELZAHLEN.csv"), index=False)
    return KZ




# =====================================================================
# Q1-Q3: SELECTION, CHANNEL ABLATION AND RECONNECTION
#   Q1 Selektion: Ideen vs. KI-Nutzung (rho = 0 / 0.3, Schichtung nach KI-Nutzung)
#   Q2 Faktorielle Abschaltung der drei KI-Kanaele
#   Q3 Theorie: individuelle Kooperationsbereitschaft vs. tatsaechliche Rueckkehr
# Same model and configurations as P1-P8.
# =====================================================================
def terciles(x):
    return np.argsort(np.argsort(x)) * 3 // len(x)


def Q1_selektion(out):
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for rho in [0.0, 0.3]:
            for om in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for tr in [False, True]:
                    for s in seeds(REPS, 90 + d_i):
                        df, ag, _ = simulate(dom, constant_schedule(A_END), seed=s, rho=rho,
                                             overrides=dict(cost_scale=om, transfers=tr))
                        a, u, iso = ag.ideen.values, ag.ki_nutzung.values, ag.grad.values == 0
                        qa, qu, tu = quint(a), quint(u), terciles(u)
                        r = dict(domaene=dom, rho=rho, kosten_skalierung=om, transfers=tr, seed=s,
                                 isoliert=float(iso.mean()),
                                 korr_ideen=float(np.corrcoef(pd.Series(a).rank(), iso)[0, 1]) if iso.any() else 0.0,
                                 korr_ki=float(np.corrcoef(pd.Series(u).rank(), iso)[0, 1]) if iso.any() else 0.0)
                        for q in range(5):
                            r[f"iso_ideen_q{q+1}"] = float(iso[qa == q].mean())
                            r[f"iso_ki_q{q+1}"] = float(iso[qu == q].mean())
                        for t in range(3):
                            for q in (0, 4):
                                m = (qa == q) & (tu == t)
                                r[f"iso_ideen_q{q+1}_ki_t{t+1}"] = float(iso[m].mean()) if m.any() else np.nan
                        # Partielle Korrelationen (Rang): Isolation ~ Ideen | KI-Nutzung und umgekehrt
                        R = np.c_[pd.Series(a).rank(), pd.Series(u).rank(), iso.astype(float)]
                        if iso.any() and iso.std() > 0:
                            pc = prcc(pd.DataFrame(R[:, :2], columns=["ideen", "ki"]), R[:, 2])
                            r["pkorr_ideen"], r["pkorr_ki"] = float(pc["ideen"]), float(pc["ki"])
                        else:
                            r["pkorr_ideen"] = r["pkorr_ki"] = 0.0
                        rows.append(r)
            print(f"   Q1 {dom}, rho={rho}: fertig")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "Q1_selektion.csv"), index=False)
    return R


ABLATION = {  # Kanal: (aus, an)
    "Substitution": ("phi_cap", 0.0, None),
    "Amplifikation": ("lam", 0.0, None),
    "Koordination": ("kappa", 0.0, 1.0),
}


def Q2_kanaele(out):
    """2 x 2 x 2: Substitution (phi_cap = 0 / Standard), Amplifikation (lambda = 0 / Standard),
    Koordination (kappa = 0 / 1). alpha = 6, verglichen mit alpha = 0 (identische Personen und Seeds)."""
    rows = []
    for d_i, dom in enumerate(DOMS2):
        base = dict(DOMAINS[dom])
        for cfg_name, cfg in [("BASIS", CONFIGS["BASIS"]), ("NADELOEHR", CONFIGS["NADELOEHR"])]:
            for S_on in (0, 1):
                for A_on in (0, 1):
                    for C_on in (0, 1):
                        ov = dict(cfg)
                        ov["phi_cap"] = base["phi_cap"] if S_on else 0.0
                        ov["lam"] = base["lam"] if A_on else 0.0
                        ov["kappa"] = 1.0 if C_on else 0.0
                        for s in seeds(REPS, 100 + d_i):
                            r0, _, _ = simulate(dom, constant_schedule(0.0), seed=s, overrides=ov)
                            r1, _, _ = simulate(dom, constant_schedule(A_END), seed=s, overrides=ov)
                            r0, r1 = r0.iloc[0], r1.iloc[0]
                            row = dict(domaene=dom, konfiguration=cfg_name, substitution=S_on, amplifikation=A_on,
                                       koordination=C_on, seed=s,
                                       grad_verhaeltnis=r1.mittlerer_grad / max(r0.mittlerer_grad, 1e-9),
                                       isoliert=r1.anteil_isoliert, abhaengigkeit_0=r0.abhaengigkeit,
                                       abhaengigkeit=r1.abhaengigkeit, gini_zuwachs=r1.gini_output - r0.gini_output)
                            if cfg_name == "NADELOEHR":
                                row.update(kantenanteil_spez_0=r0.kantenanteil_spezialisten,
                                           kantenanteil_spez=r1.kantenanteil_spezialisten,
                                           grad_spez_verhaeltnis=r1.grad_spezialisten / max(r0.grad_spezialisten, 1e-9))
                            rows.append(row)
            print(f"   Q2 {dom}, {cfg_name}: fertig")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "Q2_kanaele.csv"), index=False)
    return R


def Q3_rueckkehr(out):
    """Gleiche Personen ueber alle alpha (identische Seeds): Wer ist in der SIMULATION bei einem alpha
    isoliert und bei einem hoeheren wieder vernetzt? Vergleich mit der individuellen
    Kooperationsbereitschaft der Mean-Field-Theorie. Zusatz: lambda = 0 mit kappa > 0."""
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    rows, srows = [], []
    for d_i, dom in enumerate(DOMS2):
        for s in seeds(max(3, REPS - 2), 10 + d_i):
            p, a, u, *_ = agent_setup(dom, s, {})
            G = np.stack([mf_first_gain(p, a, u, al) for al in grid], axis=1)
            Ds = []
            for al in grid:
                _, ag, _ = simulate(dom, constant_schedule(al), seed=s)
                Ds.append(ag.grad.values)
            D = np.stack(Ds, axis=1)
            iso_sim, iso_th = D == 0, G < 0
            ever_sim = np.maximum.accumulate(iso_sim, axis=1)
            ever_th = np.maximum.accumulate(iso_th, axis=1)
            for k, al in enumerate(grid):
                rows.append(dict(domaene=dom, seed=s, alpha=al,
                                 sim_isoliert=float(iso_sim[:, k].mean()), theorie_unwillig=float(iso_th[:, k].mean()),
                                 sim_zurueck=float((ever_sim[:, k] & ~iso_sim[:, k]).mean()),
                                 theorie_wieder_willig=float((ever_th[:, k] & ~iso_th[:, k]).mean()),
                                 sim_jemals=float(ever_sim[:, k].mean()), theorie_jemals=float(ever_th[:, k].mean()),
                                 uebereinstimmung=float((iso_sim[:, k] == iso_th[:, k]).mean())))
        # Theorie-Varianten: welche Kanaele erlauben Rueckkehr?
        p0, a, u, *_ = agent_setup(dom, seeds(1, 10 + d_i)[0], {})
        g2 = np.round(np.arange(0, 40.001, 0.1), 2)
        for lam, kap in [(p0["lam"], 0.0), (0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (p0["lam"], 1.0)]:
            p = dict(p0, lam=lam, kappa=kap)
            G = np.stack([mf_first_gain(p, a, u, al) for al in g2], axis=1)
            typ, a_in, a_out = trajectory_types(G, g2, a_max=40.0)
            vc = pd.Series(typ).value_counts(normalize=True)
            ever = (np.isfinite(a_in) & (a_in > 0)).sum()
            back = (np.isfinite(a_out) & np.isfinite(a_in) & (a_in > 0)).sum()
            srows.append(dict(domaene=dom, lam=lam, kappa=kap, jemals_unwillig=float(ever / len(a)),
                              davon_wieder_willig=float(back / max(ever, 1)),
                              ein_kipppunkt=float(vc.get("ein Kipppunkt (kein Wiedereintritt bis alpha_max)", 0.0))))
        print(f"   Q3 {dom}: fertig")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "Q3_rueckkehr_simulation.csv"), index=False)
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(out, "Q3_theorie_kanaele.csv"), index=False)
    return R, S




# =====================================================================
# SUPPLEMENTARY EXPERIMENTS
# =====================================================================
def S_A_zeitreihen(out):
    rows = []
    for d_i, dom in enumerate(DOMS3):
        for s in seeds(REPS, 60 + d_i):
            df, _, _ = simulate(dom, logistic_schedule(A_END), seed=s, record="all",
                                     papers=(dom == "Wissenschaft"))
            df["domaene"], df["seed"] = dom, s
            rows.append(df)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(out, "SA_zeitreihen.csv"), index=False)
    print("   S_A fertig")


def S_B_topologien(out):
    rows = []
    keep = DYN["p_rev"]
    for p_rev, lab in [(0.5, "flexibel"), (0.05, "starr")]:
        DYN["p_rev"] = p_rev
        for d_i, dom in enumerate(DOMS2):
            for topo in TOPOLOGIES:
                for al in [0.0, 6.0]:
                    for s in seeds(max(3, REPS - 2), 70 + d_i):
                        df, _, _ = simulate(dom, constant_schedule(al), topology=topo, seed=s)
                        df["domaene"], df["topologie"], df["netz"], df["alpha_param"], df["seed"] = dom, topo, lab, al, s
                        rows.append(df)
        print(f"   S_B {lab}: fertig")
    DYN["p_rev"] = keep
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(os.path.join(out, "SB_topologien.csv"), index=False)


def S_C_reziprozitaet(out):
    """Fuer jede Kante: Nutzen b = Output-Verlust ohne den Partner, Kosten c = Grenzkosten der Kante.
    Ohtsuki-Nowak-Regel fuer Netzwerk-Reziprozitaet: Kooperation setzt sich durch, wenn b/c > k."""
    rows = []
    for d_i, dom in enumerate(DOMS2):
        for al in [0.0, 2.0, 4.0, 6.0, 8.0]:
            for s in seeds(max(3, REPS - 2), 10 + d_i):
                _, _, sn = simulate(dom, constant_schedule(al), seed=s)
                p, a, u, S, W, phi_vec, spec, need, L = agent_setup(dom, s, {})
                A = sn["final"]["A"]
                st = evaluate(A, a, u, S, W, phi_vec, L, al, p)
                ii, jj = np.nonzero(A)
                if len(ii) == 0:
                    continue
                b = st["Y"][ii] - _Y_with(st, ii, -L[jj])
                c = st["ceff"][ii] * (1 + p["gamma"] * (st["deg"][ii] - 1))
                k = A.sum(1)
                rows.append(dict(domaene=dom, alpha_param=al, seed=s, b_c_median=float(np.median(b / c)),
                                 b_c_mittel=float(np.mean(b / c)), grad_mittel=float(k[k > 0].mean()),
                                 anteil_kanten_bc_gt_k=float(np.mean(b / c > k[ii]))))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "SC_reziprozitaet.csv"), index=False)
    print("   S_C fertig")




def main():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    out = os.path.join(here, "results")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    t0 = time.time()
    st = myerson_selftest()
    clean = lambda d: {k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}
    with open(os.path.join(out, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(dict(quick=QUICK, N=N_AGENTS, T_burn=T_BURN, T_ai=T_AI, reps=REPS, m_perm=M_PERM,
                       seed=MASTER_SEED, alpha_end=A_END, alphas=ALPHAS, configurations=CONFIGS,
                       bottleneck_defaults=NB_DEFAULT, theta_author=THETA_AUTHOR,
                       domains={d: clean(p) for d, p in DOMAINS.items() if d in DOMS3}, dynamics=DYN,
                       myerson_selftest=st, python=sys.version, numpy=np.__version__,
                       pandas=pd.__version__), f, indent=2)
    steps = []
    def step(name):
        print(f"[{len(steps)+1}] {name} ... ({(time.time()-t0)/60:.1f} min)")
        steps.append(name)
    step("P1 network dissolution, independence and exit");      P1res, K = P1_aufloesung(out)
    step("P2 inequality and uptake atlas");                     G, R2, D = P2_ungleichheit(out)
    step("P3 publications");                                    R3, AG3, TH3 = P3_publikationen(out)
    step("P4 bottleneck skills and labour");                    R4, H4, W4 = P4_nadeloehr(out)
    step("P5 Myerson value");                                   M5 = P5_myerson(out)
    step("P6 theory and global sensitivity");                   V6, TR6, PH6, HI6, PR6 = P6_theorie(out, P1res)
    step("P8 cost scaling and paid collaboration");             R8 = P8_robustheit(out)
    step("Q1 selection on ideas vs AI use");                    Q1_selektion(out)
    step("Q2 channel ablation");                                Q2_kanaele(out)
    step("Q3 willingness vs actual return");                    Q3_rueckkehr(out)
    step("S_A-S_C supplementary experiments");                  S_A_zeitreihen(out); S_B_topologien(out); S_C_reziprozitaet(out)
    step("P7 table of reported numbers")
    P7_schluesselzahlen(out, P1res, K, G, R2, D, R3, AG3, TH3, R4, H4, W4, M5, V6, TR6, HI6, PR6, R8)
    print(f"Finished after {(time.time()-t0)/60:.1f} min. Results in {out}")


if __name__ == "__main__":
    main()
