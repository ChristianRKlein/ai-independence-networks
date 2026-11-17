# -*- coding: utf-8 -*-
"""
=====================================================================
  KI-Unabhaengigkeit und Netzwerke  --  V6: EMPIRISCHE PRUEFUNG
=====================================================================

Dieses Programm laedt echte Daten aus dem Internet und prueft zwei
Vorhersagen unseres Modells. Es braucht nur numpy, pandas, matplotlib
und eine Internetverbindung.

  WISSENSCHAFT (OpenAlex, ~250 Mio. Publikationen, 26 Fachgebiete)
    Vorhersage 1: Seit der KI-Verbreitung (ab 2023) steigt der Anteil an
      Einzelautor-Papers staerker als der Vor-KI-Trend erwarten laesst -
      aber v.a. in Faechern, deren Ko-Autoren-Arbeit KI-ersetzbar ist
      (rechnerisch/theoretisch). In Faechern mit physischen Nadeloehren
      (Labor, Klinik) ist der Effekt schwach.
  WIRTSCHAFT (US Census Business Formation Statistics, direkt vom Census Bureau)
    Vorhersage 2: Der Anteil von Gruendungen MIT geplanten Angestellten
      sinkt nach 2023 in wissensintensiven, KI-exponierten Branchen
      staerker als in physischen Branchen.

!!! VOR DEM START (einmalig, ca. 1 Minute): !!!
  OpenAlex verlangt seit Februar 2026 einen kostenlosen API-Schluessel:
    1. Auf https://openalex.org ein kostenloses Konto anlegen
    2. Unter https://openalex.org/settings/api den Schluessel kopieren
    3. Unten bei OPENALEX_API_KEY zwischen die Anfuehrungszeichen einfuegen
  Ohne Schluessel laeuft nur der Wirtschaftsteil (Census braucht keinen).

Dann mit F5 starten. Laufzeit ca. 3-6 Minuten (Downloads).
Ergebnis: results_V6.zip im selben Ordner -> bitte in den Chat laden.
"""

OPENALEX_API_KEY = ""          # optional: free OpenAlex API key

import os
import sys
import json
import time
import shutil
import zipfile
import io
import urllib.request
import urllib.parse
import urllib.error

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

VERSION = "V6"
YEARS = (2010, 2026)
PRE = (2012, 2022)            # Vor-KI-Trendperiode Wissenschaft (ChatGPT: Nov. 2022)
POST = (2023, 2025)           # Nach-KI-Periode (2026 noch unvollstaendig indexiert)
PRE_BFS = (2012, 2019)        # Vor-Trend Wirtschaft (ohne Corona-Sonderjahre 2020-2022)
POST_BFS = (2023, 2026)
N_BOOT = 2000

# ---------------------------------------------------------------
# A-PRIORI-KLASSIFIKATION der OpenAlex-Fachgebiete (vor Sicht der Daten
# festgelegt): Ist die Arbeit von Ko-Autoren ueberwiegend KI-ersetzbar
# (rechnerisch/theoretisch/textbasiert) oder an physische Nadeloehre
# gebunden (Labor, Klinik, Geraete, Feldarbeit)?
# ---------------------------------------------------------------
FIELD_CLASS = {
    "Computer Science": "rechnerisch", "Mathematics": "rechnerisch",
    "Economics, Econometrics and Finance": "rechnerisch", "Decision Sciences": "rechnerisch",
    "Business, Management and Accounting": "rechnerisch", "Social Sciences": "rechnerisch",
    "Arts and Humanities": "rechnerisch", "Psychology": "gemischt",
    "Physics and Astronomy": "gemischt", "Engineering": "gemischt", "Energy": "gemischt",
    "Earth and Planetary Sciences": "gemischt", "Environmental Science": "gemischt",
    "Materials Science": "physisch", "Chemistry": "physisch", "Chemical Engineering": "physisch",
    "Biochemistry, Genetics and Molecular Biology": "physisch", "Medicine": "physisch",
    "Immunology and Microbiology": "physisch", "Neuroscience": "physisch",
    "Pharmacology, Toxicology and Pharmaceutics": "physisch", "Agricultural and Biological Sciences": "physisch",
    "Veterinary": "physisch", "Dentistry": "physisch", "Nursing": "physisch", "Health Professions": "physisch",
}
CLASS_COL = {"rechnerisch": "#2a6fdb", "gemischt": "#999999", "physisch": "#b3001b"}

# US-Branchen (NAICS) fuer die Gruendungsstatistik
BFS_SECTORS = {
    "TOTAL": ("Alle Branchen", "gesamt"),
    "NAICS51": ("Information", "wissensintensiv"),
    "NAICS54": ("Freiberufl./wiss./techn. Dienste", "wissensintensiv"),
    "NAICS52": ("Finanzen & Versicherung", "wissensintensiv"),
    "NAICS23": ("Bau", "physisch"),
    "NAICS72": ("Gastgewerbe", "physisch"),
    "NAICS62": ("Gesundheit & Soziales", "physisch"),
    "NAICS81": ("Sonstige Dienstleistungen", "physisch"),
}
BFS_COL = {"wissensintensiv": "#2a6fdb", "physisch": "#b3001b", "gesamt": "#555555"}


# =====================================================================
#  HTTP-HILFEN
# =====================================================================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KI-Netzwerk-Studie/1.0 (Wissenschaft; nicht-kommerziell)"


def http_bytes(url, tries=3, timeout=45):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404):
                raise
            time.sleep(2 + 3 * k)
        except Exception as e:      # Zeitueberschreitung, Verbindungsabbruch
            last = e
            time.sleep(2 + 3 * k)
    raise last


def http_get(url, tries=3, timeout=45):
    return http_bytes(url, tries, timeout).decode("utf-8")


# =====================================================================
#  TEIL A: OPENALEX
# =====================================================================
OA = "https://api.openalex.org"


def oa_url(path, params):
    params = dict(params)
    params["api_key"] = OPENALEX_API_KEY
    return OA + path + "?" + urllib.parse.urlencode(params, safe=":,|>/<-")


def oa_fields():
    js = json.loads(http_get(oa_url("/fields", {"per_page": 50})))
    return [(f["id"].replace("https://openalex.org/", ""), f["display_name"]) for f in js["results"]]


def oa_counts_by_year(extra_filter):
    flt = "publication_year:%d-%d,type:article,%s" % (YEARS[0], YEARS[1], extra_filter)
    js = json.loads(http_get(oa_url("/works", {"filter": flt, "group_by": "publication_year",
                                               "per_page": 200})))
    return {int(g["key"]): int(g["count"]) for g in js["group_by"]}


BINS = {"gesamt": "authors_count:>0", "a1": "authors_count:1", "a2": "authors_count:2",
        "a3": "authors_count:3", "a10plus": "authors_count:>9"}


def openalex_download(out):
    fields = oa_fields()
    print("   %d Fachgebiete gefunden" % len(fields))
    rows = []
    targets = [("ALLE", "Alle Fachgebiete", "")] + [(fid, name, "primary_topic.field.id:" + fid)
                                                     for fid, name in fields]
    for n, (fid, name, ff) in enumerate(targets, 1):
        rec = {}
        for b, f in BINS.items():
            flt = f if not ff else f + "," + ff
            rec[b] = oa_counts_by_year(flt)
            time.sleep(0.15)
        for y in range(YEARS[0], YEARS[1] + 1):
            rows.append(dict(feld_id=fid, feld=name, jahr=y, **{b: rec[b].get(y, 0) for b in BINS}))
        print("   OpenAlex %2d/%d: %s" % (n, len(targets), name))
    df = pd.DataFrame(rows)
    df["a4_9"] = df.gesamt - df.a1 - df.a2 - df.a3 - df.a10plus
    for b in ["a1", "a2", "a3", "a4_9", "a10plus"]:
        df["anteil_" + b] = df[b] / df.gesamt.replace(0, np.nan)
    df["klasse"] = df.feld.map(FIELD_CLASS).fillna("gesamt")
    df.to_csv(os.path.join(out, "A_openalex_teamgroessen.csv"), index=False)
    return df


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def trend_deviation(years, values, pre, post, transform=logit):
    """Linearer Vor-Trend (auf transformierter Skala), Abweichung in der Nach-Periode."""
    y = np.asarray(years)
    v = transform(np.asarray(values, float))
    m = (y >= pre[0]) & (y <= pre[1]) & np.isfinite(v)
    if m.sum() < 4:
        return np.nan, np.nan, None
    b = np.polyfit(y[m], v[m], 1)
    resid_pre = v[m] - np.polyval(b, y[m])
    mp = (y >= post[0]) & (y <= post[1]) & np.isfinite(v)
    dev = float(np.mean(v[mp] - np.polyval(b, y[mp]))) if mp.any() else np.nan
    z = dev / max(resid_pre.std(ddof=2), 1e-9) if mp.any() else np.nan
    return dev, z, b


def boot_diff(a, b, rng):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def openalex_analyse(df, out, L):
    rng = np.random.default_rng(1)
    res = []
    for (fid, name), g in df[df.feld_id != "ALLE"].groupby(["feld_id", "feld"]):
        g = g.sort_values("jahr")
        if g.gesamt.sum() < 10000:
            continue
        r = dict(feld=name, klasse=FIELD_CLASS.get(name, "unklassifiziert"),
                 papers_2025=int(g[g.jahr == 2025].gesamt.sum()))
        for k in ["anteil_a1", "anteil_a2", "anteil_a4_9", "anteil_a10plus"]:
            dev, z, _ = trend_deviation(g.jahr, g[k], PRE, POST)
            r["abw_" + k] = dev
            r["z_" + k] = z
        pre_ = g[(g.jahr >= 2018) & (g.jahr <= 2022)].anteil_a1.mean()
        post_ = g[(g.jahr >= POST[0]) & (g.jahr <= POST[1])].anteil_a1.mean()
        r["einzelautor_2018_22"] = pre_
        r["einzelautor_2023_25"] = post_
        res.append(r)
    res = pd.DataFrame(res).sort_values("abw_anteil_a1", ascending=False)
    res.to_csv(os.path.join(out, "A_openalex_trendabweichung.csv"), index=False)

    L.append("=== A. WISSENSCHAFT (OpenAlex) ===")
    L.append("Abweichung vom Vor-KI-Trend %d-%d in %d-%d (logit-Skala; >0 = mehr als erwartet)."
             % (PRE[0], PRE[1], POST[0], POST[1]))
    L.append(res[["feld", "klasse", "einzelautor_2018_22", "einzelautor_2023_25", "abw_anteil_a1",
                  "z_anteil_a1", "abw_anteil_a10plus"]].round(4).to_string(index=False))
    L.append("\nGruppenmittel der Trendabweichung:")
    L.append(res.groupby("klasse")[["abw_anteil_a1", "abw_anteil_a2", "abw_anteil_a4_9",
                                    "abw_anteil_a10plus"]].mean().round(4).to_string())
    summ = {}
    for k, lab in [("abw_anteil_a1", "Einzelautoren"), ("abw_anteil_a10plus", "Grossteams >=10")]:
        d, lo, hi = boot_diff(res[res.klasse == "rechnerisch"][k], res[res.klasse == "physisch"][k], rng)
        summ[k] = (d, lo, hi)
        L.append("Differenz-in-Differenzen %s (rechnerisch - physisch): %.4f  [95%%-KI %.4f; %.4f]"
                 % (lab, d, lo, hi))
    L.append("")

    if HAS_MPL:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        ax = axs[0, 0]
        for cl, col in CLASS_COL.items():
            names = [n for n, c in FIELD_CLASS.items() if c == cl]
            g = df[df.feld.isin(names)].groupby("jahr")[["a1", "gesamt"]].sum()
            ax.plot(g.index, g.a1 / g.gesamt, "o-", ms=3, color=col, label=cl)
        g = df[df.feld_id == "ALLE"].set_index("jahr")
        ax.plot(g.index, g.anteil_a1, "k--", lw=1, label="alle Fachgebiete")
        ax.axvspan(2022.9, YEARS[1] + 0.5, color="#f4a259", alpha=0.15)
        ax.set_title("Anteil Einzelautor-Artikel", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax = axs[0, 1]
        for cl, col in CLASS_COL.items():
            names = [n for n, c in FIELD_CLASS.items() if c == cl]
            g = df[df.feld.isin(names)].groupby("jahr")[["a10plus", "gesamt"]].sum()
            ax.plot(g.index, g.a10plus / g.gesamt, "o-", ms=3, color=col, label=cl)
        ax.axvspan(2022.9, YEARS[1] + 0.5, color="#f4a259", alpha=0.15)
        ax.set_title("Anteil Grossteams (>=10 Autoren)", fontsize=10)
        ax = axs[0, 2]
        for cl, col in CLASS_COL.items():
            names = [n for n, c in FIELD_CLASS.items() if c == cl]
            g = df[df.feld.isin(names)].groupby("jahr")[["a1", "a2", "a3", "a4_9", "a10plus", "gesamt"]].sum()
            mid = (1 * g.a1 + 2 * g.a2 + 3 * g.a3 + 6 * g.a4_9 + 14 * g.a10plus) / g.gesamt
            ax.plot(g.index, mid, "o-", ms=3, color=col, label=cl)
        ax.axvspan(2022.9, YEARS[1] + 0.5, color="#f4a259", alpha=0.15)
        ax.set_title("Mittlere Teamgroesse (Naeherung aus Klassen)", fontsize=10)
        ax = axs[1, 0]
        r2 = res.sort_values("abw_anteil_a1")
        ax.barh(range(len(r2)), r2.abw_anteil_a1, color=[CLASS_COL.get(c, "#cccccc") for c in r2.klasse])
        ax.set_yticks(range(len(r2)))
        ax.set_yticklabels(r2.feld, fontsize=6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title("Einzelautoren: Abweichung vom Vor-KI-Trend (logit)", fontsize=10)
        ax = axs[1, 1]
        keys = ["abw_anteil_a1", "abw_anteil_a2", "abw_anteil_a4_9", "abw_anteil_a10plus"]
        labs = ["1 Autor", "2 Autoren", "4-9", ">=10"]
        x = np.arange(len(keys))
        for i, (cl, col) in enumerate(CLASS_COL.items()):
            g = res[res.klasse == cl][keys]
            ax.bar(x + (i - 1) * 0.27, g.mean().values, 0.27, yerr=g.std().values / np.sqrt(max(len(g), 1)),
                   color=col, label=cl, error_kw=dict(lw=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("Trendabweichung nach Teamgroesse und Fachtyp", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax = axs[1, 2]
        ax.scatter(res.einzelautor_2018_22, res.einzelautor_2023_25,
                   c=[CLASS_COL.get(c, "#cccccc") for c in res.klasse], s=25)
        lim = [0, max(res.einzelautor_2018_22.max(), res.einzelautor_2023_25.max()) * 1.05]
        ax.plot(lim, lim, "k:", lw=0.7)
        for _, r in res.iterrows():
            ax.annotate(r.feld[:18], (r.einzelautor_2018_22, r.einzelautor_2023_25), fontsize=5, alpha=0.7)
        ax.set_xlabel("Einzelautoren-Anteil 2018-22", fontsize=8)
        ax.set_ylabel("Einzelautoren-Anteil 2023-25", fontsize=8)
        ax.set_title("Vorher / nachher je Fachgebiet", fontsize=10)
        for ax in axs.flat:
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle("A: OpenAlex - Teamgroessen in der Wissenschaft (Artikel, 26 Fachgebiete)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "A_openalex.png"), dpi=150)
        plt.close(fig)
    return res


# =====================================================================
#  TEIL B: US-GRUENDUNGSSTATISTIK (CENSUS BUREAU)
# =====================================================================
CENSUS_BFS_URL = "https://www.census.gov/econ_getzippedfile/?programCode=BFS"
SECTIONS = ["CATEGORIES", "DATA TYPES", "ERROR TYPES", "GEO LEVELS", "TIME PERIODS", "NOTES", "DATA"]


def parse_census_mf(text):
    """Liest das Census-'mf'-Format: mehrere Abschnitte (CATEGORIES, DATA TYPES,
    GEO LEVELS, TIME PERIODS, DATA), jeweils mit eigener Kopfzeile."""
    import csv
    secs, cur, header = {}, None, None
    for row in csv.reader(io.StringIO(text)):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        first = cells[0].upper()
        if first in SECTIONS and all(not c for c in cells[1:]):
            cur, header = first, None
            secs[cur] = []
            continue
        if cur is None:
            continue
        if header is None:
            header = [c.lower() for c in cells]
            secs[cur + "_header"] = header
            continue
        secs[cur].append(dict(zip(header, cells)))
    return {k: pd.DataFrame(v) for k, v in secs.items() if not k.endswith("_header")}


def bfs_download(out):
    print("   lade Census-Gesamtdatei (ein Download, ca. 5-30 MB) ...")
    raw = http_bytes(CENSUS_BFS_URL, tries=3, timeout=180)
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    text = zf.read(name).decode("utf-8", errors="replace")
    with open(os.path.join(out, "B_census_rohformat_auszug.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text.splitlines()[:120]))          # fuer Fehlersuche
    secs = parse_census_mf(text)
    print("   Abschnitte gefunden: %s" % ", ".join("%s(%d)" % (k, len(v)) for k, v in secs.items()))
    D = secs["DATA"]
    cat = secs["CATEGORIES"].set_index("cat_idx")["cat_code"]
    dt = secs["DATA TYPES"].set_index("dt_idx")["dt_code"]
    geo = secs["GEO LEVELS"].set_index("geo_idx")["geo_code"]
    per = secs["TIME PERIODS"].set_index("per_idx")["per_name"]
    D = D.assign(sektor=D.cat_idx.map(cat), dtc=D.dt_idx.map(dt), geo=D.geo_idx.map(geo),
                 periode=D.per_idx.map(per))
    D = D[(D.geo == "US") & (D.is_adj.astype(str) == "1")]
    D = D[D.dtc.isin(["BA_BA", "BA_HBA", "BA_WBA"]) & D.sektor.isin(list(BFS_SECTORS))]
    datum = pd.to_datetime(D.periode, format="%b%Y", errors="coerce")
    for fmt in ["%b-%Y", "%Y-%m", "%m/%Y"]:
        datum = datum.fillna(pd.to_datetime(D.periode, format=fmt, errors="coerce"))
    df = pd.DataFrame(dict(datum=datum, wert=pd.to_numeric(D.val, errors="coerce"),
                           sektor=D.sektor, serie=D.dtc.str.replace("BA_", "", regex=False)))
    df["name"] = df.sektor.map(lambda c: BFS_SECTORS[c][0])
    df["klasse"] = df.sektor.map(lambda c: BFS_SECTORS[c][1])
    df = df.dropna(subset=["datum", "wert"])
    print("   %d Monatswerte fuer %d Branchen x %d Reihen" % (len(df), df.sektor.nunique(), df.serie.nunique()))
    if len(df) == 0:
        return None
    df.to_csv(os.path.join(out, "B_bfs_monatlich.csv"), index=False)
    return df


def bfs_analyse(df, out, L):
    df = df.copy()
    df["jahr"] = df.datum.dt.year
    ann = df.groupby(["sektor", "name", "klasse", "serie", "jahr"]).wert.mean().unstack("serie").reset_index()
    if "HBA" in ann:
        ann["anteil_HBA"] = ann.HBA / ann.BA
    if "WBA" in ann:
        ann["anteil_WBA"] = ann.WBA / ann.BA
    ann.to_csv(os.path.join(out, "B_bfs_jaehrlich.csv"), index=False)
    res = []
    for (code, name, cl), g in ann.groupby(["sektor", "name", "klasse"]):
        g = g.sort_values("jahr")
        r = dict(sektor=code, name=name, klasse=cl)
        for k in ["anteil_HBA", "anteil_WBA"]:
            if k in g and g[k].notna().sum() > 8:
                dev, z, _ = trend_deviation(g.jahr, g[k], PRE_BFS, POST_BFS)
                r["abw_" + k], r["z_" + k] = dev, z
                r[k + "_2019"] = float(g[g.jahr == 2019][k].mean())
                r[k + "_2025"] = float(g[g.jahr == 2025][k].mean())
        r["BA_wachstum_2019_2025"] = float(g[g.jahr == 2025].BA.mean() / g[g.jahr == 2019].BA.mean())
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(out, "B_bfs_trendabweichung.csv"), index=False)
    L.append("=== B. WIRTSCHAFT (US Business Formation Statistics) ===")
    L.append("Anteil Gruendungsantraege mit hoher Arbeitgeber-Wahrscheinlichkeit (HBA) bzw. mit")
    L.append("geplanten Loehnen (WBA) an allen Antraegen; Abweichung vom Vor-Trend %d-%d in %d-%d."
             % (PRE_BFS[0], PRE_BFS[1], POST_BFS[0], POST_BFS[1]))
    L.append(res.round(4).to_string(index=False))
    rng = np.random.default_rng(2)
    for k in ["abw_anteil_HBA", "abw_anteil_WBA"]:
        if k in res:
            d, lo, hi = boot_diff(res[res.klasse == "wissensintensiv"][k], res[res.klasse == "physisch"][k], rng)
            L.append("Differenz %s (wissensintensiv - physisch): %.4f [95%%-KI %.4f; %.4f]  (<0 = Modell bestaetigt)"
                     % (k, d, lo, hi))
    L.append("")
    if HAS_MPL:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, k, t in [(axs[0], "anteil_HBA", "Anteil Antraege mit hoher Arbeitgeber-Wahrscheinlichkeit"),
                         (axs[1], "anteil_WBA", "Anteil Antraege mit geplanten Loehnen")]:
            if k not in ann:
                continue
            for (code, name, cl), g in ann.groupby(["sektor", "name", "klasse"]):
                g = g.sort_values("jahr")
                ax.plot(g.jahr, g[k] / g[g.jahr == 2019][k].mean(), "-o", ms=2.5, color=BFS_COL[cl],
                        alpha=0.9 if cl != "gesamt" else 1, lw=1.0 if cl != "gesamt" else 2, label=name)
            ax.axvspan(2019.5, 2022.5, color="#dddddd", alpha=0.5)
            ax.axvspan(2022.9, 2026.5, color="#f4a259", alpha=0.15)
            ax.set_title(t + " (2019 = 1)", fontsize=10)
            ax.legend(fontsize=6, frameon=False)
        ax = axs[2]
        if "abw_anteil_WBA" in res:
            r2 = res.sort_values("abw_anteil_WBA")
            ax.barh(range(len(r2)), r2.abw_anteil_WBA, color=[BFS_COL[c] for c in r2.klasse])
            ax.set_yticks(range(len(r2)))
            ax.set_yticklabels(r2.name, fontsize=8)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title("Geplante Loehne: Abweichung vom Vor-Trend (logit)", fontsize=10)
        for ax in axs:
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle("B: US-Gruendungen - werden neue Firmen ohne Angestellte geplant? (grau = Corona)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "B_gruendungen.png"), dpi=150)
        plt.close(fig)
    return res


# =====================================================================
#  HAUPTPROGRAMM
# =====================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    out = os.path.join(here, "results_%s" % VERSION)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    L = ["Ergebnisse %s - empirische Pruefung | Abruf: %s" % (VERSION, time.strftime("%Y-%m-%d %H:%M")), ""]
    t0 = time.time()
    print("=== Empirie %s startet ===" % VERSION)

    print("[1/2] OpenAlex (Wissenschaft) ...")
    if not OPENALEX_API_KEY.strip():
        msg = "OpenAlex uebersprungen: kein API-Schluessel eingetragen (siehe Anleitung oben im Code)."
        print("   " + msg)
        L.append(msg + "\n")
    else:
        try:
            df = openalex_download(out)
            openalex_analyse(df, out, L)
        except urllib.error.HTTPError as e:
            msg = "OpenAlex-Fehler HTTP %s: %s" % (e.code, e.read()[:300])
            if e.code in (401, 403):
                msg += " -> Bitte API-Schluessel pruefen."
            print("   " + msg)
            L.append(msg + "\n")
        except Exception as e:
            msg = "OpenAlex-Fehler: %r" % (e,)
            print("   " + msg)
            L.append(msg + "\n")
    print("      (%.1f min)" % ((time.time() - t0) / 60))

    print("[2/2] US-Gruendungsstatistik (Census Bureau) ...")
    try:
        bfs = bfs_download(out)
        if bfs is not None:
            bfs_analyse(bfs, out, L)
        else:
            L.append("Census: keine passenden Daten gefunden (siehe B_census_rohformat_auszug.txt).\n")
    except Exception as e:
        msg = "Census-Fehler: %r" % (e,)
        print("   " + msg)
        L.append(msg + "\n")

    L.append("Laufzeit %.1f min | Python %s | numpy %s | pandas %s"
             % ((time.time() - t0) / 60, sys.version.split()[0], np.__version__, pd.__version__))
    txt = "\n".join(L)
    with open(os.path.join(out, "ZUSAMMENFASSUNG.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)
    zpath = os.path.join(here, "results_%s.zip" % VERSION)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), arcname="results_%s/%s" % (VERSION, fn))
    print("\nFERTIG. Bitte diese Datei in den Chat laden:\n  %s" % zpath)


if __name__ == "__main__":
    main()
