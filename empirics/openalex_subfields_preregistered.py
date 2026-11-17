# -*- coding: utf-8 -*-
"""
=====================================================================
  KI-Unabhaengigkeit und Netzwerke  --  V7: VORREGISTRIERTER TEST
  der Nadeloehr-Hypothese auf Ebene der ~250 OpenAlex-Teilgebiete
=====================================================================

ABLAUF IN ZWEI STUFEN (wichtig fuer die Vorregistrierung!):

  STUFE = 1  (ca. 10 Sekunden)
      Laedt NUR die Liste der Teilgebiete (keine Publikationsdaten) und
      teilt sie nach den unten festgelegten Regeln ein. Erzeugt im Ordner
      'vorregistrierung_V7':
        - V7_Klassifikation.csv   (Einteilung aller Teilgebiete)
        - V7_Vorregistrierung.md  (fertiger Text fuer OSF)
      -> Beide Dateien auf https://osf.io hochladen und die Registrierung
         abschicken (Zeitstempel!). ERST DANACH Stufe 2 starten.

  STUFE = 2  (ca. 10-20 Minuten, Fortschritt in der Konsole)
      Laedt die Teamgroessen-Daten 2010-2026 fuer alle Teilgebiete und
      testet die vorregistrierten Hypothesen. Abgebrochene Laeufe setzen
      beim naechsten Start dort fort, wo sie aufgehoert haben.
      Ergebnis: results_V7.zip -> bitte in den Chat laden.

Der OpenAlex-Schluessel wird wie in V6 unten eingetragen.
"""

STUFE = 1                      # <-- erst 1, nach OSF-Registrierung auf 2 setzen
OPENALEX_API_KEY = ""          # optional: free OpenAlex API key

import os
import re
import sys
import json
import time
import shutil
import zipfile
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

VERSION = "V7"
YEARS = (2010, 2026)
PRE = (2012, 2022)
POST = (2023, 2025)
PLACEBO_PRE, PLACEBO_POST = (2010, 2018), (2019, 2021)
MIN_ARTICLES_PER_YEAR = 1000    # Teilgebiete mit weniger Artikeln/Jahr (Mittel 2012-2022) werden ausgeschlossen
N_BOOT = 5000
SEED = 20261117

# =====================================================================
#  VORAB FESTGELEGTE KLASSIFIKATIONSREGELN
#  Frage: Kann die Arbeit von Ko-Autoren in diesem Teilgebiet ueberwiegend
#  von KI uebernommen werden (Theorie, Rechnen, Text, Sekundaerdaten) -
#  oder ist sie an physische Nadeloehre gebunden (Labor, Klinik, Patienten,
#  Geraete, Feldarbeit, Tiere, Pflanzen)?
#  Reihenfolge: (1) exakte Namen, (2) Schluesselwoerter, (3) Fachgebiets-Standard.
# =====================================================================
EXACT_RECHNERISCH = {
    # Mathematik / Informatik / Statistik
    "Algebra and Number Theory", "Analysis", "Applied Mathematics", "Computational Mathematics",
    "Control and Optimization", "Discrete Mathematics and Combinatorics", "Geometry and Topology",
    "Logic", "Mathematical Physics", "Modeling and Simulation", "Numerical Analysis",
    "Statistics and Probability", "Theoretical Computer Science", "Artificial Intelligence",
    "Computational Theory and Mathematics", "Computer Graphics and Computer-Aided Design",
    "Computer Networks and Communications", "Computer Science Applications",
    "Computer Vision and Pattern Recognition", "Information Systems", "Signal Processing", "Software",
    "Statistics, Probability and Uncertainty", "Management Science and Operations Research",
    "Information Systems and Management", "Statistical and Nonlinear Physics",
    # Wirtschaft / Sozial- und Geisteswissenschaften
    "Economics and Econometrics", "Finance", "Accounting", "Strategy and Management", "Marketing",
    "Management Information Systems", "Business and International Management",
    "Law", "Political Science and International Relations", "Sociology and Political Science",
    "Library and Information Sciences", "Linguistics and Language", "Language and Linguistics",
    "Communication", "Public Administration", "Philosophy", "History", "Literature and Literary Theory",
    "Religious studies", "Classics", "Cultural Studies", "Gender Studies",
    "History and Philosophy of Science", "Visual Arts and Performing Arts", "Music",
    # rechnerische Teilgebiete in sonst physischen Fachgebieten (entscheidend fuer den Test innerhalb von Faechern)
    "Health Informatics", "Health Policy",
}
EXACT_GEMISCHT = {   # Teilgebiete mit erheblichem Anteil beider Arbeitsformen -> nicht getestet
    "Physical and Theoretical Chemistry", "Human-Computer Interaction", "Hardware and Architecture",
    "Epidemiology", "Public Health, Environmental and Occupational Health", "Astronomy and Astrophysics",
    "Cognitive Neuroscience", "Education", "Geography, Planning and Development", "Archeology",
    "Psychiatry and Mental health", "Clinical Psychology",
}
EXACT_PHYSISCH = {
    "Surgery", "Oncology", "Cardiology and Cardiovascular Medicine", "Radiology, Nuclear Medicine and Imaging",
    "Obstetrics and Gynecology", "Pediatrics, Perinatology and Child Health", "Orthopedics and Sports Medicine",
    "Neurology (clinical)", "Gastroenterology", "Nephrology", "Pulmonary and Respiratory Medicine",
    "Infectious Diseases", "Hematology", "Endocrinology, Diabetes and Metabolism", "Ophthalmology",
    "Otorhinolaryngology", "Urology", "Anesthesiology and Pain Medicine", "Emergency Medicine",
    "Critical Care and Intensive Care Medicine", "Dermatology", "Rheumatology", "Transplantation",
    "Hepatology", "Immunology and Allergy", "Reproductive Medicine", "Pathology and Forensic Medicine",
    "Physical Therapy, Sports Therapy and Rehabilitation",
    "Molecular Biology", "Cell Biology", "Genetics", "Biochemistry", "Biophysics", "Structural Biology",
    "Developmental Biology", "Cancer Research", "Physiology", "Clinical Biochemistry", "Molecular Medicine",
    "Organic Chemistry", "Inorganic Chemistry", "Analytical Chemistry", "Electrochemistry", "Spectroscopy",
    "Catalysis", "Biomaterials", "Ceramics and Composites", "Electronic, Optical and Magnetic Materials",
    "Metals and Alloys", "Polymers and Plastics", "Surfaces, Coatings and Films", "Materials Chemistry",
    "Immunology", "Microbiology", "Virology", "Parasitology", "Applied Microbiology and Biotechnology",
    "Cellular and Molecular Neuroscience", "Behavioral Neuroscience", "Sensory Systems",
    "Pharmacology", "Pharmaceutical Science", "Drug Discovery", "Toxicology",
    "Agronomy and Crop Science", "Animal Science and Zoology", "Plant Science", "Soil Science",
    "Horticulture", "Food Science", "Insect Science", "Aquatic Science", "Forestry",
    "Nuclear and High Energy Physics", "Condensed Matter Physics", "Atomic and Molecular Physics, and Optics",
    "Instrumentation", "Radiation", "Bioengineering", "Biotechnology",
}
KEYWORDS_PHYSISCH = [r"surg", r"clinic", r"oncolog", r"cardio", r"chemistry", r"molecular", r"cell",
                     r"microbio", r"immun", r"pharmac", r"veterin", r"dent", r"nursing", r"plant", r"animal",
                     r"materials?\b", r"experimental", r"laborator"]
KEYWORDS_RECHNERISCH = [r"theor", r"computa", r"mathemat", r"statistic", r"econom", r"informatics",
                        r"philosoph", r"histor", r"literat", r"linguist", r"\blaw\b", r"policy", r"software",
                        r"artificial intelligence", r"information"]
FIELD_DEFAULT = {   # identisch mit V6 (dort vorab festgelegt)
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


def classify(name, field):
    if name in EXACT_GEMISCHT:
        return "gemischt", "exakt"
    if name in EXACT_RECHNERISCH:
        return "rechnerisch", "exakt"
    if name in EXACT_PHYSISCH:
        return "physisch", "exakt"
    low = name.lower()
    r = any(re.search(k, low) for k in KEYWORDS_RECHNERISCH)
    p = any(re.search(k, low) for k in KEYWORDS_PHYSISCH)
    if r and not p:
        return "rechnerisch", "schluesselwort"
    if p and not r:
        return "physisch", "schluesselwort"
    return FIELD_DEFAULT.get(field, "gemischt"), "fachgebiet"


# =====================================================================
#  HTTP
# =====================================================================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KI-Netzwerk-Studie/1.0 (Wissenschaft; nicht-kommerziell)"
OA = "https://api.openalex.org"


def http_get(url, tries=4, timeout=60):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404):
                raise
            time.sleep(2 + 4 * k)
        except Exception as e:
            last = e
            time.sleep(2 + 4 * k)
    raise last


def oa(path, params):
    params = dict(params)
    params["api_key"] = OPENALEX_API_KEY
    return json.loads(http_get(OA + path + "?" + urllib.parse.urlencode(params, safe=":,|>/<-*")))


# =====================================================================
#  STUFE 1: KLASSIFIKATION + VORREGISTRIERUNG
# =====================================================================
def stufe1(here):
    out = os.path.join(here, "vorregistrierung_%s" % VERSION)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    rows, page = [], 1
    while True:
        js = oa("/subfields", {"per_page": 200, "page": page})
        for s in js["results"]:
            rows.append(dict(subfield_id=s["id"].replace("https://openalex.org/", ""),
                             teilgebiet=s["display_name"],
                             fachgebiet=(s.get("field") or {}).get("display_name", "")))
        if len(js["results"]) < 200:
            break
        page += 1
    df = pd.DataFrame(rows)
    cl = df.apply(lambda r: classify(r.teilgebiet, r.fachgebiet), axis=1)
    df["klasse"] = [c[0] for c in cl]
    df["regel"] = [c[1] for c in cl]
    df = df.sort_values(["fachgebiet", "teilgebiet"])
    df.to_csv(os.path.join(out, "V7_Klassifikation.csv"), index=False)
    counts = df.klasse.value_counts().to_dict()
    mixed_fields = df.groupby("fachgebiet").klasse.apply(lambda s: {"rechnerisch", "physisch"} <= set(s))
    mixed_fields = sorted(mixed_fields[mixed_fields].index)
    txt = VORREG_TEMPLATE.format(
        datum=time.strftime("%Y-%m-%d %H:%M"), n=len(df),
        n_r=counts.get("rechnerisch", 0), n_p=counts.get("physisch", 0), n_g=counts.get("gemischt", 0),
        felder=", ".join(mixed_fields) if mixed_fields else "(keine)",
        pre="%d-%d" % PRE, post="%d-%d" % POST, minart=MIN_ARTICLES_PER_YEAR, nboot=N_BOOT,
        ppre="%d-%d" % PLACEBO_PRE, ppost="%d-%d" % PLACEBO_POST)
    with open(os.path.join(out, "V7_Vorregistrierung.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(out, "V7_Analysecode_run_all.py"), "w", encoding="utf-8") as f:
        f.write(open(os.path.abspath(__file__), encoding="utf-8").read().replace(OPENALEX_API_KEY, "<SCHLUESSEL ENTFERNT>")
                if OPENALEX_API_KEY else open(os.path.abspath(__file__), encoding="utf-8").read())
    print(df.klasse.value_counts().to_string())
    print("\nFachgebiete mit beiden Klassen (fuer den Test innerhalb von Fachgebieten): %d" % len(mixed_fields))
    print("\nFERTIG (Stufe 1). Im Ordner\n  %s\nliegen drei Dateien:" % out)
    print("  V7_Vorregistrierung.md, V7_Klassifikation.csv, V7_Analysecode_run_all.py")
    print("-> Auf https://osf.io hochladen und registrieren (siehe ANLEITUNG_V7.md).")
    print("-> Danach oben STUFE = 2 setzen und erneut starten.")


VORREG_TEMPLATE = u"""# Vorregistrierung: KI und die Rueckkehr der Einzelautorschaft - Test der Nadeloehr-Hypothese

Erstellt: {datum} (automatisch durch Stufe 1 des Analysecodes; Publikationsdaten wurden zu diesem
Zeitpunkt fuer diese Analyse NICHT abgerufen)

## 1. Hintergrund und bereits bekannte Daten (Offenlegung)
Ein agentenbasiertes Netzwerkmodell sagt voraus, dass generative KI Kooperation dort ersetzt, wo die
Arbeit von Ko-Autoren KI-ersetzbar ist, und dort stabil laesst, wo physische Nadeloehre (Labor, Klinik,
Geraete, Feldarbeit) bestehen. Eine fruehere Analyse derselben Datenquelle auf Ebene von 26 Fachgebieten
(V6) ist bereits bekannt: Der Einzelautoren-Anteil liegt 2023-2025 in 24 von 26 Fachgebieten ueber dem
Vor-Trend; der Unterschied zwischen rechnerischen und physischen FACHGEBIETEN war nicht signifikant
(Punktschaetzer in Gegenrichtung). Diese Registrierung prueft die Hypothese auf der feineren Ebene der
TEILGEBIETE und insbesondere INNERHALB von Fachgebieten. Die Teilgebiets-Daten wurden vorher nicht
ausgewertet.

## 2. Hypothesen
H1 (primaer): Innerhalb von Fachgebieten (Fachgebiets-feste-Effekte) ist die Abweichung des
Einzelautoren-Anteils vom Vor-KI-Trend in rechnerischen Teilgebieten groesser als in physischen.
H2 (sekundaer): Ohne Fachgebiets-Effekte ist dieselbe Differenz positiv.
H3 (sekundaer): Die Abweichung des Anteils von Grossteams (>=10 Autoren) ist in rechnerischen
Teilgebieten negativer als in physischen (Grossteams bleiben um Nadeloehre erhalten).
Placebo (Validitaetspruefung): Bei einem fiktiven Bruch {ppost} (Vor-Trend {ppre}) ist die H1-Differenz
nicht von null verschieden.

## 3. Daten
OpenAlex-API, Werke vom Typ 'article', Jahre 2010-2026, Zuordnung ueber primary_topic.subfield.
Je Teilgebiet und Jahr: Zahl der Artikel mit >=1 Autor, mit genau 1 Autor, mit >=10 Autoren.
Ausschluss: Teilgebiete mit im Mittel weniger als {minart} Artikeln pro Jahr (2012-2022).
Das Jahr 2026 wird wegen unvollstaendiger Indexierung nicht ausgewertet.

## 4. Klassifikation (vor Sicht der Daten festgelegt, siehe V7_Klassifikation.csv)
Regeln in dieser Reihenfolge: (1) Liste exakter Teilgebietsnamen, (2) Schluesselwoerter,
(3) Standardklasse des Fachgebiets (identisch mit V6). Ergebnis bei {n} Teilgebieten:
rechnerisch = {n_r}, physisch = {n_p}, gemischt = {n_g} (gemischt wird nicht getestet).
Fachgebiete mit beiden Klassen (tragen zu H1 bei): {felder}.

## 5. Ergebnisvariable und Analyse
Fuer jedes Teilgebiet: linearer Trend des logit-Anteils {pre}; Ergebnis = mittlere Abweichung
(logit) in {post}. H1: OLS-Regression Abweichung ~ Klasse + Fachgebiets-Dummies (nur rechnerisch/
physisch); Koeffizient der Klasse. Konfidenzintervall per Cluster-Bootstrap ueber Fachgebiete
({nboot} Ziehungen), zusaetzlich Permutationstest (Klassen innerhalb von Fachgebieten permutiert).
H2: einfache Mittelwertdifferenz, Bootstrap ueber Teilgebiete. H3: wie H1 fuer den Grossteam-Anteil.
Signifikanzniveau 5 % (zweiseitig). Sensitivitaet: Gewichtung mit Wurzel der Artikelzahl; quadratischer
Vor-Trend; Ausschluss der Teilgebiete, die nur ueber den Fachgebiets-Standard klassifiziert wurden.

## 6. Interpretation (vorab festgelegt)
H1 bestaetigt, wenn der Koeffizient > 0 und das 95%-KI null ausschliesst. Wird H1 nicht bestaetigt,
berichten wir die Nadeloehr-Hypothese auf Teilgebietsebene als nicht gestuetzt.
"""


# =====================================================================
#  STUFE 2: DATEN + TEST
# =====================================================================
BINS = {"gesamt": "authors_count:>0", "a1": "authors_count:1", "a10plus": "authors_count:>9"}


def counts_by_year(extra):
    flt = "publication_year:%d-%d,type:article,%s" % (YEARS[0], YEARS[1], extra)
    js = oa("/works", {"filter": flt, "group_by": "publication_year", "per_page": 200})
    return {int(g["key"]): int(g["count"]) for g in js["group_by"]}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def deviation(years, share, pre, post, deg=1):
    y, v = np.asarray(years), logit(share)
    m = (y >= pre[0]) & (y <= pre[1]) & np.isfinite(v)
    mp = (y >= post[0]) & (y <= post[1]) & np.isfinite(v)
    if m.sum() < 4 or not mp.any():
        return np.nan
    b = np.polyfit(y[m], v[m], deg)
    return float(np.mean(v[mp] - np.polyval(b, y[mp])))


def _field_parts(x, y, w):
    """Beitrag eines Fachgebiets zum Within-Schaetzer (Zaehler, Nenner)."""
    xd = x - np.average(x, weights=w)
    yd = y - np.average(y, weights=w)
    return np.sum(w * xd * yd), np.sum(w * xd * xd)


def fe_test(d, ycol, rng, wcol=None):
    """OLS y ~ rechnerisch + Fachgebiets-Dummies (Within-Schaetzer).
    KI: Cluster-Bootstrap ueber Fachgebiete; p: Permutation der Klassen innerhalb der Fachgebiete."""
    d = d[d.klasse.isin(["rechnerisch", "physisch"])].dropna(subset=[ycol])
    both = d.groupby("fachgebiet").klasse.apply(lambda s: s.nunique() == 2)
    d = d[d.fachgebiet.isin(both[both].index)]
    groups = []
    for f, g in d.groupby("fachgebiet"):
        groups.append(((g.klasse == "rechnerisch").astype(float).values, g[ycol].values,
                       g[wcol].values if wcol else np.ones(len(g))))
    if len(groups) < 2:
        return dict(koef=np.nan, ki_lo=np.nan, ki_hi=np.nan, p_perm=np.nan, n_teil=len(d), n_fach=len(groups))
    parts = np.array([_field_parts(*g) for g in groups])
    est = parts[:, 0].sum() / parts[:, 1].sum()
    idx = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    num, den = parts[idx, 0].sum(1), parts[idx, 1].sum(1)
    boots = num[den > 0] / den[den > 0]
    perm = np.empty(N_BOOT)
    for k in range(N_BOOT):
        pp = np.array([_field_parts(rng.permutation(x), y, w) for x, y, w in groups])
        perm[k] = pp[:, 0].sum() / pp[:, 1].sum()
    return dict(koef=float(est), ki_lo=float(np.percentile(boots, 2.5)), ki_hi=float(np.percentile(boots, 97.5)),
                p_perm=float(np.mean(np.abs(perm) >= abs(est))), n_teil=len(d), n_fach=len(groups))


def diff_test(d, ycol, rng):
    a = d[d.klasse == "rechnerisch"][ycol].dropna().values
    b = d[d.klasse == "physisch"][ycol].dropna().values
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    return dict(diff=float(a.mean() - b.mean()), ki_lo=float(np.percentile(bs, 2.5)),
                ki_hi=float(np.percentile(bs, 97.5)), n_r=len(a), n_p=len(b))


def stufe2(here):
    if not OPENALEX_API_KEY.strip():
        print("Bitte zuerst den OpenAlex-Schluessel eintragen.")
        return
    kl_path = os.path.join(here, "vorregistrierung_%s" % VERSION, "V7_Klassifikation.csv")
    if not os.path.exists(kl_path):
        print("Klassifikation fehlt - bitte zuerst STUFE = 1 ausfuehren und auf OSF registrieren.")
        return
    kl = pd.read_csv(kl_path)
    cache = os.path.join(here, "cache_%s" % VERSION)
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(here, "results_%s" % VERSION)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    t0 = time.time()
    print("=== %s Stufe 2: %d Teilgebiete x %d Abfragen ===" % (VERSION, len(kl), len(BINS)))
    rows = []
    for n, r in enumerate(kl.itertuples(), 1):
        cf = os.path.join(cache, r.subfield_id.replace("/", "_") + ".json")
        if os.path.exists(cf):
            rec = json.load(open(cf))
        else:
            rec = {}
            for b, f in BINS.items():
                rec[b] = counts_by_year(f + ",primary_topic.subfield.id:" + r.subfield_id)
                time.sleep(0.1)
            json.dump(rec, open(cf, "w"))
        for y in range(YEARS[0], YEARS[1] + 1):
            rows.append(dict(subfield_id=r.subfield_id, teilgebiet=r.teilgebiet, fachgebiet=r.fachgebiet,
                             klasse=r.klasse, regel=r.regel, jahr=y,
                             **{b: rec[b].get(str(y), rec[b].get(y, 0)) for b in BINS}))
        if n % 10 == 0 or n == len(kl):
            el = (time.time() - t0) / 60
            print("   %3d/%d Teilgebiete  (%.1f min, noch ca. %.0f min)" % (n, len(kl), el, el / n * (len(kl) - n)))
    df = pd.DataFrame(rows)
    df["anteil_a1"] = df.a1 / df.gesamt.replace(0, np.nan)
    df["anteil_a10plus"] = df.a10plus / df.gesamt.replace(0, np.nan)
    df.to_csv(os.path.join(out, "V7_daten_teilgebiete.csv"), index=False)

    # Ergebnisvariablen je Teilgebiet
    res = []
    for sid, g in df.groupby("subfield_id"):
        g = g.sort_values("jahr")
        size = g[(g.jahr >= PRE[0]) & (g.jahr <= PRE[1])].gesamt.mean()
        res.append(dict(subfield_id=sid, teilgebiet=g.teilgebiet.iloc[0], fachgebiet=g.fachgebiet.iloc[0],
                        klasse=g.klasse.iloc[0], regel=g.regel.iloc[0], artikel_pro_jahr=size,
                        gewicht=np.sqrt(size),
                        abw_a1=deviation(g.jahr, g.anteil_a1, PRE, POST),
                        abw_a1_quadr=deviation(g.jahr, g.anteil_a1, PRE, POST, deg=2),
                        abw_a10=deviation(g.jahr, g.anteil_a10plus, PRE, POST),
                        placebo_a1=deviation(g.jahr, g.anteil_a1, PLACEBO_PRE, PLACEBO_POST),
                        einzel_2018_22=g[(g.jahr >= 2018) & (g.jahr <= 2022)].anteil_a1.mean(),
                        einzel_2023_25=g[(g.jahr >= POST[0]) & (g.jahr <= POST[1])].anteil_a1.mean()))
    res = pd.DataFrame(res)
    res["eingeschlossen"] = res.artikel_pro_jahr >= MIN_ARTICLES_PER_YEAR
    res.to_csv(os.path.join(out, "V7_abweichungen_teilgebiete.csv"), index=False)
    d = res[res.eingeschlossen]
    rng = np.random.default_rng(SEED)

    L = ["Ergebnisse %s - vorregistrierter Test | Abruf %s" % (VERSION, time.strftime("%Y-%m-%d %H:%M")),
         "Teilgebiete gesamt %d, eingeschlossen %d (>= %d Artikel/Jahr)" % (len(res), len(d), MIN_ARTICLES_PER_YEAR),
         "Klassen (eingeschlossen): %s" % d.klasse.value_counts().to_dict(), ""]
    tests = [("H1 (primaer) Einzelautoren, innerhalb Fachgebiete", "fe", "abw_a1", None, d),
             ("H2 Einzelautoren, ohne Fachgebiets-Effekte", "diff", "abw_a1", None, d),
             ("H3 Grossteams >=10, innerhalb Fachgebiete (erwartet < 0)", "fe", "abw_a10", None, d),
             ("Placebo 2019: Einzelautoren, innerhalb Fachgebiete (erwartet ~0)", "fe", "placebo_a1", None, d),
             ("Sens.: H1 gewichtet (Wurzel Artikelzahl)", "fe", "abw_a1", "gewicht", d),
             ("Sens.: H1 quadratischer Vor-Trend", "fe", "abw_a1_quadr", None, d),
             ("Sens.: H1 ohne Fachgebiets-Standard-Klassifikation", "fe", "abw_a1", None, d[d.regel != "fachgebiet"])]
    summ = []
    for lab, kind, col, w, dd in tests:
        if kind == "fe":
            t = fe_test(dd, col, rng, w)
            line = "%-66s Koef = %+.4f  [95%%-KI %+.4f; %+.4f]  p_perm = %.3f  (n = %d Teilgebiete in %d Fachgebieten)" % (
                lab, t["koef"], t["ki_lo"], t["ki_hi"], t["p_perm"], t["n_teil"], t["n_fach"])
        else:
            t = diff_test(dd, col, rng)
            line = "%-66s Diff = %+.4f  [95%%-KI %+.4f; %+.4f]  (n = %d vs %d)" % (
                lab, t["diff"], t["ki_lo"], t["ki_hi"], t["n_r"], t["n_p"])
        print(line)
        L.append(line)
        t["test"] = lab
        summ.append(t)
    pd.DataFrame(summ).to_csv(os.path.join(out, "V7_testergebnisse.csv"), index=False)
    L.append("")
    L.append("Allgemeiner Trendbruch (deskriptiv): Anteil Teilgebiete mit Abweichung > 0: %.2f (Placebo: %.2f)"
             % ((d.abw_a1 > 0).mean(), (d.placebo_a1 > 0).mean()))
    L.append("\nMittlere Abweichung je Klasse:")
    L.append(d.groupby("klasse")[["abw_a1", "abw_a10", "placebo_a1"]].agg(["mean", "count"]).round(4).to_string())
    L.append("\nJe Fachgebiet mit beiden Klassen (Mittel abw_a1):")
    both = d[d.klasse.isin(["rechnerisch", "physisch"])].groupby(["fachgebiet", "klasse"]).abw_a1.mean().unstack()
    L.append(both.dropna().round(4).to_string())

    if HAS_MPL:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        ax = axs[0]
        dd = d[d.klasse.isin(["rechnerisch", "physisch"])]
        both_f = dd.groupby("fachgebiet").klasse.nunique()
        fl = sorted(both_f[both_f == 2].index)
        for i, f in enumerate(fl):
            for cl, dx in [("rechnerisch", -0.15), ("physisch", 0.15)]:
                v = dd[(dd.fachgebiet == f) & (dd.klasse == cl)].abw_a1
                ax.scatter(np.full(len(v), i + dx), v, color=CLASS_COL[cl], s=14, alpha=0.8)
                if len(v):
                    ax.plot([i + dx - 0.1, i + dx + 0.1], [v.mean()] * 2, color=CLASS_COL[cl], lw=2)
        ax.set_xticks(range(len(fl)))
        ax.set_xticklabels([f[:22] for f in fl], rotation=40, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("H1: Abweichung Einzelautoren innerhalb von Fachgebieten\n(blau = rechnerisch, rot = physisch)",
                     fontsize=10)
        ax = axs[1]
        s = pd.DataFrame(summ)
        s["wert"] = s.koef.fillna(s.get("diff"))
        ax.errorbar(s.wert, range(len(s)), xerr=[s.wert - s.ki_lo, s.ki_hi - s.wert], fmt="o", color="k", capsize=3)
        ax.axvline(0, color="#b3001b", lw=0.8)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([t[:48] for t in s.test], fontsize=7)
        ax.invert_yaxis()
        ax.set_title("Vorregistrierte Tests (95%-KI)", fontsize=10)
        ax = axs[2]
        for cl, col in CLASS_COL.items():
            ids = d[d.klasse == cl].subfield_id
            g = df[df.subfield_id.isin(ids)].groupby("jahr")[["a1", "gesamt"]].sum()
            ax.plot(g.index, g.a1 / g.gesamt, "o-", ms=3, color=col, label=cl)
        ax.axvspan(2022.9, YEARS[1] + 0.5, color="#f4a259", alpha=0.15)
        ax.set_title("Einzelautoren-Anteil nach Klasse (Teilgebiete)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        for ax in axs:
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "V7_test.png"), dpi=150)
        plt.close(fig)

    L.append("\nLaufzeit %.1f min | Python %s | numpy %s | pandas %s" % (
        (time.time() - t0) / 60, sys.version.split()[0], np.__version__, pd.__version__))
    txt = "\n".join(L)
    with open(os.path.join(out, "ZUSAMMENFASSUNG.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)
    shutil.copy(kl_path, os.path.join(out, "V7_Klassifikation.csv"))
    zpath = os.path.join(here, "results_%s.zip" % VERSION)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), arcname="results_%s/%s" % (VERSION, fn))
    print("\nFERTIG. Bitte diese Datei in den Chat laden:\n  %s" % zpath)


def main():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if not OPENALEX_API_KEY.strip():
        print("Bitte oben den OpenAlex-Schluessel eintragen (derselbe wie in V6).")
        return
    try:
        if STUFE == 1:
            stufe1(here)
        else:
            stufe2(here)
    except urllib.error.HTTPError as e:
        print("OpenAlex-Fehler HTTP %s: %s" % (e.code, e.read()[:300]))
        if e.code in (401, 403):
            print("-> Bitte API-Schluessel pruefen.")


if __name__ == "__main__":
    main()
