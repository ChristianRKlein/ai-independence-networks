# AI-driven independence in collaboration networks: code

Code for the paper *AI can substitute for collaborators and reshape networks in science and the economy* (C. R. Klein). One script reproduces every simulation result, the figures and the table of reported numbers.

## Requirements

Python ≥ 3.7 with NumPy, pandas and Matplotlib (`pip install -r requirements.txt`). No other dependencies. Tested with Python 3.7.9 / NumPy 1.19.3 / pandas 1.3.5 (Windows) and Python 3.13 / NumPy 2.5 / pandas 3.0 (Linux).

## Run

```
python run_all.py                    # all simulations -> results/   (about 15 min on one core)
python make_figures.py               # Figs 1-8          -> figures/
python make_supplementary_figures.py # Supplementary Figs 1-10 -> figures_supplementary/
python weighted_dependence.py       # output-weighted vs. mean individual dependence
```

`results/SCHLUESSELZAHLEN.csv` links every number reported in the paper to its source file and configuration (Supplementary Data 1). `results/parameters.json` records all parameters.

## Structure

| File | Content |
| --- | --- |
| `weighted_dependence.py` | check of mean individual vs. output-weighted dependence (Results) |
| `run_all.py` | model, network dynamics, mean-field theory, Myerson value and all experiments (P1–P8, Q1–Q3, S_A–S_C) |
| `make_figures.py`, `make_supplementary_figures.py` | figures from `results/` and `data/empirical/` |
| `empirics/openalex_fields_and_census.py` | field-level OpenAlex counts and US Census Business Formation Statistics |
| `empirics/openalex_subfields_preregistered.py` | preregistered subfield analysis (OSF: https://doi.org/10.17605/OSF.IO/WEFPT) |
| `data/empirical/` | outputs of the two empirical scripts as used in the paper |

The empirical scripts query the public OpenAlex API; an (optional, free) API key can be entered at the top of each script. No key is included.

## Experiments → figures

| Experiment | Output files | Figures |
| --- | --- | --- |
| P1 dissolution, independence, exit | `P1_*.csv` | Fig. 2 (incl. coordination grid), 3a–c; Supplementary Fig. 4 |
| P2 uptake atlas, η and ρ | `P2_*.csv` | Fig. 4; Supplementary Fig. 2 |
| P3 publications | `P3_*.csv` | Fig. 5; Supplementary Fig. 5 |
| P4 bottleneck and labour | `P4_*.csv` | Fig. 6a–c; Supplementary Fig. 6 |
| P5 Myerson value | `P5_*.csv` | Fig. 6d,e; Supplementary Fig. 7 |
| P6 theory and global sensitivity | `P6_*.csv` | Fig. 7a,c; Supplementary Figs 9, 10 |
| P8 robustness of selection | `P8_*.csv` | Supplementary Fig. 8 |
| Q1 selection under cost scaling | `Q1_selektion.csv` | Fig. 3d; Supplementary Fig. 8 |
| Q2 channel ablation | `Q2_kanaele.csv` | Fig. 7d |
| Q3 returns | `Q3_*.csv` | Fig. 7b |
| S_A–S_C time series, topologies, reciprocity | `SA_*`, `SB_*`, `SC_*` | Supplementary Figs 1, 3; Supplementary Note 4 |
| P7 reported numbers | `SCHLUESSELZAHLEN.csv` | Supplementary Data 1 |

## Glossary (column names are German)

Wissenschaft = science; Wirtschaft = economy; domaene = domain; mittlerer_grad = mean degree; anteil_isoliert = share isolated; abhaengigkeit = dependence (share of output owed to partners); einzelautor_anteil = share of single-authored papers; ideen / ideen_q = idea quality / quintile; nutzung / ki = AI uptake; Nadeloehr = bottleneck; Spezialisten = specialists; Transfers / Ausgleichszahlungen = transfers; gegenseitige Zustimmung = mutual consent; BASIS / WERT / NADELOEHR = Baseline / Value / Bottleneck configuration; unabhaengig = self-sufficient; abgehaengt = selected out; ausgeschlossen = excluded; suchend = searching; einkommen = income; kennzahl / wert / quelle = statistic / value / source; fertig = done.

## Reproducibility

All runs use fixed seeds. Runs with transfers are bit-identical across platforms; runs under mutual consent can differ slightly between platforms because floating-point rounding can flip near-zero link decisions (see Methods and Supplementary Note 5).

## Licence

MIT.
