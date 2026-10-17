# FORBIOM-ON model note

Technical paper and reproducible code for **FORBIOM-ON**, an interregional optimization model of Ontario forest products, mass timber, and carbon pricing (2020–2030).

**Authors:** Rasoul Yousefpour and Yasser Elfahli

The LaTeX source is [`paper/forbiom.tex`](paper/forbiom.tex).

Related package: [forbiom-on](https://github.com/yassirelfahli/forbiom-on).

## What is in this repository

| Path | Contents |
|---|---|
| `paper/forbiom.tex` | Technical paper |
| `code/FORBIOM_ON.gms` | Reference GAMS model (solved as an NLP because revenue multiplies a fixed price variable by sales) |
| `code/sa13_gamspy.py` | Equivalent linear program in GAMSPy |
| `results/` | Summary tables from the GAMSPy solve, plus the GAMS summary used for the numerical check |

## Compile the paper

From `paper/`, with a TeX installation that provides `article`, `geometry`, `amsmath`, `booktabs`, and `hyperref`:

```bash
pdflatex forbiom.tex
```

## Run the model

Python 3.11 or newer, plus [GAMSPy](https://gamspy.readthedocs.io/):

```bash
pip install gamspy
python code/sa13_gamspy.py
```

The script solves the linear form with CPLEX. Random draws use the GAMS uniform generator at seed `3141`, in the same order as the GAMS loops. If you have a full GAMS license, point to it before running:

```bash
set GAMSPY_LICENSE=C:\path\to\gamslice.txt
```

The GAMS file needs GAMS and an NLP solver such as CONOPT:

```bash
gams code/FORBIOM_ON.gms
```

## Numerical agreement

On the published tables (objective, emissions, storage, net carbon, net-carbon change, and total harvest), the GAMSPy solution and the GAMS solution differ by at most `4.6e-5`.

## Scope of the data

Yield entries other than the coded white pine, jack pine, and tolerant-hardwood curves are proportional proxies. Arc and mill capacities are placeholders. They do not bind in the scenarios reported in the paper. Carbon accounts cover process emissions, freight, and carbon stored in products. They are not a full forest-carbon budget.
