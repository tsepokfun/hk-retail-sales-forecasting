# AI Sales Prediction for Hong Kong Retail

**A multi-horizon forecasting system for 23 Hong Kong retail outlet categories, comparing two deep-learning engines (xLSTM, iTransformer) against 17 statistical and machine-learning baselines — with a category-routed multi-dimensional text layer built from 38,298 government press headlines.**

<p align="center">
  <img src="figures/fig1_system_architecture.png" width="820" alt="End-to-end system architecture">
</p>

> **Research code and results only.** This repository contains the full modelling pipeline, evaluation harness, result tables and the written study. Coursework administration artifacts, the lecturer's assignment brief, and an unrelated individual-work dataset are deliberately **not** included — see [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).
>
> 中文版說明（原始工作筆記）→ [`README.zh.md`](README.zh.md)

---

## Abstract

Hong Kong's retail sector fell 7.33 % in 2024 while Lunar New Year moved 19 days later into February. A model that treats each category as a repetition of its own seasonal cycle therefore fails twice at once: it repeats a festival that arrived in the wrong month, and it applies one narrative to a jeweller and a supermarket.

This project builds an end-to-end system whose organising principle is that **the horizon and the category are inputs to the model, not dimensions of the report**:

1. **Multi-modal inputs** — 23 retail series, a macro/people/weather block, and a text stream drawn from 42,594 GovHK press headlines.
2. **A category-semantic router** that decomposes the text into four orthogonal streams (macro & policy, tourism & cross-border, durables/auto/property, staples & department retail) and routes it to each category by cosine attention against hand-written prototypes — every category gets its own 28-number text block per month instead of four shared numbers.
3. **Two AI engines** consuming it: an `xLSTM` state-space memory network with a learnable dual anchor and gated per-horizon text injection, and an inverted-attention `iTransformer` whose 23 sales tokens are joined by four text-modality tokens at zero extra trunk parameters.
4. **`MinT_wls` hierarchical reconciliation** across 8 parents and 15 children, so the published forecast satisfies the retail hierarchy's accounting identities exactly.

---

## Key results — 2024 out-of-sample, 1,104 scored points per model

| Model | MASE (ALL) | WAPE (ALL) | 2024 total forecast (HK$ m) | Year bias |
|---|---:|---:|---:|---:|
| Actual retail sales (reference) | — | — | 376,847 | — |
| Seasonal naive | 1.0303 | 10.44 % | 406,650 | +7.9 % |
| Ridge (residual target) | 1.1713 | 10.98 % | 382,394 | +1.5 % |
| LightGBM (tuned, residual) | 1.0411 | 10.41 % | 383,342 | +1.7 % |
| xLSTM + dual anchor | 0.9481 | 10.02 % | 390,854 | +3.7 % |
| xLSTM + M1/M3 pre-training | 0.9247 | 9.16 % | 387,351 | +2.8 % |
| **xLSTM + category-routed text** | **0.9055** | 9.20 % | **380,889** | **+1.07 %** |
| iTransformer + routed text tokens | 0.9223 | 9.03 % | 387,258 | +2.8 % |
| **iTransformer + dual anchor (champion)** | **0.9020** | **8.78 %** | 384,782 | +2.1 % |

**MASE < 1 means "better than repeating last year's same month".** The seasonal-naive rule would have carried HK$29,803 m of excess inventory in 2024; the champion removes HK$21,868 m of it, and the routed xLSTM — which posts the smallest dollar error in the study — removes HK$25,761 m.

**Lunar New Year window (2024 Jan–Feb, pooled):** seasonal naive 13.12 % WAPE vs xLSTM **8.81 %** (−32.8 %), entirely from the anchor's season term and the known target-month calendar.

<p align="center">
  <img src="figures/fig4_model_leaderboard.png" width="820" alt="Model leaderboard by MASE and WAPE">
</p>

Three findings worth the reader's time:

- **The anchor is worth more than the network.** Abandoning direct level prediction for a residual on a two-term anchor moves MASE from 2.096 to ≈ 1.0 — with **four parameters**.
- **Routing, not "more text", is the innovation.** Lumped text injected into the shared hidden state cost **9.7 %**; the same four columns injected per head cost 0.5 %; splitting the month into four dimensions without routing buys ~0.1 pp; routing those 28 features per category buys **3.7 pp**.
- **The text gain is horizon-structured, not global.** Against a matched no-text baseline the routed arm improves **10.0 % at h = 3** and **9.6 % at h = 6** (Diebold–Mariano p = 0.015 and p < 0.0001; better on 19 of 23 categories, Wilcoxon p = 0.0003 / 0.0031), while h = 1 and h = 12 are statistically indistinguishable. No harm is claimed there.

Negative results are kept in the repository on purpose: the 34-series panel expansion produced **no observable gain**, the M1/M3 pre-training gain (**−2.47 %**) is **smaller than one seed standard deviation**, and the pre-specified "close the h = 1 text gate" hypothesis was **refuted** by the data. See `docs/` for each.

---

## Repository layout

```
FINAL_REPORT.md              The study (6 chapters: problem, data, methodology, results, QA, deployment)
01_ARCHITECTURE.md           Architecture in full, including the 24 ablations
02_DATASET.md                Dataset card with the known-issues list
README.zh.md                 Original working notes in Chinese
figures/                     Six publication figures (dpi 300), regenerated by generate_report_figures.py

stage0/                      Core engine: xLSTM + dual anchor + residual formulation
  eval/harness.py            Locked evaluation harness (MASE / WAPE / three-level Diebold-Mariano)
  models/                    xLSTM, FiLM, neural-linear, iTransformer
  data/                      Panel builders, feature construction, exogenous fetchers, frozen CSVs
  results/                   Predictions, metrics, ablations, rolling-origin runs, audit reports
  RESULTS.md                 Stage 0 report (14 sections, including a correction notice)
  run_*.py                   One script per experiment family

stage1/RESULTS.md            Exogenous variables and FiLM gating
stage2_stage4/RESULTS.md     Two-stage residual correction and hierarchical reconciliation
stage3/                      Text: GovHK headline fetching, features, and the category-semantic router
  news_fetch.py              GovHK daily archive scraper
  news_features.py           text_scalar + MiniLM/PCA features
  multidim_text_model.py     Four-stream routing -> both engines -> the full ablation ladder
  RESULTS.md                 Stage 3 report

data_collectors/             TD vehicle first-registration collector (271 issues, 288 months, zero gaps)
docs/                        Literature reviews, data audits, reproduction notes, corrections log
```

---

## Quickstart

The pipeline reuses the system `torch` via `--system-site-packages` to avoid a 2.5 GB duplicate install.

```powershell
python -m venv .venv --system-site-packages
.venv\Scripts\pip install -r requirements.txt
```

**Rebuild data** (needs network; calls C&SD, HKO and GovHK):

```powershell
.venv\Scripts\python stage0\data\build_panel.py        # 23-series monthly panel
.venv\Scripts\python stage0\data\make_features.py      # 27 features per series-month
.venv\Scripts\python stage0\data\exog_fetch.py         # macro / people / weather raw snapshots
.venv\Scripts\python stage0\data\build_exog.py         # exogenous block
.venv\Scripts\python stage3\news_fetch.py              # GovHK daily archive
.venv\Scripts\python stage3\news_features.py           # text features
```

**Verify causality before trusting any number** — both must PASS:

```powershell
.venv\Scripts\python stage0\data\verify_exog_crosscheck.py
.venv\Scripts\python stage3\verify_text_causality.py
```

**Run the experiments:**

```powershell
.venv\Scripts\python stage0\run_stage0.py --models all --epochs 100
powershell -File stage0\run_pv_all.ps1        # leakage control: leaky vs purged split
powershell -File stage0\run_exog_all.ps1      # exogenous ablation + width-matched noise control
powershell -File stage3\run_stage3_all.ps1    # text fusion + LSTM backbone
.venv\Scripts\python stage0\run_stage4_reconcile.py
.venv\Scripts\python stage0\run_stage2_residual.py
.venv\Scripts\python stage0\make_business_tables.py
```

**Regenerate the figures:**

```powershell
.venv\Scripts\python generate_report_figures.py
```

---

## Evaluation protocol (fixed before results existed)

- **12 rolling origins**, 2023-12 → 2024-11; **4 direct horizons** h ∈ {1, 3, 6, 12}. Direct multi-horizon means four separate heads regress their target month straight from the origin, so errors never compound through the horizon.
- **1,104 scored points per model** (12 x 4 x 23), always on the original HK$ million scale after inverse transformation.
- **MASE** with the denominator computed as the in-sample seasonal-naive MAE at lag 12 **on the training period only** — scale-free, so a HK$15 m category weighs the same as a HK$50 m one.
- **WAPE** as the volume-weighted companion, because a retailer is paid in dollars, not ratios. A model can win on the median category and lose on the total, and only the pair reveals it.
- **Ensembles**: every deep arm runs seeds 42, 7 and 2024; **ENS3** averages the three prediction vectors and is scored as one model.
- **Significance**: Diebold–Mariano with Newey–West at lag h−1 on absolute and squared loss, plus a paired Wilcoxon collapsing the panel to one observation per category.

Everything at or before **2023-12** is training; **2024-01 … 2024-12** is the test year. Text features are prefix-masked so no feature at month *t* uses a headline published after *t*.

---

## Data sources

| Block | Source | Publisher |
|---|---|---|
| Retail sales (`VAL_RS`, 23 outlet-type series) | C&SD table `620-67002` | Census and Statistics Department, HKSAR |
| Composite CPI, policy rate, EERI | C&SD public tables | C&SD, HKMA |
| Inbound visitor arrivals | HKTB / C&SD public tables | HKTB, C&SD |
| MTR patronage | MTR Corporation monthly patronage releases | MTR |
| Rainfall, temperature | Hong Kong Observatory open data | HKO |
| Cargo throughput | Port Development / Marine Department statistics | HKSAR |
| Text corpus | GovHK daily press-release archive | HKSAR Government |
| Vehicle first registration | TD table 4.1(a)/(c)/(e), 271 issues | Transport Department |

All sources are publicly published government or public-transport data. **Nothing here redistributes licensed content.** Coverage, extraction quirks and known gaps are documented in [`docs/DATA_INVENTORY.md`](docs/DATA_INVENTORY.md) and [`02_DATASET.md`](02_DATASET.md).

---

## Five data traps this repository documents

These cost real debugging time and are recorded so the next person does not pay again:

1. **`sLSTMLayerConfig.backend` defaults to `'cuda'`** → on this machine it triggers a source compile and then `ImportError: DLL load failed`. Always pass `backend="vanilla"`.
2. **The `series` column is parsed as int64 by pandas** → `KeyError`. Always read with `dtype={"series": str}`.
3. **`web_table.html` is a JavaScript shell.** The actual data comes from `api/get.php?id=<table>&lang=en&full_series=1`.
4. **`api.hkma.gov.hk` was DNS-hijacked** at the time of collection → permanently excluded; HKMA figures come from published tables instead.
5. **Seed spread is 0.0387 MASE across 5 seeds.** Any difference smaller than **0.02 MASE is inside the noise** and is labelled as such everywhere in the report.

---

## Documentation index

| Document | What it answers |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | The complete study, including QA and a deployment guide |
| [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) | Every design decision, plus 24 ablations |
| [`02_DATASET.md`](02_DATASET.md) | Dataset card, hierarchy identities, known issues |
| [`docs/LITERATURE_2026_LANDSCAPE.md`](docs/LITERATURE_2026_LANDSCAPE.md) | Where this sits in the current literature |
| [`docs/LITERATURE_MODEL_CRITIQUES.md`](docs/LITERATURE_MODEL_CRITIQUES.md) | Honest critique of each baseline and engine |
| [`docs/LITERATURE_TEXT_AS_INPUT.md`](docs/LITERATURE_TEXT_AS_INPUT.md) | Prior work on text as a forecasting input |
| [`docs/DESIGN_FUSION_PROPOSAL.md`](docs/DESIGN_FUSION_PROPOSAL.md) | The design rationale for the routing layer |
| [`docs/UPGRADE_34SERIES_RESULTS.md`](docs/UPGRADE_34SERIES_RESULTS.md) | Negative result: 34-series expansion gave no gain |
| [`docs/PRETRAIN_M1M3_RESULTS.md`](docs/PRETRAIN_M1M3_RESULTS.md) | M1/M3 pre-training, and why the gain is not a headline |
| [`docs/REPRO_STATS_BASELINES.md`](docs/REPRO_STATS_BASELINES.md) | How each classical baseline was reproduced |
| [`docs/REPRO_EXOG_TOKEN.md`](docs/REPRO_EXOG_TOKEN.md) | Reproducing the exogenous and token ablations |
| [`docs/CORRECTIONS_LOG.md`](docs/CORRECTIONS_LOG.md) | Every error found and what it changed |
| [`docs/DATA_TABLE_AUDIT.md`](docs/DATA_TABLE_AUDIT.md) | Audit of every C&SD table used |
| [`docs/DATA_EXPANSION_ASSESSMENT.md`](docs/DATA_EXPANSION_ASSESSMENT.md) | Would more data help? (measured, not guessed) |
| [`docs/VERIFIED_ANCHORS.md`](docs/VERIFIED_ANCHORS.md) | Hand-checked anchor values |
| [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) | What is **not** in this repository and how to get it |

---

## Environment

Verified on Python 3.13.5, PyTorch 2.7.1+cu128, one RTX 4060 Ti (16 GB), Windows 11.
`requirements.txt` pins the Python packages; `xlstm==2.0.6` is a pure-Python wheel (no compilation).
Triton is **not** required — the pure-PyTorch path is used throughout.

---

## Status and scope

The modelling, evaluation and written study are complete. Known limitations are stated in the report rather than hidden, in particular:

- The text corpus is **official government communication** — arrivals, prices, property transactions, policy. It does not measure consumer mood, and the report says so.
- The MASE gap between the champion (0.9020) and the routed xLSTM (0.9055) is **not resolvable** on this dataset; they should be read as equivalent on the pooled statistic and different only in where they are strong. That is why the deployment guide is organised by horizon.
- Pre-training and panel expansion are reported as **weak or null** results.

## Acknowledgement

Data is published by the Census and Statistics Department, the Hong Kong Observatory, the Transport Department, the MTR Corporation and the GovHK press archive of the HKSAR Government; copyright remains with the originating bodies. The xLSTM architecture follows Beck et al. (2024); the iTransformer follows Liu et al. (2024).
