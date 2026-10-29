# Data availability

This repository ships **code, frozen derived tables, result tables and the written study**. It does not ship every byte of raw material, for three reasons: size, licensing, and scope. This page states exactly what is missing and how to obtain or rebuild it, so the results remain reproducible.

---

## 1. Included — enough to re-verify every published number

| Item | Path | Size | Why it is committed |
|---|---|---|---|
| Monthly panel, 23 series x 263 months | `stage0/data/panel_wide.csv` | small | The target matrix; no gaps in any series |
| Feature table | `stage0/data/features_long.csv` | ~1.9 MB | 27 features per series-month, exactly as used in training |
| Scaling parameters | `stage0/data/scaling.csv` | small | Log-scale mean/sigma fitted on period <= 2023-12 only |
| 34-series panel + features | `stage0/data/panel_wide_34.csv`, `features_long_34.csv`, `scaling_34.csv` | ~2.8 MB | The negative-result expansion experiment |
| Exogenous block | `stage0/data/` exogenous CSV snapshots | small | The macro / people / weather / calendar inputs |
| Text headline corpus (extracted) | `stage3/data/news_headlines.csv` | ~3.4 MB | 42,594 headlines with dates, after extraction |
| Text wide features | `stage3/data/news_text_wide.csv` | small | Monthly text scalars |
| Multidim routing metadata | `stage3/data/multidim/routed_meta.json` | small | Stream definitions, prototype audit, admission counts |
| Every prediction and metric | `stage0/results/`, `stage3/results/` | ~46 MB | Point forecasts, metrics by horizon, ablations, DM tests |

Because the frozen panel, features and scaling files are committed, **`eval/harness.py` can recompute every MASE, WAPE and Diebold–Mariano statistic in the report without any network access.**

---

## 2. Not included, and how to rebuild it

### 2.1 GovHK daily archive cache

- **Missing:** `stage3/data/news_cache/` — about 200 monthly JSON files of raw scraped HTML.
- **Rebuild:** `.venv\Scripts\python stage3\news_fetch.py`
- **Source:** GovHK press-release archive, `https://www.info.gov.hk/gia/general/YYYYMMDD.htm` (public government pages, one page per day).
- **Why excluded:** it is a byte-level scrape cache; `news_headlines.csv` already contains the extracted, deduplicated result.

### 2.2 Sentence embeddings

- **Missing:** `stage3/data/multidim/headline_emb.npy` (56 MB).
- **Rebuild:** `.venv\Scripts\python stage3\news_features.py` then the embedding step of `stage3/multidim_text_model.py`.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), downloaded on first run.
- **Why excluded:** a 56 MB derived float matrix that is exactly reproducible from the committed headline list and a public model.

### 2.3 Raw exogenous snapshots

- **Missing:** `stage0/data/exog_raw/` (largest single file 20.9 MB of C&SD JSON).
- **Rebuild:** `.venv\Scripts\python stage0\data\exog_fetch.py` then `build_exog.py`.
- **Sources:** C&SD API (`https://www.censtatd.gov.hk/api/get.php?id=<table>&lang=en&full_series=1`), HKO open data, HKTB/C&SD, MTR patronage pages.
- **Why excluded:** re-fetchable in minutes; the API is authoritative and versioned by publication date.

### 2.4 TD vehicle first-registration data

- **Missing:** `data_collectors/out/td_first_registration_long.csv` (179 MB, 271 issues, 288 months).
- **Rebuild:** `python data_collectors\td_digest_collector.py` (all 271 table issues, then `make_verification.py`).
- **Source:** Transport Department monthly "Registration and Licensing of Vehicles by Class" tables 4.1(a)/(c)/(e), `https://www.td.gov.hk/`.
- **Why excluded:** 179 MB of raw rows for a feature that enters the model only as a monthly year-on-year aggregate. The collector is committed so the file can be regenerated exactly.
- **Note:** this was stored in the author's private archive repository via Git LFS; it is intentionally not carried into this public research repository.

### 2.5 M4 / M5 benchmark datasets

- **Missing:** `data_external/raw/benchmarks/` (about 600 MB of M4 `.tsf` and M5 CSVs).
- **Rebuild:** download from the original competition sources — M4 (`https://github.com/Mcompetitions/M4-methods`) and M5 (`https://www.kaggle.com/competitions/m5-forecasting-accuracy`).
- **Why excluded:** third-party competition data with their own terms; they were used only for domain-shift assessment (`docs/DATA_EXPANSION_ASSESSMENT.md`), never to produce a published metric.

### 2.6 Virtual environment and model checkpoints

- **Missing:** `.venv/`, `*.pt`, `*.pth`, `__pycache__/`.
- **Rebuild:** `python -m venv .venv --system-site-packages && .venv\Scripts\pip install -r requirements.txt`.
- **Why excluded:** platform-specific absolute paths and multi-GB binaries. Because the protocol fixes the seeds (42, 7, 2024) and the harness is deterministic given the same torch build, retraining reproduces the reported numbers; the committed point predictions let you verify without retraining.

---

## 3. Deliberately out of scope — not research material

The following exist in the author's private archive repository and are **not** published here, by design:

| Item | Reason |
|---|---|
| The course assignment brief (lecturer's document) | Third-party copyrighted teaching material |
| Course administration notes (module, class, submission logistics, student identifier) | Personal and administrative, not research |
| An unrelated individual-work dataset used for a different assignment section | Out of scope for this project and not redistributed |
| Draft and backup report versions | Superseded by `FINAL_REPORT.md` |
| One-off diagnostic scratch scripts (`_*.py`) | Not reproducible inputs; their findings are folded into `docs/CORRECTIONS_LOG.md` |

---

## 4. Licensing of the data

All series originate from public HKSAR Government publications and from public transport-operator releases. **Copyright remains with the originating bodies**; this repository redistributes only derived aggregates and short factual series that the publishers make publicly available, and cites every source. If you intend to redistribute the raw data further, consult the terms of the originating body (C&SD, HKO, TD, GovHK) first.
