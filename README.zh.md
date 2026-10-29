# ITP4524 — AI 香港零售銷售預測系統

香港 23 個零售行業類別的**多視界（1/3/6/12 個月）銷售額預測系統**。
以香港政府統計處（C&SD）零售業銷貨價值 `620-67002` 為目標，
比較兩代深度學習架構（xLSTM、iTransformer）與 17 個傳統統計／機器學習基線。

> **主報告 → [`FINAL_REPORT.md`](FINAL_REPORT.md)**（v5.0，含 Executive Summary、業務對照表、QA 章節）

---

## 核心結果（2024 年樣本外測試，1,104 點）

| 模型 | MASE_ALL | WAPE_ALL | 相對 Seasonal Naive |
|---|---:|---:|---:|
| **iTransformer + 可學習錨點** | **0.90198** | **8.78%** | **−12.45%** |
| xLSTM + `MinT_wls` 調和 | 0.94455 | 10.03% | −8.32% |
| xLSTM（集成） | 0.94810 | 10.02% | −8.32% |
| 調參後 LightGBM | 1.04108 | 10.33% | +1.05% |
| Seasonal Naive | 1.03029 | 10.44% | —— |

**總額層面**：naive 全年高估 HK\$29,803m；iTransformer 壓到 HK\$7,935m。

**農曆年窗口（2024 年 1–2 月 pooled WAPE）**：xLSTM **8.81%** vs naive 13.12%（**−32.84%**）。

---

## 目錄結構

```
FINAL_REPORT.md              主報告（v5.0）
01_ARCHITECTURE.md           架構設計全文，含 24 項消融
02_DATASET.md                資料集卡，含已知問題清單

stage0/                      核心引擎：xLSTM + 雙錨點 + 殘差化
  data/                      面板、特徵層、外生變數原始快照
  eval/harness.py            鎖死的評估框架（MASE / WAPE / 三層 DM）
  models/                    xLSTM、FiLM、神經線性、iTransformer
  results/                   預測、指標、消融、審計報告
  RESULTS.md                 階段 0 報告
stage1/  RESULTS.md          外生變數與 FiLM 門控
stage2_stage4/ RESULTS.md    兩階段殘差、層級調和
stage3/                      新聞文本融合
  news_fetch.py              GovHK 每日歸檔抓取
  news_features.py           text_scalar + MiniLM/PCA
  RESULTS.md

data_collectors/             TD 車輛首次登記收集器（271 期，288 月零缺口）
_research/                   資料源調查與快取
```

---

## 重現方式

```powershell
# 1) 環境（本專案用 --system-site-packages 重用系統 torch）
python -m venv stage0\.venv --system-site-packages
stage0\.venv\Scripts\pip install -r requirements.txt

# 2) 重建資料（會重新呼叫 C&SD / HKO / GovHK，需網路）
stage0\.venv\Scripts\python stage0\data\build_panel.py
stage0\.venv\Scripts\python stage0\data\make_features.py
stage0\.venv\Scripts\python stage0\data\exog_fetch.py
stage0\.venv\Scripts\python stage0\data\build_exog.py
stage0\.venv\Scripts\python stage3\news_fetch.py
stage0\.venv\Scripts\python stage3\news_features.py

# 3) 因果性驗證（必須全部 PASS 才可信）
stage0\.venv\Scripts\python stage0\data\verify_exog_crosscheck.py
stage0\.venv\Scripts\python stage3\verify_text_causality.py

# 4) 主實驗
stage0\.venv\Scripts\python stage0\run_stage0.py --models all --epochs 100
powershell -File stage0\run_pv_all.ps1          # 洩漏對照（leaky vs purged）
powershell -File stage0\run_exog_all.ps1        # 外生消融 + 同寬度噪音對照
powershell -File stage3\run_stage3_all.ps1      # 新聞文本 + LSTM 骨幹
stage0\.venv\Scripts\python stage0\run_stage4_reconcile.py
stage0\.venv\Scripts\python stage0\run_stage2_residual.py

# 5) 業務對照表
stage0\.venv\Scripts\python stage0\make_business_tables.py
```

---

## 五個必須知道的資料陷阱

1. **`sLSTMLayerConfig.backend` 預設 `'cuda'`** → 本機 `ImportError: DLL load failed`。一律傳 `backend="vanilla"`。
2. **`series` 欄會被 pandas 讀成 int64** → `KeyError`。一律 `dtype={"series": str}`。
3. **`web_table.html` 是 JS 外殼**，必須用 `api/get.php?id=<表號>&lang=en&full_series=1`。
4. **`api.hkma.gov.hk` 被 DNS 劫持**至阿里雲，永久排除。
5. **種子變異 0.0387**（5 個種子）→ **任何 < 0.02 MASE 的差異都在雜訊內**，報表必須標註。

完整清單見 `FINAL_REPORT.md` §4（品質保證）與 `02_DATASET.md` §7。

---

## 授權與用途

課程作業（ITP4524 Artificial Intelligence and Machine Learning, Section B）。
資料來源為香港政府公開統計（C&SD、HKO、TD、GovHK 新聞稿），版權屬原發布機構。
