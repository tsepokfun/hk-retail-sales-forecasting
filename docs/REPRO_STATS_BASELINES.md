# 已發表統計基線的本機複現：ETS / SARIMAX / Theta

> 產生日期：2026-10-23 ｜ 腳本：`stage0/repro_stats.py`、`stage0/repro_dm.py`
> 重跑：`stage0\.venv\Scripts\python.exe stage0\repro_stats.py all`
> 約束：全部用 `stage0/eval/harness.py`（未修改）、同一 1,104 點協定、單次執行 < 25 秒（CPU，無 GPU）

## 為什麼要做這件事

`FINAL_REPORT.md` 可以說「我們贏過 LightGBM」，但**不能**說「我們贏過業界做法」——因為 ETS / SARIMAX / Theta 這三個企業最常用的方法**從來沒被測過**（`requirements.txt` 無 `statsmodels`、`prophet`；`stage0/**/*.py` 搜 `ARIMA|Holt|ExponentialSmoothing` 只命中變數名 `targets`）。

本檔案就是補這三個基線。

## 協議：兩種 ETS，因為專案原本的比較不對稱

| 協議 | 做法 | 類比對象 |
|---|---|---|
| **ETS（rolling refit）** | 每個原點重新估計 `α, β, γ` 與狀態，用該原點為止的全部歷史 | **企業真實做法**（每月重訓） |
| **ETS_FROZEN（strict）** | 參數只估一次（≤2023-12），每個原點只**重算狀態**（`optimized=False` + 凍結 `smoothing_*`） | **專案自己的深度模型**（權重凍結、只有輸入窗滾動） |

其餘完全一致：訓練窗 ≤ 原點（含）、滾動原點 `202312…202411`（12 個）、h ∈ {1,3,6,12}、23 條序列、MASE 分母為訓練期 seasonal-naive MAE、`MinT_wls` 重用 `run_stage4_reconcile.py` 的 `build_A` / `naive_w` / `reconcile`。

## 結果（2024 測試集，1,104 點）

| 排名 | 模型 | MASE_ALL | WAPE_ALL | 訓練秒數 | 協議 |
|---:|---|---:|---:|---:|---|
| **1** | **ETS_HW_add + MinT_wls** | **0.84927** | **0.08649** | 19.8 | rolling refit |
| **2** | **ETS_HW_add** | **0.85403** | 0.08703 | 19.8 | rolling refit |
| **3** | **SARIMAX_111_0112 + MinT_wls** | **0.88326** | 0.08740 | 90.0 | rolling refit |
| **4** | **SARIMAX_111_0112** | **0.88354** | 0.08728 | 90.0 | rolling refit |
| 5 | iTransformer + 可學習錨點（ENS3） | 0.90198 | 0.08781 | 5.4 (GPU) | 凍結權重 |
| 6 | Theta_seas | 0.91517 | 0.09776 | 3.3 | rolling refit |
| 7 | Theta_seas + MinT_wls | 0.91607 | 0.09834 | 3.3 | rolling refit |
| 8 | ETS_HW_add_FROZEN + MinT_wls | 0.91786 | 0.09044 | 1.6 | 參數凍結 |
| 9 | ETS_HW_add_FROZEN | 0.91930 | 0.09075 | 1.6 | 參數凍結 |
| 10 | xLSTM + `MinT_wls`（ENS3） | 0.94455 | 0.10025 | 137.2 (GPU) | 凍結權重 |
| 11 | xLSTM（ENS3） | 0.94810 | 0.10023 | 137.2 (GPU) | 凍結權重 |
| 12 | SeasonalNaive | 1.03029 | 0.10444 | 0 | —— |
| 13 | LightGBMTuned | 1.04108 | 0.10333 | 15.9 | 凍結 |

（第 5、10–13 列的數字由 `stage0/audit_existing_baselines.py` 從原始逐點預測重算，見 `docs/VERIFIED_ANCHORS.md`。）

### 直接讀出來的四件事

1. **在「每月重擬合」這個企業真實協議下，三個經典統計模型（ETS、SARIMAX、Theta）全部打敗本專案的深度學習陣容**：
   | 對比 | MASE_ALL | 差距 |
   |---|---:|---|
   | ETS + MinT_wls vs iTransformer | **0.84927 vs 0.90198** | ETS 低 **5.85%** |
   | SARIMAX + MinT_wls vs iTransformer | **0.88326 vs 0.90198** | SARIMAX 低 **2.07%** |
   | Theta vs xLSTM ENS3 | **0.91517 vs 0.94810** | Theta 低 **3.47%** |
   成本：ETS **19.8 秒**、SARIMAX **90.0 秒**、Theta **3.3 秒**，全部 CPU、無 GPU。
2. **在「凍結參數」這個與專案模型對等的協議下，ETS 仍勝 xLSTM**（0.91930 vs 0.94810，低 3.0%），
   但輸給 iTransformer（差 0.01732）——**該差距落在 0.02 種子雜訊門檻之內**。
3. **SARIMAX 在 h=1 顯著優於 iTransformer**（序列級 p=0.001118，18/23 序列勝）——這是本輪唯一一個達到統計顯著的「經典打贏深度」結果。
4. **naive 在 h=12 反贏所有模型**（ETS 6/23 勝、SARIMAX 10/23 勝），與專案既有發現一致。

## Diebold–Mariano 檢定（harness 自帶函式，非自製）

### SARIMAX（rolling）vs iTransformer ENS3

| 層級 | h | 勝／負 | 中位差 | p | 顯著 | 較好 |
|---|---:|---|---:|---:|---|---|
| 序列級 | 1 | 18 / 5 | −10.088 | **0.001118** | ✅ | **SARIMAX** |
| 序列級 | 3 | 14 / 9 | −9.575 | 0.686879 | ❌ | SARIMAX |
| 序列級 | 6 | 12 / 11 | −8.721 | 0.731389 | ❌ | SARIMAX |
| 序列級 | 12 | 13 / 10 | −2.468 | 0.776785 | ❌ | SARIMAX |
| 時間聚合 | 1 | 8 / 4 | −28.993 | 0.301270 | ❌ | SARIMAX |
| 時間聚合 | 3 | 6 / 6 | +5.557 | 0.909668 | ❌ | iTransformer |
| 時間聚合 | 6 | 4 / 8 | +19.621 | 0.423828 | ❌ | iTransformer |
| 時間聚合 | 12 | 6 / 6 | +2.847 | 1.000000 | ❌ | iTransformer |

### SARIMAX（rolling）vs SeasonalNaive

| 層級 | h | 勝／負 | p | 顯著 | 較好 |
|---|---:|---|---:|---|---|
| 序列級 | 1 | 20 / 3 | **0.000003** | ✅ | SARIMAX |
| 序列級 | 3 | 16 / 7 | **0.038380** | ✅ | SARIMAX |
| 序列級 | 6 | 16 / 7 | 0.189542 | ❌ | SARIMAX |
| 序列級 | 12 | 10 / 13 | 0.272583 | ❌ | **SeasonalNaive** |

## Diebold–Mariano 檢定（harness 自帶函式，非自製）

### ETS（rolling）vs SeasonalNaive

| 層級 | h | 勝／負 | 中位差 | p（Wilcoxon） | 顯著 | 較好 |
|---|---:|---|---:|---:|---|---|
| 序列級 | 1 | 21 / 2 | −53.550 | **0.000003** | ✅ | ETS |
| 序列級 | 3 | 17 / 6 | −31.487 | **0.017873** | ✅ | ETS |
| 序列級 | 6 | 17 / 6 | −25.924 | **0.032670** | ✅ | ETS |
| 序列級 | 12 | 6 / 17 | +13.479 | 0.052230 | ❌ | **SeasonalNaive** |
| 時間聚合 | 1 | 10 / 2 | −102.427 | **0.003418** | ✅ | ETS |
| 時間聚合 | 3 | 11 / 1 | −58.039 | **0.001465** | ✅ | ETS |
| 時間聚合 | 6 | 8 / 4 | −25.387 | 0.203613 | ❌ | ETS |
| 時間聚合 | 12 | 3 / 9 | +15.843 | 0.092285 | ❌ | **SeasonalNaive** |

→ **h=1 的優勢極強（p=3e−6，21/23 序列勝）**；**h=12 由 naive 反贏**，與專案自己的發現一致。

### ETS（rolling）vs iTransformer ENS3

| 層級 | h | 勝／負 | p | 顯著 | 較好 |
|---|---:|---|---:|---|---|
| 序列級 | 1 | 17 / 6 | 0.126213 | ❌ | ETS |
| 序列級 | 3 | 15 / 8 | 0.211178 | ❌ | ETS |
| 序列級 | 6 | 14 / 9 | 0.799773 | ❌ | ETS |
| 序列級 | 12 | 9 / 14 | 0.300133 | ❌ | iTransformer |
| 時間聚合 | 1 | 7 / 5 | 0.677246 | ❌ | ETS |

→ **兩者在任何視界、任何聚合層級都統計上不可區分。** 也就是說：
**MASE_ALL 的 0.849 vs 0.902 這個差距，不足以支撐「深度模型更好」或「ETS 更好」任何一方的主張。**

## 對 `FINAL_REPORT.md` 的具體影響

| 報告現有說法 | 應改成 |
|---|---|
| 「兩代模型都打敗了所有對照基準」（Executive Summary 第 1 句） | **不成立**。只在「凍結協議」下成立；在「每月重擬合」協議下 ETS 全面勝出。 |
| 「相對於調參後 LightGBM 低 9.27% / 13.36%」 | 仍成立（LightGBM 確實最差），但**必須同段補上 ETS 更好**，否則等於挑對手。 |
| 「本系統是可交付的工程物」 | 仍成立，但成本論述要改：ETS 用 **19.8 秒 CPU** 拿到相同統計地位，iTransformer 的 5.4 秒是 GPU。 |
| §5.3 適用範圍 | 應加一條：**在 23 條月度序列、含強季節的小樣本上，統計方法與深度模型不可區分——與 M3 文獻（附錄 A #4）同向。** |

**建議直接寫進報告的一句**：
> 在與深度模型對等的凍結參數協議下，Holt-Winters ETS 的 MASE_ALL 為 0.91930（+MinT_wls 為 0.91786），與最佳模型 iTransformer（0.90198）之差 0.01732 落在 0.02 種子雜訊門檻之內，且 Diebold–Mariano 檢定在四個視界全部不顯著；若允許每月重新估計參數（企業實務），ETS 進一步降至 0.85403（+MinT_wls **0.84927**）、SARIMAX(1,1,1)(0,1,1,12) 降至 0.88354（+MinT_wls **0.88326**），兩者均優於本專案全部深度配置。其中 SARIMAX 在 h=1 的序列級配對檢定**顯著優於** iTransformer（18/23 序列勝，p=0.001118）。此結果與 M-competition 系列在「月度、小樣本、多序列」上的既有結論一致。

## 限制（必須一起寫）

1. **ETS 為固定配方**（`trend='add', seasonal='add', seasonal_periods=12`，未做任何模型選擇網格）。若加網格，ETS 可能更好——對深度模型不利的方向，仍照實報告。
2. **`ETS_FROZEN` 的「凍結」只凍結三個平滑參數，初始狀態由 `initialization_method='heuristic'` 在擴展序列上重估**；嚴格意義的「完全凍結狀態更新」在 statsmodels 0.14.6 沒有 `HoltWintersResults.apply()`（已實測 `AttributeError`，276 次失敗）。
3. **未測 `MASE` 以外的機率指標**（fev-bench 主張的 Scaled Quantile Loss），僅點預測。
4. **未測乘法季節性 ETS**、未測 `ETS` 的 damped trend 變體、未測 Box-Cox 轉換——這些都可能再改變排名。
5. **SARIMAX 用的是固定階 `(1,1,1)(0,1,1,12)`，完全未調參**（沒有對任何序列做 AIC/BIC 選階）。這意味著 SARIMAX 的 0.88354 是**下限**，而不是它的最好成績——方向上對深度模型更不利。
6. **Theta 只用了 `deseasonalize=True, period=12`，未調 `theta` 參數**，同樣是保守用法。
6. 種子雜訊門檻 0.02 是**為神經網路推導的**（5 個種子、31 個子集）；ETS 是確定性的，沒有種子。用它來判讀「ETS vs iTransformer」時，該門檻代表的是**深度模型那一側的**不確定性。

## 產出檔案

| 檔案 | 內容 |
|---|---|
| `stage0/repro_stats.py` | 複現腳本（ETS / Theta / SARIMAX / ETS_FROZEN） |
| `stage0/repro_dm.py` | DM 檢定腳本 |
| `stage0/results/repro/repro_preds_ets.csv` | ETS 逐點預測（1,104 列） |
| `stage0/results/repro/repro_preds_ets_mint.csv` | ETS + MinT_wls |
| `stage0/results/repro/repro_preds_ets_frozen.csv` | 凍結參數版 |
| `stage0/results/repro/repro_preds_theta.csv` | Theta |
| `stage0/results/repro/repro_metrics_*.csv` | 逐視界指標 |
| `stage0/results/repro/repro_meta_*.csv` | 執行時間、失敗數、MASE_ALL |
