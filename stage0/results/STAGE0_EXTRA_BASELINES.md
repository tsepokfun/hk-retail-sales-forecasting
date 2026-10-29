# Stage 0 — 補充基線：五個過去未實測的模型

> 全部數字由 `stage0\run_baselines_extra.py` 實際執行產生，無任何估計值。
> 評估框架 = `stage0\eval\harness.py`（未修改）。

## 0. 協定自檢（先確認沒有搞錯協定）

用同一支 harness 重跑 Seasonal Naive，得到 h=6 MASE = 0.97471（驗收值 0.97471）、h=12 MASE = 0.80900（驗收值 0.80900）—— **完全一致，協定確認**。

| 項目 | 值 |
|---|---|
| 原點 | 12 個（202312 .. 202411） |
| 視界 | h = 1, 3, 6, 12（Direct，逐 h 一個模型） |
| 訓練切片 | `target_period <= origin`，每個原點重新 fit |
| 目標形式 | 每個學習器都跑 `level` 與 `residual` |
| 種子 | [42, 7, 2024]（LightGBM）；Ridge/kNN/NLinear/DLinear/KMeans 為確定性，跑一次 |
| xLSTM 參考 | MASE_ALL = 0.98355（purged 協定，三種子均值，來自 `results/stage0_purged_val_agg.csv`） |

## 1. 統一對照表（MASE，越低越好；< 1 = 打敗 seasonal naive）

| 模型 | 目標形式 | h=1 | h=3 | h=6 | h=12 | MASE_ALL | 相對 xLSTM 0.98355 |
|---|---|---:|---:|---:|---:|---:|---:|
| SeasonalNaive | naive | 1.2123 | 1.1251 | 0.9747 | 0.8090 | 1.0303 | +4.75% |
| Drift | n/a | 1.2053 | 1.7904 | 2.0066 | 0.8620 | 1.4661 | +49.06% |
| Ridge_level | n/a | 1.6942 | 1.7639 | 1.8369 | 1.8053 | 1.7751 | +80.48% |
| Ridge_residual | n/a | 1.0750 | 1.2991 | 1.4456 | 0.8653 | 1.1713 | +19.09% |
| LightGBM_level | n/a | 1.0745 | 1.2353 | 1.3405 | 1.4631 | 1.2784 | +29.97% |
| LightGBM_residual | n/a | 0.9710 | 1.1330 | 1.2753 | 0.9777 | 1.0893 | +10.75% |
| xLSTM (purged, 3-seed mean) | residual | 0.8916 | 1.0700 | 1.1096 | 0.8629 | 0.9836 | +0.00% |
| KMeansRidge_level | level | 1.7050 | 1.7339 | 1.7981 | 1.7852 | 1.7556 | +78.49% |
| KMeansRidge_residual | residual | 1.0295 | 1.1874 | 1.2352 | 0.8754 | 1.0819 | +10.00% |
| kNN_level | level | 1.5709 | 1.6587 | 1.7501 | 1.9573 | 1.7342 | +76.32% |
| kNN_residual | residual | 0.9468 | 1.1258 | 1.1836 | 0.8390 | 1.0238 | +4.09% |
| NLinear_level | level | 1.0697 | 1.2212 | 1.3375 | 1.0148 | 1.1608 | +18.02% |
| NLinear_residual | residual | 1.0636 | 1.2049 | 1.3234 | 1.0132 | 1.1513 | +17.05% |
| DLinear_level | level | 1.1489 | 1.5501 | 1.6270 | 1.0440 | 1.3425 | +36.49% |
| DLinear_residual | residual | 1.1489 | 1.5501 | 1.6270 | 1.0440 | 1.3425 | +36.49% |
| LightGBMTuned_level | level | 0.9813 | 1.1469 | 1.2662 | 1.6137 | 1.2520 | +27.29% |
| LightGBMTuned_residual | residual | 0.8957 | 1.0640 | 1.2561 | 0.9486 | 1.0411 | +5.85% |

說明：
- 前 6 列（SeasonalNaive / Drift / Ridge_* / LightGBM_*）是已鎖死的既有結果，直接取自 `results/stage0_results.csv`（同一支 harness、同一協定），不是重算值。
- `xLSTM (purged, 3-seed mean)` 的逐 h 數字是用 `results/purged_val_predictions.csv` 的三個種子預測經 `H.metrics_from_frame` 逐種子算 MASE 後取均值；其 MASE_ALL 均值即 0.98355。
- 最後 10 列是本輪新實測；LightGBMTuned 為三種子均值，其餘為單次確定性執行。

### 1.1 一個必須指出的數學事實：DLinear 的 level 與 residual 完全等價

實測 `max|pred_level − pred_residual|` = **1.36e-11**（DLinear）、**220.4**（NLinear，逐點差異最大值，單位 HK$m）。

DLinear 的設計矩陣是 `[trend | seasonal]`，而 `trend + seasonal = 原始 6 欄`，所以 `y_z(t)` 本身就在設計矩陣的行空間內。殘差目標 `y_z(t+h) − y_z(t)` 與水平目標 `y_z(t+h)` 只差一個落在行空間內的向量，OLS 給出的預測因此**逐點相同**。
這不是 bug，是線性模型在『殘差化』上的天花板：**對 DLinear 而言，殘差化目標沒有免費的錨點可拿**，與 Ridge（設計矩陣不含 `y_z(t)`，殘差化值 −34%）形成直接對照。

## 2. 對 xLSTM 的三層 DM 檢定（只做 MASE_ALL 最好的 3 個）

入選（依 MASE_ALL）：**kNN_residual, LightGBMTuned_residual, KMeansRidge_residual**。
對照組 = `xLSTM_purged_s42`（purged 協定的主種子 42，MASE_ALL = 1.01743）。

### 2.1 Pooled DM（N=276，Newey–West 修正；已知偏樂觀）

| h | model | n | DM_sq | p_sq | better_sq | sig | DM_abs | p_abs | better_abs | sig |
|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| 1 | kNN_residual | 276 | -1.2290 | 0.2191 | model |  | -1.0518 | 0.2929 | model |  |
| 1 | LightGBMTuned_residual | 276 | -2.3309 | 0.0198 | model | ★ | -2.3169 | 0.0205 | model | ★ |
| 1 | KMeansRidge_residual | 276 | -1.5659 | 0.1174 | model |  | -1.2971 | 0.1946 | model |  |
| 3 | kNN_residual | 276 | +1.1899 | 0.2341 | baseline |  | -0.0823 | 0.9344 | model |  |
| 3 | LightGBMTuned_residual | 276 | -1.6648 | 0.0960 | model |  | -1.9807 | 0.0476 | model | ★ |
| 3 | KMeansRidge_residual | 276 | -2.0734 | 0.0381 | model | ★ | -1.6329 | 0.1025 | model |  |
| 6 | kNN_residual | 276 | -0.3943 | 0.6933 | model |  | -1.5706 | 0.1163 | model |  |
| 6 | LightGBMTuned_residual | 276 | -0.7792 | 0.4359 | model |  | -2.0226 | 0.0431 | model | ★ |
| 6 | KMeansRidge_residual | 276 | -1.3172 | 0.1878 | model |  | -1.5884 | 0.1122 | model |  |
| 12 | kNN_residual | 276 | +2.5059 | 0.0122 | baseline | ★ | +0.8677 | 0.3856 | baseline |  |
| 12 | LightGBMTuned_residual | 276 | +3.1331 | 0.0017 | baseline | ★ | +2.8588 | 0.0043 | baseline | ★ |
| 12 | KMeansRidge_residual | 276 | +1.2083 | 0.2269 | baseline |  | +0.5199 | 0.6031 | baseline |  |

`DM > 0` ⇒ 該模型損失較高（xLSTM 較好）。★ = p < 0.05。

### 2.2 Series-level（N=23，配對 Wilcoxon）

| h | model | n_series | won | lost | median_diff(HK$m) | p_wilcoxon | better | sig |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | kNN_residual | 23 | 11 | 12 | 0.86 | 0.6434 | xLSTM_purged_s42 |  |
| 1 | LightGBMTuned_residual | 23 | 17 | 6 | -4.35 | 0.0254 | LightGBMTuned_residual | ★ |
| 1 | KMeansRidge_residual | 23 | 11 | 12 | 2.35 | 0.6010 | xLSTM_purged_s42 |  |
| 3 | kNN_residual | 23 | 8 | 15 | 9.04 | 0.4820 | xLSTM_purged_s42 |  |
| 3 | LightGBMTuned_residual | 23 | 12 | 11 | -2.01 | 0.5399 | LightGBMTuned_residual |  |
| 3 | KMeansRidge_residual | 23 | 9 | 14 | 5.14 | 0.7768 | xLSTM_purged_s42 |  |
| 6 | kNN_residual | 23 | 10 | 13 | 2.91 | 0.7314 | xLSTM_purged_s42 |  |
| 6 | LightGBMTuned_residual | 23 | 13 | 10 | -7.54 | 0.1695 | LightGBMTuned_residual |  |
| 6 | KMeansRidge_residual | 23 | 11 | 12 | 1.84 | 0.4452 | xLSTM_purged_s42 |  |
| 12 | kNN_residual | 23 | 14 | 9 | -6.68 | 0.5009 | kNN_residual |  |
| 12 | LightGBMTuned_residual | 23 | 10 | 13 | 2.48 | 0.5803 | xLSTM_purged_s42 |  |
| 12 | KMeansRidge_residual | 23 | 13 | 10 | -2.15 | 0.7314 | KMeansRidge_residual |  |

`better` 依 median(該模型 AE − xLSTM AE) 的符號決定；負值 ⇒ 該模型較好。

### 2.3 Time-aggregated（T=12，配對 Wilcoxon）

| h | model | n_origins | won | lost | median_diff(HK$m) | p_wilcoxon | better | sig |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | kNN_residual | 12 | 7 | 5 | -14.51 | 0.5693 | kNN_residual |  |
| 1 | LightGBMTuned_residual | 12 | 6 | 6 | -0.96 | 0.4238 | LightGBMTuned_residual |  |
| 1 | KMeansRidge_residual | 12 | 6 | 6 | 12.69 | 0.6772 | xLSTM_purged_s42 |  |
| 3 | kNN_residual | 12 | 6 | 6 | 6.15 | 0.9097 | xLSTM_purged_s42 |  |
| 3 | LightGBMTuned_residual | 12 | 8 | 4 | -25.70 | 0.2661 | LightGBMTuned_residual |  |
| 3 | KMeansRidge_residual | 12 | 7 | 5 | -26.21 | 0.3394 | KMeansRidge_residual |  |
| 6 | kNN_residual | 12 | 9 | 3 | -21.06 | 0.1514 | kNN_residual |  |
| 6 | LightGBMTuned_residual | 12 | 7 | 5 | -38.73 | 0.1294 | LightGBMTuned_residual |  |
| 6 | KMeansRidge_residual | 12 | 7 | 5 | -32.80 | 0.3394 | KMeansRidge_residual |  |
| 12 | kNN_residual | 12 | 4 | 8 | 20.69 | 0.2036 | xLSTM_purged_s42 |  |
| 12 | LightGBMTuned_residual | 12 | 2 | 10 | 48.33 | 0.0210 | xLSTM_purged_s42 | ★ |
| 12 | KMeansRidge_residual | 12 | 5 | 7 | 2.78 | 0.6772 | xLSTM_purged_s42 |  |

## 3. 誠實結論：哪個基線在哪些視界贏過 xLSTM？

**沒有任何一個新基線在 MASE_ALL 上贏過 xLSTM。** 最好的新基線是 `kNN_residual`，MASE_ALL = 1.02377，比 xLSTM 的 0.98355 高 4.09%。

逐視界（每個 h 上最好的新基線 vs xLSTM 同視界的三種子均值）：

| 視界 | 最佳新基線 | 其 MASE | xLSTM MASE | 勝負 | 差距 |
|---|---|---:|---:|---:|---:|
| h=1 | LightGBMTuned_residual | 0.89568 | 0.89161 | 輸 | +0.46% |
| h=3 | LightGBMTuned_residual | 1.06398 | 1.07002 | 贏 | -0.56% |
| h=6 | kNN_residual | 1.18355 | 1.10963 | 輸 | +6.66% |
| h=12 | kNN_residual | 0.83898 | 0.86294 | 贏 | -2.78% |

→ **在 h=3, h=12 上，新基線的 MASE 低於 xLSTM。** 注意這是逐格比較、未做多重比較校正；是否顯著看上節 DM（三層）。

統計面：pooled(abs) 顯著贏過 xLSTM 的格數 = 3；series-level 顯著 = 1；time-aggregated 顯著 = 0。

### 3.1 ⚠️ MASE 與 DM 在 h=12 上互相矛盾——必須誠實寫出來

h=12 時 `kNN_residual` 的 MASE = 0.83898 低於 xLSTM 的 0.86294，但 DM（pooled，絕對與平方損失）一致說 xLSTM 較好，WAPE 也是 xLSTM 較好（9.16% vs 8.72%）。

原因是兩個指標的加權方式不同：**MASE 先在每條序列內部正規化再等權平均**（23 條序列等權，小序列與大序列份量相同），**WAPE 與 pooled DM 則是金額加權**（被 `18`、`35`、`40` 等大序列主導）。kNN 在小序列上較準、在大序列上較差，於是同一組預測同時「MASE 贏、WAPE 輸」。
用 §2.2 的 series-level DM 交叉檢查：h=12 時 kNN 在 23 條裡的 14 條贏（median_diff -6.68 HK$m），但 Wilcoxon p=0.5009 → **不顯著**。

**所以「kNN 在 h=12 贏過 xLSTM」這句話只能寫成：在等權 MASE 上贏 2.78%，但在金額加權誤差與三層 DM 上都不成立。**

### 3.2 三個附帶發現

1. **分群有效但不足以翻盤**：`KMeansRidge_residual` MASE_ALL = 1.08189，比 `Ridge_residual`（1.17127）改善 -7.63%，但距離 xLSTM 仍差 10.00%。分群把「一條 Ridge 服務 23 個類別」的問題縮小，但不是主要瓶頸。
2. **殘差化依然是最大的單一因素**：本輪 5 個模型幾乎全部重演 RESULTS.md §5 的結論 — `kNN` 1.7342 → 1.0238、`KMeansRidge` 1.7556 → 1.0819、`LightGBMTuned` 1.2520 → 1.0411、`NLinear` 1.1608 → 1.1513；只有 `DLinear` 因 §1.1 的代數原因完全沒變。**沒有任何一個架構改動，效果比『把目標換成殘差』更大。**
3. **NLinear / DLinear 這兩個 2023 年的線性 SOTA，在本資料上不如 seasonal naive**（NLinear_residual 1.1513、DLinear_residual 1.3425，naive 1.0303）。原因是它們的輸入只有 6 個純量欄位（本專案沒有真正的多變量序列，`H.FEATURE_COLS` 是工程好的滯後 + 日曆 + 制度 dummy），**這兩個架構的長處（長 context 的通道獨立線性外推）在這裡沒有施展空間。**報告引用它們時必須連這句一起寫，否則會被誤讀成「線性模型很弱」。


## 4. LightGBM 網格搜索（訓練期內部時間序列驗證，未碰 2024）

驗證設計：對每個 h，取「目標月仍 <= 202312」的最後 24 個原點，逐原點 `train{target_period<=o} → predict{period==o}`，以 y_z 空間 pooled MAE 選參。16 組候選全表在 `results/extra_lgb_tuning_audit.csv`。

實際 fit 次數 = 4 個視界 × 16 組候選 × 24 個驗證原點 = **1,536**（可由 audit 表的 `n_val_origins` 欄加總核對）。選參用 `residual` 目標形式，選中的超參**原封不動套用到 level 與 residual 兩種模式**，避免「重新調參」混淆 level/residual 的對照。

| 視界 | num_leaves | learning_rate | min_child_samples | n_estimators | val_MAE_yz |
|---|---:|---:|---:|---:|---:|
| h=1 | 31 | 0.1 | 5 | 600 | 0.38404 |
| h=3 | 31 | 0.1 | 5 | 600 | 0.51843 |
| h=6 | 31 | 0.1 | 20 | 600 | 0.62856 |
| h=12 | 31 | 0.03 | 5 | 200 | 0.53741 |

## 5. K-Means 分群（僅用訓練期畫像）

| cluster | n_series | mean(y_log 均值) | mean(y_log 標準差) | mean(lag12 自相關) | series |
|---:|---:|---:|---:|---:|---|
| 0 | 6 | 7.30 | 0.40 | 0.735 | 12, 13, 14, 19, 32, 7 |
| 1 | 8 | 6.14 | 0.24 | 0.857 | 23, 3, 36, 37, 38, 4, 6, 9 |
| 2 | 9 | 7.93 | 0.33 | 0.902 | 18, 2, 30, 35, 39, 40, 5, 51, 8 |

3 維特徵 = `y_log` 的 mean / std（訓練期）＋ `y_z` 對 `lag12` 的相關係數（訓練期）；StandardScaler 後 `KMeans(n_clusters=3, random_state=42, n_init=10)`。分群只用 `period <= 202312` 的資料，2024 未參與。

## 6. 產出檔案

```
results/stage0_baselines_extra.csv            主表（model x h）
results/stage0_baselines_extra_perseed.csv    逐種子原表
results/stage0_extra_dm_pooled.csv            pooled DM
results/stage0_extra_dm_series.csv            series-level DM
results/stage0_extra_dm_time.csv              time-aggregated DM
results/extra_lgb_tuning.json                 選中的 LightGBM 參數
results/extra_lgb_tuning_audit.csv            16 組候選的驗證分數
results/extra_series_clusters.csv             K-Means 分群結果
results/extra_train_sec.json                  各模型的實測訓練秒數
results/_extra_cache/*.csv                    每個 (model, seed) 的預測（續跑用）
```

## 7. 各模型實測 train_sec（檔尾）

| 模型 | 種子設定 | 總訓練秒數 | 每次訓練秒數 |
|---|---|---:|---:|
| KMeansRidge_level | deterministic (single run) | 0.90 | 0.90 |
| KMeansRidge_residual | deterministic (single run) | 0.70 | 0.70 |
| kNN_level | deterministic (single run) | 0.40 | 0.40 |
| kNN_residual | deterministic (single run) | 0.40 | 0.40 |
| NLinear_level | deterministic (single run) | 0.40 | 0.40 |
| NLinear_residual | deterministic (single run) | 0.40 | 0.40 |
| DLinear_level | deterministic (single run) | 0.30 | 0.30 |
| DLinear_residual | deterministic (single run) | 0.30 | 0.30 |
| LightGBMTuned_level | 3 seeds [42, 7, 2024] | 46.40 | 15.47 |
| LightGBMTuned_residual | 3 seeds [42, 7, 2024] | 46.20 | 15.40 |

上表的秒數是**實際 fit 的 wall-clock**（單機 CPU，`n_jobs=4`），不含快取讀取時間；由 `results/extra_train_sec.json` 保存，重跑時沿用。
LightGBM 網格搜索另計 1,536 次 fit（見 §4），與上表的 6 次最終 fit 分開。
