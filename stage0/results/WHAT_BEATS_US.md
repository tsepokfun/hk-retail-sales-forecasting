# WHAT BEATS US — 方法論審計（外部模型對最終 xLSTM 的實測比較）

**專案**：ITP4524 AI Group Project · Stage 0/Stage 3 預測模型
**審計日期**：本回合（純 CPU：pandas / numpy / scikit-learn / lightgbm；**未訓練任何神經網路、未使用 GPU**）
**被審計對象**：`xLSTM_none`，`purge_val=True`，種子 42/7/2024
**一句話結論**：**沒有任何一個獨立模型在 MASE_ALL 上贏過最終模型（0.94455）**。贏的是**「按視界切換」這個機制**：
2024 上 seasonal naive 在 h6（0.97471）與 h12（0.80900）兩格贏過最終模型（1.07016／0.84317），
但**驗證期（≤2022）只支持 h12 這一格**。把唯一被支持的那一格切換做進選擇器，MASE_ALL 從 0.94455 推到
**0.93601**；代價是同一個切換讓金額加權的 WAPE 從 0.10025 **變差**到 0.10083，且該切點在 2024 測試集上
本身並不顯著（23 序列配對 11 勝 12 敗、p=0.665）。

> 本文件所有數字都來自本回合實際執行的腳本（見 §9 重現方式），沒有一個是估計值或轉抄。
> 文獻表的數字一律標明「文獻結果」，並在 §6 逐條加註。

---

## 0. 先更正任務簡報裡的三個事實前提（都不是小事）

審計的第一個產出是：**簡報本身有三處會把結論帶偏**。

| # | 簡報的說法 | 實測 | 影響 |
|---|---|---|---|
| 0.1 | 「可用 `stage0\results\pv_preds\` 做驗證期選擇依據」 | **不成立**。`pv_preds/xLSTM_*_s{42,7,2024}.csv` 的 `origin` 全部是 `202312..202411`（＝2024 測試期）。枚舉命令：對 `stage*/results/**/*.csv` 逐檔讀 `origin` 欄，見 §9 第 1 步；6 個 pv_preds 檔的 origin 聯集 = {202312, 202401 … 202411}，無任何 ≤2022 的原點。`pv_meta/*.json` 裡的 `val_origins:[202112,202212]` 只是**訓練時 early-stopping 的內部分割**，沒有存逐點預測，也不是可比較的評估集。 | 必須自建驗證區塊（見 §3） |
| 0.2 | 「`kNN_residual` 在 h=12 是 0.83898，**優於 xLSTM 的 0.86294**」 | 0.86294 是**逐種子 MASE 的算術平均**（＝`single_seed_mean`），不是集成。對照對象應該是：ensemble **0.84642**、ensemble+MinT_wls **0.84317**。kNN 的優勢因此從 −0.0240 縮到 **−0.0074（對集成）／−0.0042（對最終模型）**。 | 這個「贏」在 23 條序列的配對檢定下 p=**0.5399**（9 勝 14 敗），時間聚合 p=**0.042**（naive 反而贏）；用 MASE 講它贏是用錯基準 |
| 0.3 | 「`LightGBMTuned_residual` = 1.04108」 | 1.04108 是三個種子**逐種子 MASE 的平均**。其**預測集成**是 **1.02866**（單跑 s42 = 1.03016）。本文件在需要與 `xLSTM_ensemble` 對等的比較裡一律用 3 種子預測平均，並在表內標明。 | 與 §1 的「single_seed_mean ≠ ensemble」是同一個錯誤型態，只是發生在基線端 |

另外兩個順手查到、會影響引用的問題：

* `stage0\results\predictions.csv` 裡的 `xLSTM` 列（MASE_ALL **0.93828**）**不是**可以引用的成績：
  它與 `pv_preds/xLSTM_leaky_s42.csv` 逐格相同（每視界 MASE 0.88044 / 0.94831 / 1.06220 / 0.86216 完全一致），
  也就是 `purge_val=False` 的版本——early stopping 的驗證集就是 2024，2024 分數因此樂觀偏誤。
  同種子的誠實版是 `xLSTM_purged_s42` = **1.01743**。**「舊版比新版好 0.079」是協定假象，不是模型退化。**
  正確的讀法是：`purge_val=True` 讓同種子從 0.93828 退到 1.01743，說明了洩漏能買到多少假精度。
* `DLinear_level` 與 `DLinear_residual` 的逐點預測**最大絕對差 1.36e-11**（1104 格全比），指標全同。
  原因：`_dlinear_design` 的 6 個欄位含 `y_z` 本身，而 trend+seasonal 可精確重建每個欄位，
  所以「預測殘差再加回 y_z」與「直接預測 level」在線性 span 上是**同一個估計量**。
  這兩列在 `stage0_baselines_extra.csv` 裡是**重複的基線，不是兩個獨立證據**。

---

## 1. 表 A — 集成階梯（獨立重算，`stage0\results\ensemble_ladder.csv`）

口徑：`H.metrics_from_frame`，原始 HK\$m 尺度，1104 點/變體（12 原點 × 4 視界 × 23 序列）。
`n` 為該列的點數；`single_seed_mean` 的 MASE/WAPE/bias_pct 是三種子各自計算後的**算術平均**
（WAPE 與 bias_pct 也用平均，因為逐種子都是同一組 y_true，可以平均；這點與「集成」不同）。

| variant | h | MASE | WAPE | bias_pct | n |
|---|---|---|---|---|---|
| single_seed_mean | 1 | 0.89161 | 0.09206 | −3.489 | 276 |
| single_seed_mean | 3 | 1.07002 | 0.11903 | −5.851 | 276 |
| single_seed_mean | 6 | 1.10963 | 0.11808 | −5.347 | 276 |
| single_seed_mean | 12 | 0.86294 | 0.08557 | −2.995 | 276 |
| **single_seed_mean** | **ALL** | **0.98355** | **0.10355** | **−4.409** | **1104** |
| ensemble | 1 | 0.84786 | 0.08882 | −3.489 | 276 |
| ensemble | 3 | 1.02888 | 0.11513 | −5.851 | 276 |
| ensemble | 6 | 1.06923 | 0.11389 | −5.347 | 276 |
| ensemble | 12 | 0.84642 | 0.08361 | −2.995 | 276 |
| **ensemble** | **ALL** | **0.94810** | **0.10023** | **−4.410** | **1104** |
| ensemble+BU | 1 | 0.85616 | 0.08962 | −3.274 | 276 |
| ensemble+BU | 3 | 1.02643 | 0.11283 | −5.603 | 276 |
| ensemble+BU | 6 | 1.07850 | 0.11477 | −5.087 | 276 |
| ensemble+BU | 12 | 0.84532 | 0.08418 | −3.037 | 276 |
| **ensemble+BU** | **ALL** | **0.95160** | **0.10023** | **−4.240** | **1104** |
| ensemble+MinT_ols | 1 | 0.93074 | 0.08979 | −3.617 | 276 |
| ensemble+MinT_ols | 3 | 1.10914 | 0.11678 | −5.947 | 276 |
| ensemble+MinT_ols | 6 | 1.17867 | 0.11592 | −5.479 | 276 |
| ensemble+MinT_ols | 12 | 0.84980 | 0.08323 | −2.977 | 276 |
| **ensemble+MinT_ols** | **ALL** | **1.01709** | **0.10129** | **−4.494** | **1104** |
| ensemble+MinT_wls | 1 | 0.84213 | 0.08831 | −3.416 | 276 |
| ensemble+MinT_wls | 3 | 1.02274 | 0.11499 | −5.816 | 276 |
| ensemble+MinT_wls | 6 | 1.07016 | 0.11485 | −5.304 | 276 |
| ensemble+MinT_wls | 12 | 0.84317 | 0.08340 | −3.002 | 276 |
| **ensemble+MinT_wls** | **ALL** | **0.94455** | **0.10025** | **−4.374** | **1104** |

**核對結果：兩項引用數字完全吻合（差 0.00000）**
`ensemble = 0.94810`（重算 0.94810）、`ensemble+MinT_wls = 0.94455`（重算 0.94455）、
逐視界 0.8421 / 1.0227 / 1.0702 / 0.8432 也完全一致。所以報告目前的引用**不是**把
「逐種子平均」誤當集成——這個坑出現在別處（§0.2、§0.3）。

值得注意的三件事（都是本表自己說出來的）：

1. **集成賺 0.03545**（0.98355 → 0.94810）。三種子 MASE_ALL 的標準差是 **0.0301**（1.01743 / 0.95989 / 0.97333，
   sd=0.0301，全距 0.0575），所以「集成」大約等於 1.2 個種子標準差——真實但邊際。
2. **MinT_ols 是災難**：0.94810 → **1.01709**（+0.069）。W=I 的等權投影把誤差推給高變異序列；
   只有用訓練期 naive 誤差變異數當權重的 MinT_wls 才微幅改善（−0.00355）。
3. **BU 與 MinT_wls 在 MASE 上反向**（BU 0.95160 vs MinT_wls 0.94455），
   但兩者的 WAPE **完全相同 0.10023**。這正是本任務提醒的指標分歧：調和主要在動「序列等權」這一側。

---

## 2. 逐點預測補齊（步驟 2 產出）

| 模型 | 檔案 | 列數 | 重算 MASE_ALL | 與 `stage0_baselines_extra.csv` | 耗時 |
|---|---|---|---|---|---|
| kNN_residual | `extra_preds_kNN_residual.csv` | 1104 | 1.02377 | 完全一致 | 1.9 s |
| KMeansRidge_residual | `extra_preds_KMeansRidge_residual.csv` | 1104 | 1.08189 | 完全一致 | 0.9 s |
| NLinear_residual | `extra_preds_NLinear_residual.csv` | 1104 | 1.15128 | 完全一致 | 0.4 s |
| DLinear_residual | `extra_preds_DLinear_residual.csv` | 1104 | 1.34248 | 完全一致 | 0.2 s |
| LightGBMTuned_residual | `extra_preds_LightGBMTuned_residual.csv`（＋`_s7`、`_s2024`） | 1104 | 1.03016（s42） | 逐種子一致；檔案中的 1.04108 是三種子平均 | 15.9 s |

超參數直接讀 `results/extra_lgb_tuning.json`（原本就是 train-only 調出來的），本回合**沒有重新調參**，
所以 2024 標籤沒有進入任何選擇。5 個模型都只跑 `target_mode="residual"`。
`DLinear_*` 兩列同值的原因見 §0。

---

## 3. 表 B — 按視界選模：四欄對比（本任務最重要的一張表）

### 3.1 可部署欄用了什麼依據（必讀）

**只能用「原點 ≤ 2022-12 的驗證表現」。** 但整個 repo **不存在**任何 xLSTM 在 2022 原點上的逐點預測：

* 枚舉結果（§9 第 1 步）：所有含 `origin` 欄的結果檔，origin 聯集只有兩種 ——
  `202312..202411`（測試期）與 **`201901..202111`**（`stage2_oof_s{42,7,2024}.csv`，xLSTM 的 out-of-fold 預測）。
* 要造出 2022 原點的 xLSTM 預測必須重訓網路 → 本任務明令禁止。**所以 xLSTM 的驗證證據只有一個區塊：V1。**

**V1 區塊（唯一含 xLSTM 的驗證證據）**
起源 2019-01 … 2021-11（35 個原點），所有候選**一律凍結在 `target_period ≤ 2017-12` 訓練**（每視界 3082–3335 列）。
xLSTM 用 `stage2_oof`：同一個 `XLSTMForecaster` 類別（`use_anchor=True, context=12, embedding=32, blocks=2`，
與 `xLSTM_none` 完全同構，只差訓練窗），三種子（42/7/2024）預測平均。CPU 候選由本回合用**完全相同**的
凍結窗協定重跑（`frozen_cpu`），避免「xLSTM 凍結、對手滾動」這種對對手有利的偏差。

**V2 區塊（第二意見，但 xLSTM 缺席）**
起源 2022-01 … 2022-12（目標月 ≤ 2023-12），每個原點重訓 → 貼近部署協定、且是後疫情常態。
但此區塊**沒有 xLSTM 可點預測**，只能用來檢查「CPU 候選之間的排序是否穩定」。

**選擇規則（明確寫出，可被反駁）**
> 每個 h：取 V1 MASE 最小者；但**只有**當該候選對 `xLSTM_ensemble` 的
> 序列層級配對 Wilcoxon（N=23，per-series MAE）p<0.05 時才真的切換，否則維持 xLSTM。
> 規則只讀 V1（≤2021 標籤）與 CPU 候選的凍結窗重跑，**沒有讀任何 2024 標籤**。

規則執行的結果（`results/audit_selector_rule.json`）：

| h | V1 argmin | V1 MASE（argmin） | V1 MASE（xLSTM incumbent） | 亞軍 MASE | Wilcoxon p vs incumbent | 決定 | 部署 |
|---|---|---|---|---|---|---|---|
| 1 | xLSTM_ensemble | 1.06367 | 1.06367 | 1.20163 (LGBMTuned) | — | keep incumbent | xLSTM |
| 3 | xLSTM_ensemble | 1.54848 | 1.54848 | 1.79247 (LGBMTuned) | — | keep incumbent | xLSTM |
| 6 | xLSTM_ensemble | 1.98123 | 1.98123 | 2.18021 (LGBMTuned) | — | keep incumbent（見下註） | xLSTM |
| 12 | **SeasonalNaive** | **2.11690** | 2.26764 | 2.19215 (LGBMTuned) | **0.001279** | **switch** | **SeasonalNaive** |

「驗證期能不能可靠區分？」用 V1 的**種子雜訊帶**回答（xLSTM 逐種子 V1 MASE，sd）：h1 **0.049**、h3 **0.077**、
h6 **0.015**、h12 **0.028**。h12 的差距 0.151 是種子 sd 的 5.4 倍，加上 19/23 序列勝、p=0.0013
→ **h12 的切換是可靠的**（在 V1 這個制度內）。h6 的 MASE 差距 0.199 雖然遠超 sd=0.015，
但**序列層級配對檢定不顯著**（勝 17/23、p=0.0749）；兩種證據打架，因此**不切換**——這是保守的選擇，
代價見 §3.3。

V2（CPU-only）的排序（`results/audit_val_v2.csv`）只在「贏過 naive」這件事上與 V1 一致，
**不能用來選 h1/h3**：V2 的 argmin 是 naive（h1 0.86019、h3 1.06001），因為該區塊裡沒有 xLSTM 當對手。
若誤用 V2 決策，2024 的 h1/h3 會從 0.84786/1.02888 掉到 1.21234/1.12512。**這個坑必須寫進報告。**

### 3.2 四欄（＋兩個必要的補充列）

2024 測試集，1104 點，`H.metrics_from_frame`。MASE 為序列等權，WAPE 為金額加權。

| 欄位 | variant | h1 | h3 | h6 | h12 | **MASE_ALL** | WAPE_ALL | bias_pct |
|---|---|---|---|---|---|---|---|---|
| 純 xLSTM（集成） | `xLSTM_ensemble` | 0.84786 | 1.02888 | 1.06923 | 0.84642 | **0.94810** | 0.10023 | −4.410 |
| 純 SeasonalNaive | `SeasonalNaive` | 1.21234 | 1.12512 | 0.97471 | 0.80900 | **1.03029** | 0.10444 | −5.707 |
| **可部署按視界選模** | `Deployable_V1_per_h` | 0.84786 | 1.02888 | 1.06923 | **0.80900** | **0.93874** | 0.10076 | −3.738 |
| **可部署（先調和再混合）** | `Deployable_V1_per_h_MinT` | 0.84213 | 1.02274 | 1.07016 | **0.80900** | **0.93601** | 0.10083 | −3.700 |
| **oracle 上界（按 MASE，不可部署）** | `Oracle_per_h_MASE` | 0.84213 | 1.02274 | **0.97471** | **0.80900** | **0.91215** | 0.09674 | −3.988 |
| 補充：最終模型 | `xLSTM_ensemble_MinT_wls` | 0.84213 | 1.02274 | 1.07016 | 0.84317 | **0.94455** | 0.10025 | −4.374 |
| 補充：oracle 若改按 WAPE 選（不可部署） | `Oracle_per_h_WAPE` | 0.88617 | 1.04242 | 0.97471 | 0.84317 | **0.93662** | **0.09378** | −3.189 |

**Oracle 選了誰（2024 標籤，不可部署，只為量化「切換最多能賺多少」）**

* 按 MASE：h1 → `xLSTM_ensemble_MinT_wls`（0.84213）、h3 → `xLSTM_ensemble_MinT_wls`（1.02274）、
  **h6 → `SeasonalNaive`（0.97471）**、**h12 → `SeasonalNaive`（0.80900）**。
* 按 WAPE：h1 → `LightGBMTuned_residual`（0.08797）、h3 → `LightGBMTuned_residual`（0.10574）、
  h6 → `SeasonalNaive`（0.09832）、h12 → `xLSTM_ensemble_MinT_wls`（0.08340）。
* **兩個 oracle 只差 0.0245（0.91215 vs 0.93662），而且選出來的組合幾乎相反**（MASE-oracle 在 h12 要 naive、
  WAPE-oracle 在 h12 要 xLSTM）。也就是說：**「按視界切換」的可賺幅度本身高度取決於你用哪個指標結帳**。

### 3.3 這張表該怎麼讀（誠實版）

* 可部署欄的 **0.93601（比最終模型好 0.00854）全部來自單一切點 h12 → naive**；h1/h3/h6 三欄原封不動。
* 這個切點在 **V1 上很硬**（19/23 序列、p=0.0013、差距 5.4 個種子 sd），但在 **2024 測試集上不顯著**：
  11 勝 12 敗、p=**0.665**（配對序列）；時間聚合 7/12 原點、p=**0.5693**（`harness.dm_time_aggregated`）。
  所以「0.00854 的改善」在統計上尚未成立，它是一個**有先驗根據但尚未被測試集確認**的切換。
* 同一刀讓 **WAPE 變差**（0.10025 → 0.10083）。原因見 §4。
* 距 oracle 還差 **0.02386**（0.93601 → 0.91215），而缺口幾乎全部來自 **h6**：V1 說 xLSTM 在 h6 最好
  （差 0.199 > 種子 sd 0.015），2024 卻說 naive 好 0.0945。**h6 是「驗證期學不到、測試期才反轉」的那一格**，
  可部署選擇器在這裡必然虧錢。

---

## 4. 指標矛盾：MASE 與 WAPE 方向相反的地方（兩個都報）

MASE 對 23 條序列**等權**；WAPE 用 |y_true| 加權，於是被序列 2/18/35/40/32 這種上千 HK\$m 的序列主導。

| 比較 | h | MASE 誰贏 | WAPE 誰贏 | 機制 |
|---|---|---|---|---|
| `SeasonalNaive` vs `xLSTM_ensemble` | 6 | naive **0.97471** vs 1.06923 | naive **0.09832** vs 0.11389 | 同向，naive 兩邊都贏 |
| `SeasonalNaive` vs `xLSTM_ensemble` | 12 | naive **0.80900** vs 0.84642 | **xLSTM 0.08361** vs 0.08569 | **反向**：naive 在 12/23 條序列（多為中小序列）小贏，但在大序列上絕對誤差較大 → 金額加權翻盤 |
| `kNN_residual` vs `xLSTM_ensemble` | 12 | kNN **0.83898** vs 0.84642 | **xLSTM 0.08361** vs 0.09157 | **反向且金額差距大（−9.5%）**：kNN 的 h12 優勢只存在於序列等權的世界 |
| `LightGBMTuned_residual` vs `xLSTM_ensemble` | 3 | **xLSTM 1.02888** vs 1.04242 | **LGBMTuned 0.10574** vs 0.11513 | **反向**：LGBMTuned 的測試期 bias −1.74% vs xLSTM −5.85%，少掉的系統性低估直接反映在金額上；但它在小序列上更差 |
| `SeasonalNaive` vs `xLSTM_ensemble` | ALL | xLSTM **0.94810** vs 1.03029 | xLSTM **0.10023** vs 0.10444 | 同向，但差距由 8.0% 縮到 4.0%——**模型的優勢有一半在中小序列上** |

**不要只挑對自己有利的那一個。** 對本專案最不利的讀法是：
以金額結帳（WAPE，那才是零售預算真正的損失函數）時，xLSTM 只比 seasonal naive 好 **4.0%**；
以序列結帳（MASE，報告的主指標）時好 **8.0%**。

---

## 5. 表 C — 逐序列弱項與結構性結論（`results/audit_series_weakness.csv`）

比較 `xLSTM_ensemble` vs `SeasonalNaive`，2024 測試集，逐序列 MASE（分母 = 該序列訓練期 naive MAE）。
`rel_vs_naive_pct > 0` 表示 **xLSTM 比 naive 差**。

### 5.1 xLSTM 比 naive 差 >10% 的序列（9/23）

| series | 父類 | 2024 平均 \|y\| (HK\$m) | MASE xLSTM | MASE naive | xLSTM 差 naive | 佔 xLSTM 總絕對誤差 |
|---|---|---|---|---|---|---|
| 3 | 2 的子 | 758.6 | 0.9342 | 0.5260 | **+77.6%** | 1.11% |
| 36 | 35 的子 | 517.5 | 2.1175 | 1.3377 | **+58.3%** | 1.71% |
| 51 | 18 的子 | 3085.9 | 1.0977 | 0.7417 | **+48.0%** | 12.56% |
| 6 | 2 的子 | 938.8 | 0.9310 | 0.6302 | **+47.7%** | 1.48% |
| 40 | 35 的子 | 4308.2 | 1.3627 | 0.9563 | **+42.5%** | 7.34% |
| 8 | —（獨立） | 4182.1 | 0.8229 | 0.6552 | **+25.6%** | 3.85% |
| 35 | **父類** | 8202.8 | 0.6462 | 0.5224 | **+23.7%** | 9.07% |
| 2 | **父類** | 3081.0 | 0.8402 | 0.7306 | **+15.0%** | 3.36% |
| 18 | **父類** | 4639.3 | 1.2238 | 1.0897 | **+12.3%** | 17.76% |

### 5.2 結構性結論：**是，而且非常乾淨——集中在 {2, 18, 35} 這三個階層區塊**

用 `run_stage4_reconcile.HIER` 的分組去看：

| 階層區塊 | 成員 | 9 條弱項裡佔幾條 | 區塊內表現 |
|---|---|---|---|
| **{2}** = 3+4+5+6+7 | 2,3,4,5,6,7 | **3**（3、6、2） | 父類與 2 個子都變差 |
| **{18}** = 19+23+51 | 18,19,23,51 | **2**（51、18） | 父類與 1 個子變差 |
| **{35}** = 36+37+38+39+40 | 35,36,37,38,39,40 | **3**（36、40、35） | 父類與 2 個子變差 |
| **{12}** = 13+14 | 12,13,14 | **0** | 三條全部大幅改善（−21.1% / −26.9% / −30.3%） |
| 不屬於任何父類 | 8,9,30,32 | 1（8） | — |

* **9 條弱項中有 8 條落在 {2}、{18}、{35} 三個區塊**，而**唯一全勝的區塊是 {12}**。這不是隨機分布。
* **4 個父類有 3 個在弱項名單裡**（2、18、35；12 是改善 −26.9%），且 **4 個父類合計吃掉 xLSTM 總絕對誤差的 36.0%**
  （它們只佔 4/23 = 17.4% 的序列數）。**「子序列學得到、父類加總學不到」的結構仍然存在**——
  這正好解釋為什麼 MinT 調和（把父類由子類重建）能把 MASE 從 0.94810 拉到 0.94455。
* 「是否集中於**大額**序列？」→ **不顯著**。序列平均金額與「比 naive 差多少」的 Spearman rho = **+0.345（p=0.107）**。
  方向對（越大越容易吃虧），但 23 條序列撐不起這個結論。**結構性在「階層角色」，不在「金額大小」。**
  證據：序列 12（3613 HK\$m）與 32（4211 HK\$m）都是大額且大幅改善；序列 3（759 HK\$m）是小額卻最差。

---

## 6. 表 D — 文獻定位（**每一條都是文獻的結果，不是本專案實測**）

檢索方式：arXiv API ＋ OpenAlex API（本回合實際發出請求並保存原始 JSON 於 `results\_lit\`）。
ScienceDirect 對本機回 403，因此凡只刊在 Elsevier 的條目**只引用可取得的摘要文字，不引用其結果表**。

| # | 文獻（URL） | 文獻的結果（**非本專案**） | 對我們的意涵 |
|---|---|---|---|
| 1 | Zeng, Chen, Zhang, Xu, *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023) — https://arxiv.org/abs/2205.13504 | 論文摘要原文：「LTSF-Linear surprisingly outperforms existing sophisticated Transformer-based LTSF models **in all cases**, and often by a large margin」（9 個真實資料集）。DLinear/NLinear 就是這組 one-layer linear 模型。 | 「線性模型打贏大模型」在 LTSF 上是**已被發表的主流結論**，不是我們的怪現象 |
| 2 | Elsayed, Thyssens, Rashed, Jomaa, Schmidt-Thieme, *Do We Really Need Deep Learning Models for Time Series Forecasting?* — https://arxiv.org/abs/2101.02118 | 把 8 個 SOTA DNN 與 window-based GBRT 放在 9 個資料集上比，摘要原文：GBRT「outperform all state-of-the-art DNN models evaluated in this paper」。 | GBDT 打贏深度模型有直接的公開證據；我們的 LightGBMTuned 只差 xLSTM 0.080 MASE 與此一致 |
| 3 | Makridakis, Spiliotis, Assimakopoulos (2018), *Statistical and Machine Learning forecasting methods: Concerns and ways forward*, PLoS ONE — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889 | 1045 條 **M3 月度**序列：ML 方法「dominated across both accuracy measures used and **for all forecasting horizons**」，且計算需求顯著更高。 | 與我們最接近的設定（月度、小樣本、多序列）上，ML 輸給統計方法是有前人證據的 |
| 4 | Makridakis, Spiliotis, Assimakopoulos (2022), *Statistical, machine learning and deep learning forecasting methods: Comparisons and ways forward*, JORS — https://doi.org/10.1080/01605682.2022.2118629 | **反向證據**：DL 模型的**組合**優於多數統計/ML 方法，「especially for the case of monthly series and long-term forecasts」，代價是計算時間大增。 | 這是唯一明確指出「月度＋長視界 DL 有優勢」的可取得證據，正好對上我們的 h6/h12 弱點；不能只引對自己有利的 #3 |
| 5 | *M5 accuracy competition: Results, findings, and conclusions*, IJF 2022 — https://doi.org/10.1016/j.ijforecast.2021.11.013 | 零售階層銷售（Walmart，42,840 條序列）。**我只取到摘要**，摘要說明競賽規模與產出但不指名得獎方法；全文未取回，**因此本表不宣稱 M5 冠軍用什麼方法**。 | 我們的問題設定（零售＋階層＋月度）在文獻裡是 M5 家族；但方法名單需要全文才能引用 |
| 6 | *Comparative analysis of modern machine learning models for retail sales forecasting*, Croatian ORR (2026) — https://doi.org/10.17535/crorr.2026.0021 （OA: https://hrcak.srce.hr/file/500982） | 零售資料（間歇性需求、大量缺失、商品流轉快）上比較統計基線、樹集成（XGBoost/LightGBM）與深度架構（N-BEATS、N-HiTS、TFT）：摘要原文「Localized tree-based methods achieve superior performance, with XGBoost attaining the lowest RMSE of **4.833**」，神經網路「remained inferior to ensemble methods」。 | **最貼近「零售上用 GBDT 打贏深度模型」的公開證據**。其 RMSE=4.833 是**該文的資料**，與我們的 HK\$m 不可比 |
| 7 | *A Survey of Machine Learning Methods for Time Series Prediction*, Applied Sciences 15(11):5957 (2025) — https://doi.org/10.3390/app15115957 | 摘要原文：樹方法（如 LightGBM）與 RNN 表現最好，且「tree-based methods offering a significant advantage in terms of their computational efficiency」。 | 與我們的成本實測（LightGBM 15.9 s CPU vs xLSTM 137–153 s GPU/種子）同方向，但**秒數是我們的，不是該文的** |
| 8 | *Applying k-nearest neighbors to time series forecasting: two new approaches*, J. Forecasting (2024) — https://arxiv.org/abs/2103.14200 ／ https://doi.org/10.1002/for.3093 | kNN 的參數調校方法，實例資料包含「retail and food services sales in the USA」的月度序列，對照 Seasonal ARIMA / Holt-Winters / ETS。 | 我們 kNN 在 h12 的 MASE 0.83898 有文獻上的近親，但該文沒有與深度模型對照，**不能拿來聲稱 kNN > xLSTM** |
| 9 | Beck et al., *xLSTM: Extended Long Short-Term Memory* — https://arxiv.org/abs/2405.04517 | 摘要的評估全部是**語言建模／大規模序列**（exponential gating、sLSTM/mLSTM、與 Transformer/SSM 比較）。摘要中**沒有任何 T<20 的月度零售小樣本實驗**。 | 「短序列（T<20）xLSTM vs nn.LSTM 的成本效益」**文獻沒有覆蓋**——這是缺口，不是我們可以引用的結論 |
| 10 | Lara-Benítez, Carranza-García, Riquelme, *An Experimental Review on Deep Learning Architectures for Time Series Forecasting* — https://arxiv.org/abs/2103.12057 | 7 類深度架構、50,000+ 序列、訓練 38,000+ 模型：摘要原文「LSTM and CNN are the best alternatives, with LSTMs obtaining the most accurate forecasts」。 | **對我們有利的反向證據**：深度架構之間確實有差、且 LSTM 系最準 → 與我們 xLSTM 0.94810 勝過 nn.LSTM 1.02272 的實測一致。但該文的比較對象是別的深度模型，不是 seasonal naive |

**成本／效益的實測（我們的數字，不是文獻的）**：完成同一套 12 原點 × 4 視界協定，
`xLSTM_none` 每種子 **137.2 s（`exog_meta_z/xLSTM_none_s42.json` 的 `train_sec`，cuda）**，另一個協定計時檔
`pv_meta/xLSTM_purged_s42.json` 記 152.5 s；對照 `LSTM_none` **21.6 s（GPU）**、
`LightGBMTuned_residual` **15.9 s（CPU，本回合實測）**、`kNN_residual` **1.9 s（CPU）**、`KMeansRidge` 0.9 s、
`NLinear` 0.4 s、`DLinear` 0.2 s（CPU）。xLSTM 用 **6.4–7.1 倍於 nn.LSTM、約 8.6 倍於 LightGBM** 的計算，
換到的是 MASE_ALL 0.94810 vs 1.02272（LSTM）/ 1.02866（LightGBM）——約 **7.3–7.8%** 的改善。

---

## 7. 結論段（照模板填寫）

> 在我們的 2024 測試集上，明確贏過最終模型（ensemble+MinT_wls = 0.94455）的是 **沒有任何一個獨立模型**——
> 唯一在 MASE_ALL 上更低的是 (a) 本審計的 oracle 按視界混合 0.91215，但它用了 2024 標籤、**不可部署**；
> (b) 同一架構在 `purge_val=False` 下的舊種子 42 執行 0.93828，其 early-stopping 看過 2024，
> 分數樂觀偏誤，**不能算贏**。
> 只在個別視界贏過、整體沒贏的是 **SeasonalNaive（h6 0.97471、h12 0.80900，兩格都贏最終模型的 1.07016/0.84317）、
> kNN_residual（僅 h12 的 MASE 0.83898，但同格 WAPE 0.09157 輸給 0.08340，且 23 序列配對 p=0.54 不顯著）、
> 以及 LightGBMTuned_residual（僅 h3 的 WAPE 0.10574 贏 0.11499，同格 MASE 反而輸 1.04242 vs 1.02274）**。
> 無法判斷（沒跑）的是 **xLSTM 在任何 2022 原點上的逐點預測（需重訓神經網路，本任務禁止，故 2022 驗證區塊
> 沒有 xLSTM 可比較）、以及「h6/h12 上 naive 是否結構性更好」——2024 的 23 序列配對檢定 p=0.665、
> 時間聚合 p=0.569，測試集本身不足以確認**。
> 可部署的按視界選模把 MASE_ALL 從 **0.94455** 推到 **0.93601**，距 oracle 上界 **0.91215** 還差 **0.02386**——
> 但這 0.00854 的改善全部來自單一 h12 切換，它在測試集上不顯著，且讓 WAPE 由 0.10025 惡化到 0.10083。
> 我們的模型在文獻座標中大約落在 **無法判斷**——本專案是 23 條香港零售月度序列、12 個原點、4 個視界、
> 以自家 seasonal naive 為分母的 MASE，與 LTSF／M5 的資料集、視界長度與指標都不可換算；
> 可說的只有方向：**「簡單模型在月度零售上打贏深度模型」是文獻的主流結論之一（表 D #1/#2/#3/#6/#7），
> 而「DL 組合在月度長期視界有優勢」也有證據（#4）——本專案的 h1/h3 贏、h6/h12 輸，正好落在這兩股力量中間。**

---

## 8. 缺席清單（哪些候選模型拿不到逐點預測，或根本沒跑）

| 候選 | 狀態 | 依據 |
|---|---|---|
| `xLSTM` 在 2022 原點的逐點預測 | **缺席**（需重訓 NN → 禁止） | §9 第 1 步的 origin 枚舉：所有 xLSTM 預測檔只有 2024 原點；唯一非 2024 的是 `stage2_oof`（2019–2021） |
| 任何 2024 以外、2022 以後的「驗證期」點預測 | **缺席** | 同上 |
| M5 / LTSF 公開基準上的本模型重評 | **沒跑**（無該資料、無授權、且與本專案定義不同） | — |
| `LightGBMTuned_residual` 在 V1 的純淨超參 | **部分可用**：V1 用的是 `extra_lgb_tuning.json` 的參數，該組參數是 train-only 但選參時看過 ≤2023-12 的標籤 | §3.1；不影響硬約束（未用 2024），但嚴格說 V1 對該候選不是「全盲」 |
| `Drift` 在 V1/V2 | V2 未納入（`run_drift` 的 drift 用 ≤TRAIN_END 資料算，對 2022 原點是未來資訊 → 剔除）；V1 由同一理由剔除 | 誠實處理，不硬塞 |
| `KMeansRidge/NLinear/DLinear/Ridge` 在 V1 | 本回合只把 4 個主候選放進 V1（決策只需要它們）；其餘在 V2 有 | `results/audit_val_v2.csv` |
| 三種子以外的 xLSTM 種子 | **沒跑**（時間） | — |

---

## 9. 重現方式（每條都是實際執行過的）

```
cd stage0
# 1) 枚舉所有含 origin 欄的結果檔（決定「驗證期到底有哪些原點」）
.venv\Scripts\python.exe -c "import pandas as pd,glob; [print(f, pd.read_csv(f,dtype={'origin':str})['origin'].min(),'..',pd.read_csv(f,dtype={'origin':str})['origin'].max()) for f in sorted(set(glob.glob('stage*/results/**/*.csv',recursive=True))) if 'origin' in pd.read_csv(f,nrows=1).columns]"
# 2) 表 A
.venv\Scripts\python.exe audit_ladder.py          -> results\ensemble_ladder.csv
# 3) 5 個基線的逐點預測
.venv\Scripts\python.exe audit_extra_preds.py     -> results\extra_preds_*.csv
# 4) 步驟 3/4/5 的全部計算
.venv\Scripts\python.exe audit_selection.py       -> audit_model_by_h.csv / audit_oracle*.csv
                                                     audit_val_v1.csv / audit_val_v2.csv
                                                     audit_series_weakness.csv
# 5) V1 種子雜訊帶
.venv\Scripts\python.exe audit_v1_seedspread.py   -> results\audit_v1_seed_spread.csv
# 6) 表 B 與測試集顯著性
.venv\Scripts\python.exe audit_tableB.py          -> results\audit_tableB.csv / audit_test_dm.csv
# 7) harness 自帶的檢定（不是自製檢定）
.venv\Scripts\python.exe audit_dm.py              -> audit_dm_series_official.csv / audit_dm_time_official.csv
# 8) 文獻
.venv\Scripts\python.exe audit_lit_fetch.py       -> results\_lit\*.json（OpenAlex 原始回傳）
```

**沒有修改 `stage0\eval\harness.py`**：本回合寫入的檔案只有 `stage0\audit_*.py` 與 `stage0\results\*`；
`harness.py` 的 `LastWriteTime` 是 2026-10-18 21:11（早於本回合），`models\baselines.py` 為 2026-10-22 16:46
（同樣早於本回合第一個腳本 17:02），兩者皆未在本回合被寫入。沒有訓練任何神經網路；
沒有在 oracle 以外的任何地方使用 2024 標籤做選擇。
