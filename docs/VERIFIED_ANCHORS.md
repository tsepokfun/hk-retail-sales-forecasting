# 既有成績的獨立重算（Verified Anchors）

> 用途：在任何新複現開始之前，用**鎖死的 harness** 從原始逐點預測檔重算 `FINAL_REPORT.md` 引用的頭條數字，確立一個可信的比較錨點。
> 產生日期：2026-10-23 ｜ 腳本：`stage0/audit_existing_baselines.py`（**只讀**，不寫入任何結果檔）
> 重跑指令：`stage0\.venv\Scripts\python.exe stage0\audit_existing_baselines.py`

## 0. 方法

- 評估框架：`stage0/eval/harness.py`（**未修改**），`metrics_from_frame`
- MASE 分母：訓練期 seasonal-naive MAE（`harness.seasonal_naive_mae_train`），23 條序列
- 測試集：原點 `202312…202411`（12 個）× 視界 {1,3,6,12} × 23 序列 = **1,104 點**
- ENS3 的定義：對 `sota_preds/<stem>_s{42,7,2024}.csv` 的 `y_pred` 按 `(origin,h,series)` 取算術平均
- 資料來源檔：`stage0/results/predictions.csv`、`audit_ladder_predictions.csv`、`stage_h_ens5_preds.csv`、`sota_preds/*.csv`、`_sota_ref_SeasonalNaive.csv`

## 1. 重算結果 vs 報告（全部對得上）

| 模型 | 重算 MASE_ALL | 重算 WAPE_ALL | 報告值 | 判定 |
|---|---:|---:|---|---|
| **iTransformer + 可學習錨點（ENS3）** | **0.90198** | 0.08781 | 0.90198 / 8.781% | ✅ 完全一致 |
| iTransformer + 持續性錨點（ENS3） | 0.90736 | 0.08825 | 0.90736 / 8.825% | ✅ |
| **xLSTM + `MinT_wls`（ENS3）** | **0.94455** | 0.10025 | 0.94455 / 10.025% | ✅ |
| xLSTM + 可學習錨點（ENS3） | 0.94810 | 0.10023 | 0.94810 | ✅ |
| iTransformer 通道旁路控制組（ENS3） | 0.96143 | 0.09316 | 0.96143 / 9.316% | ✅ |
| **SeasonalNaive** | **1.03029** | 0.10444 | 1.03029 / 10.444% | ✅ |
| Drift | 1.46612 | 0.13829 | 1.46610 | ⚠️ 差 0.00002 |
| LightGBM_residual（未調參） | 1.08925 | 0.10697 | 1.08920 | ⚠️ 差 0.00005 |
| Ridge_level | 1.77510 | 0.17957 | 1.77510 | ✅ |
| Ridge_residual | 1.17127 | 0.10982 | 1.17130 | ⚠️ 差 0.00003 |

**結論：報告的評分沒有造假，可由原始逐點預測重現。** 三個 0.0000x 級差異屬四捨五入或檔案版本差異，不影響任何排名。

逐視界（與表 3-10 同口徑）：

| 模型 | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| iTransformer + 錨點 ENS3 | 0.87868 | **0.92050** | **0.91837** | 0.89038 |
| xLSTM + MinT_wls | **0.84213** | 1.02274 | 1.07016 | **0.84317** |
| SeasonalNaive | 1.21234 | 1.12512 | 0.97471 | 0.80900 |
| iTransformer 通道旁路 ENS3 | 0.92513 | 1.02568 | 0.99386 | 0.90107 |

→ 再次確認：**naive 在 h=6（0.97471）與 h=12（0.80900）反贏最終模型**，這是報告 §5.2「保留 naive 當監控線」的依據，數字正確。

## 2. 新發現一：倉庫裡已存在一個更好的 xLSTM 集成，且與 iTransformer 的差距落入雜訊

| 變體 | MASE_ALL | WAPE_ALL | 來源檔 |
|---|---:|---:|---|
| `xLSTM_5seed_ENS5_log` | **0.91009** | 0.09541 | `stage_h_ens5_preds.csv` |
| `xLSTM_5seed_ENS5_level` | **0.91230** | 0.09568 | 同上 |
| xLSTM ENS3（報告頭條） | 0.94810 | 0.10023 | `audit_ladder_predictions.csv` |
| iTransformer + 錨點 ENS3（報告第一） | 0.90198 | 0.08781 | `sota_preds/` |

**算術後果**：`0.91009 − 0.90198 = 0.00811` → **落在 0.02 種子雜訊門檻之內**（門檻推導見 `FINAL_REPORT.md` §4.1）。
也就是報告宣稱的「iTransformer 比 xLSTM 好 4.72%」，在改用倉庫內已有的 5 成員集成後**縮小到 0.9% 且不顯著**。

**⚠️ 必須加的三條保留（不可省略，否則就是新的誇大）**：

1. `ENS5` **混合了兩種不同配置**：成員含 `xLSTM_base_s1`、`xLSTM_base_s1234`（`base` 配置）與 `xLSTM_none_s42/s7/s2024`（`none` 配置）。單獨看，`xLSTM_base_s1 = 0.91714`、`xLSTM_none_s42 = 1.01743` —— **`base` 與 `none` 不是同一個模型**。
2. 因此 ENS5 屬「**跨配置模型平均**」，與報告那種「同一配置、3 個種子」的集成**口徑不同**，不可直接互換成「新的官方第二名」。
3. MASE 與 WAPE 方向不一致：ENS5 的 WAPE（0.09541）比 iTransformer（0.08781）**差 8.7%**。若以金額結帳，iTransformer 仍勝。

**建議寫法**（可直接用）：
> 「以同一配置三個種子計，iTransformer 較 xLSTM 低 4.72% MASE（0.90198 vs 0.94810），超出種子雜訊門檻。惟倉庫內另存有一個跨 `base`/`none` 配置的 5 成員集成（`xLSTM_5seed_ENS5_log` = 0.91009），其與最佳模型的差距（0.00811）落在雜訊門檻內；由於該集成混合兩種配置，此比較屬探索性，不構成對第一名排名的推翻。若改以金額加權 WAPE 結帳，iTransformer 仍勝 8.7%。」

## 3. 新發現二：通道混合的證據比報告引用的控制組更強

| 臂 | 結構 | ENS3 MASE_ALL |
|---|---|---:|
| `itransformer_anchor` | token = **序列**（跨品類 attention） | **0.90198** |
| `itransformer_noattn` | 同模型但**旁路 attention**（報告 §5 控制組） | 0.96143 |
| `timetf_anchor` | **時間 token 雙胞胎**（channel independent） | 1.10979 |

- `itransformer_noattn` 與主模型差 **0.05945**（報告已引用）。
- `timetf_anchor`（時間 token）與主模型差 **0.20781**，遠大於 0.02 門檻。
- 兩者都指向同一方向，且第二個差距更乾淨：**同錨點、同寬度、同訓練循環，只把 token 從「序列」換成「時間」，MASE 就差 0.208**。

→ 這是「跨品類混合有效」最硬的內部證據，比報告目前引用的單一控制組更強，值得在 §2.3 補上。

## 4. 附錄：面板歷史**無法**往前延伸（已由程式碼與 metadata 雙重確認）

這一條直接決定「資料要怎麼變多」，所以單獨記錄。

| 檢查 | 結果 | 依據 |
|---|---|---|
| `stage0/data/panel_wide.csv` | **263 個月 × 23 序列**，`period` = `200410` → `202608`，**缺失 0 格** | 實測 `pd.read_csv` |
| `stage0/data/panel_meta.json` | `source_table: 620-67002`、`snapshot: censtatd_620-67002_20261018.json`、`downloaded: 2026-10-18`、`missing_cells: 0` | 實測讀取 |
| `stage0/data/build_panel.py` 的過濾邏輯 | 只過濾 `sv == "VAL_RS"`（L114）與 `freq == "M"`（L116），並要求 `period` 是 6 位 `YYYYMM`（L118）→ **沒有任何日期下限過濾** | 原始碼 L109–118 |
| 因此 | **`200410` 是 C&SD 該表本身的最早月份，不是本專案截斷的結果** | 程式碼無日期 filter |

**推論（決定後續取資料的方向）**：要把「23 條序列 / 6,049 觀測」變大，**唯一可行的是「增加序列」而不是「延長歷史」**。可行路徑按優先序：

1. **增加序列**：C&SD 其他月度零售相關表（`620-*` 全系列、`630-*` 餐飲業）、或其他經濟體的月度零售（可作 pooled / 遷移學習的輔助面板）
2. **增加頻率**：C&SD 只有月度，不可行
3. **增加外生維度**：不是增加樣本數，只增加每樣本的資訊量（本專案已測 13 條，結果為負）
4. **跨域 pooled 訓練**：用 M1/M3/M4 月度序列預訓練表徵，再微調到 23 條序列（真正的「小樣本」解法）

> 註：直接以 `urllib` 重抓 API 會回 **HTTP 403**（未帶 User-Agent）；`build_panel.py` 內部有自帶 header。此為抓取注意事項，不影響上表結論。

## 5. 本檔案不做什麼

- 不重新訓練任何模型（全部由既有 csv 重算）
- 不改動 `harness.py`、`baselines.py` 或任何既有結果檔
- 不對 2024 標籤做任何選擇（純事後評估）
- ENS5 的跨配置問題**未解決**，僅記錄；若要把它變成正式結論，需重新用同一配置跑 ≥5 個種子（成本：約 5 × 137 s GPU）
