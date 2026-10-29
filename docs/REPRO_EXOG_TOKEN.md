# 外生變數當「額外 token」：iTransformer 上的受控消融

> 產生日期：2026-10-23 ｜ 腳本：`stage0/run_exog_token.py`
> 重跑：`stage0\.venv\Scripts\python.exe stage0\run_exog_token.py exog2|exog8|control`
> 每組 3 個種子 × 100 epochs，GPU，**約 21 秒／組**

## 要驗證的設計主張

`docs/DESIGN_FUSION_PROPOSAL.md` 主張：在這個 23 條序列的面板上，加異質資料的正確方式**不是加寬每個 token 的特徵向量**（xLSTM+FiLM 走的路，代價 +32 參數/欄，而 iTransformer 只有 182 個面板窗口），而是**加 token**——因為 iTransformer 的 variate 投影是**跨 token 共享**的，額外 token 幾乎免費。

本檔案就是這個主張的直接測試：

```
tokens = [23 條零售序列] + [K 條外生序列]
```

外生 token 用自己的 12 個月窗，走**同一個** `Linear(L → d_model)` 與**同一套** encoder，其輸出頭**丟棄**（loss 只算 23 個核心 token）。

- ❌ **不是** TimeXer 的 cross-attention 版（那要 ~12k 參數/模態）
- ✅ 是「便宜 token 融合」版

## 協定

沿用 `stage0/run_sota_baselines.py`：context 12、100 epochs、`val_months=24`、purged validation、測試原點 `202312…202411`、h ∈ {1,3,6,12}、23 序列、1,104 點、MASE 分母為訓練期 seasonal-naive MAE、3 個種子（42/7/2024）平均（ENS3）。

⚠️ **本腳本自建的 origin plan 得到 191–195 個訓練原點，而專案既有 iTransformer 用的是 182 個**。因此**跨管線比較（vs 0.90198）只作參考，本檔案的主結論一律用同管線的 control 對照**。

外生資料來自 `stage0/data/exog_wide.csv`（已含公布滯後，專案以 `verify_exog_crosscheck.py` 驗證過因果性），**不再額外 shift**，以與 FiLM 消融可比。

## 結果（1,104 點，ENS3）

| 配置 | token 數 | **參數量** | 訓練原點 | **MASE_ALL** | WAPE_ALL |
|---|---:|---:|---:|---:|---:|
| **control**（2 個 N(0,1) 噪音 token） | 25 | **26,020** | 195 | **0.93408** | 0.09076 |
| **exog2**（訪客 + MTR） | 25 | **26,020** | 191 | **0.90247** | 0.08735 |
| **exog8**（全部 8 條外生） | 31 | **26,020** | 191 | **0.92640** | 0.08889 |
| （參考）專案既有 iTransformer+錨點 ENS3 | 23 | 26,024 | 182 | 0.90198 | 0.08781 |

## 三個結論

### 1. 機制獲得證實：額外 token 在參數上真的免費

`25 tokens`、`31 tokens` 與 `23 tokens` 的**參數量都是 26,020–26,024**。
→ 加 8 條外生序列**增加 0 個參數**，證實 iTransformer 的 variate 投影跨 token 共享。這正是「加 token、不加寬」在成本上的依據。

### 2. 「加 token」不傷，而「加寬特徵」會傷——這是本次最重要的對比

| 注入方式 | 結果 | 出處 |
|---|---|---|
| **FiLM 加寬**（xLSTM，8 欄外生） | MASE **1.19421 vs 0.98355**（無外生）→ **+21.4% 更差** | `FINAL_REPORT.md` 表 3-6 |
| **token 融合**（iTransformer，8 條外生） | MASE **0.92640 vs 0.93408**（噪音對照）→ **−0.8% 更好** | 本檔案 |

同樣的資料、同樣是「有害/無害」的判準，**換個注入位置，結果從大幅有害翻轉為略微有益**。

### 3. 但外生 token 贏不了「模型自己」

- exog2（**0.90247**）優於噪音對照 control（**0.93408**）**0.03161 MASE（−3.4%）**，超出 0.02 種子雜訊門檻 → **訪客與 MTR 確實帶有資訊**。
- 然而 exog2（0.90247）與**專案既有 iTransformer（0.90198）幾乎相同**（差 0.00049）。
- → **那些資訊本來就已經被 23 條零售序列的跨品類注意力吸收掉了**（訪客與零售額本來就同步）。外生 token 沒有帶來新資訊，只是換了個入口。

**這與 xLSTM 那邊的結論一致**：`visitor_mtr` 在 xLSTM 裡是「有微弱訊號但仍輸給 0 欄」（表 3-6）。兩代模型都指向同一件事：**在這份資料上，外生變數的邊際價值接近零**。

## 對報告與提案的影響

| 現有說法 | 更新後 |
|---|---|
| 「外生變數無用」（表 3-6 的讀法） | **不精確**。應改為：「在 FiLM 加寬注入下有害（+21.4%）；改為 token 融合後**轉為有益但幅度微弱**（−0.8% vs 同寬噪音），且不優於模型自學」 |
| 「加 token 幾乎免費」（設計提案） | **成立**：8 條外生 token 增加 0 參數 |
| 融合提案的預期收益 | **需下修**：成本確實近乎零，但**收益也近乎零**；若要做，應先換更貼近需求的資料源（HKEXnews 同店銷售、GDELT），而不是再加宏觀變數 |

## 限制（必須一起寫）

1. **origin plan 與專案既有 182 原點不同**（本腳本 191–195），故 `exog2` vs `0.90198` 的比較不是嚴格受控；有效比較只有 `exog2` vs `control`（同管線）。
2. **`exog8` 的訓練原點比 control 少 4 個**（2005-09..12 因訪客資料 NaN 被排除），方向上對 control 有利（更多資料），但 control 仍較差。
3. **只測了一種 token 融合實作**（共享投影 + 丟棄外生輸出頭）。未測：外生 token 也參與 loss、外生 token 專用投影、加 register token。
4. **未測 `MinT_wls` 調和後的版本**（本消融只報原始預測）。
5. **單一測試年（2024）**，與專案其他結果共享同一限制。
6. 本消融的 `control`（0.93408）比專案既有基線（0.90198）差，說明**管線差異本身就有 ~0.03 MASE 的影響**——這也提醒：跨腳本比較必須謹慎。

## 產出檔案

| 檔案 | 內容 |
|---|---|
| `stage0/run_exog_token.py` | 消融腳本（`exog2` / `exog8` / `control`） |
| `stage0/results/exog_token/preds_exog2.csv`、`preds_exog8.csv`、`preds_control.csv` | 逐點預測（3 種子） |
| `stage0/results/exog_token/metrics_*_ENS3.csv` | ENS3 指標 |
| `stage0/results/exog_token/metrics_*.csv` | 逐種子指標 |
