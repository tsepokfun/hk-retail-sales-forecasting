# 更正紀錄（Corrections Log）

> 資料日期：2026-10
> 用途：記錄本輪研究過程中**講錯、講過頭、或證據等級不足**的條目，逐條列出「原話 → 核實結果 → 現行正確說法」。
> 原則：凡已撤回的數字不得在其他文件中復活。

---

## A. 數字錯誤（已撤回或修正）

### A-1 M5 觀測點：83,000,000 → **≈59,181,090**

| 項目 | 內容 |
|---|---|
| **原話** | 對照表中列 M5「觀測點 ~83,000,000」 |
| **錯在哪** | 把 **42,840（含聚合層的階層序列數）** 當成底部序列數去乘天數 |
| **核實結果** | 42,840 為**階層序列總數**；**底部 item-store 級為 30,490**。30,490 × 1,941 ≈ **59,181,090**（且 1,941 天本身 `not verified`） |
| **現行正確說法** | 「M5 = 42,840 series（bottom 30,490）/ **≈59,181,090 obs**（days count not verified）」 |
| 來源 | [TU/e 論文 PDF](https://pure.tue.nl/ws/portalfiles/portal/331251027/Master_Thesis_Jasper_Oostvogel.pdf)、[arXiv 2606.27863](https://arxiv.org/html/2606.27863v1) |

### A-2 規模差：13,700× → **≈9,784×**（按觀測點）／**1,862.6×**（按序列）

| 項目 | 內容 |
|---|---|
| **原話** | 「M5 的觀測點約為本專案的 **13,700 倍**；序列數 1,863 倍」 |
| **錯在哪** | 分子用了錯誤的 83M |
| **核實結果** | 59,181,090 ÷ 6,049 ≈ **9,784**；42,840 ÷ 23 = **1,862.6**；底部 30,490 ÷ 23 = **1,325.7** |
| **現行正確說法** | 「**1,862× by series，≈9,784× by observations**」 |
| 注意 | 同制度最貼近的基準是 **M1-monthly（617 序列 / 44,892 觀測 / h=18）= 27× by series** |

### A-3 Rossmann 歷史長度：1,014 天 → **not verified**（片段顯示 942 天，同樣未核實）

| 項目 | 內容 |
|---|---|
| **原話** | 「Rossmann 1,115 店 × **1,014 天**（~1.02M rows）」 |
| **錯在哪** | 憑既有知識填數字，未經查證 |
| **核實結果** | **1,115 店有片段佐證**（`"historical sales data from 1,115 Rossmann stores"`）；**1,014 天無佐證**；檢索片段顯示的是 **942 天（2013-01-01～2015-07-31）且同樣 not verified**；**列數亦 not verified** |
| **現行正確說法** | 「Rossmann：1,115 店已證實；歷史長度（942 天）與列數 `not verified`」 |
| 來源 | [GitHub 專案](https://github.com/mansi-guptaa/Rossmann-Store-Sales-Prediction)、[Kaggle data](https://www.kaggle.com/c/rossmann-store-sales/data) |

### A-4 Favorita 1.25 億行 → **不可引用**

| 項目 | 內容 |
|---|---|
| **原話** | 「Favorita ~125,000,000 行」 |
| **錯在哪** | 該數字只出現在**第三方問答頁** |
| **核實結果** | 125,497,040 見於[第三方問答](https://cloud.tencent.com.cn/developer/ask/sof/116087751)，**無法回溯源頭**；店數、item 數、日期範圍亦 `not verified` |
| **現行正確說法** | 「Favorita：資料規模 `not verified`；125,497,040 **不可引用**」 |

### A-5 廠商處理規模 → **全部無可核實來源**

| 項目 | 內容 |
|---|---|
| **原話** | （暗示 SAP IBP / Oracle RDF / Blue Yonder / RELEX / o9 有可比的 SKU-store 處理規模） |
| **核實結果** | 執行 6 組查詢（`RELEX "millions of SKUs"`、`Blue Yonder "billions" SKU-store`、`SAP IBP "millions of SKUs"`、`o9 "billions of" SKU-store`、`RELEX "billions of" forecasts`、`Blue Yonder "trillions"`），回傳全為行銷手冊／案例頁，**片段內皆無數字** |
| **現行正確說法** | 「廠商宣稱覆蓋大型零售商數百萬 SKU／數千店，**具體每客戶規模無公開可核實數字**」，並標為 **vendor claim** |

---

## B. 引文紀律錯誤

### B-1 「abandons memory mixing」加引號 → **必須改 paraphrase**

| 項目 | 內容 |
|---|---|
| **原話** | 以引號呈現 `"To enable parallelization, the mLSTM abandons memory mixing, i.e., the hidden-hidden recurrent connections."` |
| **錯在哪** | 該句來自**本專案筆記**（`00_PROJECT_INFO.md`），不是論文逐字核實 |
| **核實結果** | **方向正確，但無法逐字核實**；可確認的是並行性來自「去掉 hidden-hidden 遞迴／memory mixing」這條結構性理由；論文中**沒有** `parallel prefix sum` 字樣 |
| **現行正確說法** | mLSTM 的並行性源於**放棄 memory mixing（hidden-hidden 遞迴）** —— **paraphrase，不加引號** |

### B-2 xLSTM 原論文「0 命中時序」→ 證據升級，但數字仍缺

| 項目 | 內容 |
|---|---|
| **原話** | 「xLSTM 原論文完全沒做過時序預測（全文 0 命中）」 |
| **核實結果** | **任務清單外部確認**：語言模型（15B tokens SlimPajama）＋合成任務（Nearest Neighbor Search、Cycle Nav）→ **零時序為真** |
| **仍缺** | **perplexity 具體值 not found** |
| **現行正確說法** | 「評估範圍為語言模型與合成任務，**無時序預測**」；不得引述任何 perplexity 數字 |

### B-3 `sLSTMLayerConfig.backend` 預設 `'cuda'` → 屬**本地證據**

| 項目 | 內容 |
|---|---|
| **原話** | 「上游預設 `'cuda'` 在本機會 DLL 載入失敗」 |
| **核實結果** | 上游**已記錄**的是 [Issue #39](https://github.com/NX-AI/xlstm/issues/39)（`nvcc fatal : Unknown option '-extra-device-vectorization'`）與 [Issue #66](https://github.com/NX-AI/xlstm/issues/66)（torch>=2.6 不相容）。**Windows 專屬 issue：not found；Triton 明文依賴：not found** |
| **現行正確說法** | 該三項坑屬**本機實測證據**，**不可寫成「上游已知」** |

---

## C. 方法論與原創性錯誤

### C-1 「全部答案都來自本專案的代碼與資料卡」

| 項目 | 內容 |
|---|---|
| **原話** | 以本地 `itransformer.py` docstring 與代碼回答「iTransformer / xLSTM 是什麼結構」 |
| **錯在哪** | ① 沒有論文原話（只有專案自己的註解）② **完全沒有查 2025–2026 的後續批判** ③ 沒有回答「這個制度（月度／23 序列）有無外部證據」 |
| **核實結果** | 後續批判共 **5 條**（2502.09683、Mendelowitz、2502.14045、ACL 2026、2607.19404）；且**本制度無任何已發表評測** |
| **現行正確說法** | 見 `LITERATURE_MODEL_CRITIQUES.md` 全篇 |

### C-2 「方案 A（加 token）是我的設計」→ **TimeXer 已有先例**

| 項目 | 內容 |
|---|---|
| **原話** | 把「每個資料源變成多一個 token」當作本次提案的設計 |
| **核實結果** | **TimeXer（NeurIPS 2024）已把外生變量當獨立 token 走 cross-attention** — [arXiv 2402.19072](https://arxiv.org/abs/2402.19072)；register token 亦有先例（[ViT ICLR 2024](https://arxiv.org/abs/2309.16588)、[ROSE](https://arxiv.org/abs/2405.17478)） |
| **現行正確說法** | 借 TimeXer 的**想法**、**拒絕其 cross-attention 實作**（12k／模態 = 模型的 47%），改用最便宜的早期融合（≈128／源）。**差異化在成本，不在概念** |

### C-3 「本專案控制組免疫於 instance norm 批判」→ **必須標為推論**

| 項目 | 內容 |
|---|---|
| **原話** | 「instance normalisation 兩臂共用 → −0.05407 不可能由 norm 造成」 |
| **性質** | 這是**從代碼 docstring 推得的推論**（S4 對 iTransformer 保留 ＋ 雙胞胎描述為 identical），不是論文或實驗直接陳述 |
| **必須同時寫的兩點** | ① 兩臂**都拿不到** 27 個工程特徵（S1）→ 該比較是**下限**不是上限 ② Mendelowitz 的 CD-CI **near-parity** 與本專案結論**方向相反**，必須並列 |
| **現行正確說法** | 見 `LITERATURE_MODEL_CRITIQUES.md` §G，含「⚠️ 與批判 #2 方向相反，必須並列」的標註 |

### C-4 「贏過業界做法」→ 只對 LightGBM 成立

| 項目 | 內容 |
|---|---|
| **原話** | （早期表述傾向「本專案贏過業界方法」） |
| **核實結果** | `requirements.txt` 無 `statsmodels`／`prophet`；`stage0/**/*.py` 搜 `ARIMA|Holt|ExponentialSmoothing` 只命中變數名 `targets` → **ETS／SARIMAX／Prophet 從未測過** |
| **現行正確說法** | 「贏過 LightGBM」成立；「贏過業界做法」**對 ETS／SARIMAX／Prophet 是沒測，不是贏了** |

---

## D. 證據等級需降級（不是錯，但不可當硬證）

| # | 主張 | 問題 | 應標註 |
|---|---|---|---|
| D-1 | M5 歷史「1,941 天／2011-01-29～2016-06-19」 | 當時自信陳述，未核實 | **not verified** |
| D-2 | TimesFM-3「超越所有對手」 | **媒體報導**，非論文數據 | 標「媒體稱」 |
| D-3 | EXAONE Forecast「時序榜 SOTA、勝 Google 與阿里」 | **媒體報導**（BusinessKorea / DigitalToday） | 標「媒體稱」 |
| D-4 | iTransformer「官方 repo 內建 instance norm 且預設啟用」 | 屬 **repo 層事實**（[thuml/iTransformer](https://github.com/thuml/iTransformer)），非論文正文 | 標「repo 層」 |
| D-5 | fev-bench「TiRex-2,FlowState,0.731,…」 | 片段**欄名未取得** | 不可解讀為特定指標 |
| D-6 | KDD 2026《Interpretable LSTM Encoder-Decoder with Top-Down Reconciliation》 | **會議歸屬未證實** | 標「會議未證實」 |
| D-7 | GIFT-Eval 的 7 個 domain 名單 | 片段只有部分（Energy、Healthcare…） | **名單不完整** |
| D-8 | M5 協變量（price/promo/calendar/SNAP） | 無片段含清單 | **not verified** |

---

## E. 已核實、不需修正的項目（避免過度自我否定）

| # | 主張 | 核實狀態 |
|---|---|---|
| E-1 | 本專案 MASE 0.90198（iTransformer）／0.94455（xLSTM）／1.03029（naive）／1.04108（LightGBM tuned） | 本地實測，`FINAL_REPORT.md` 表 ES-1 |
| E-2 | 種子雜訊門檻 0.02 MASE | `FINAL_REPORT.md` §4.1 |
| E-3 | 通道混合可歸因 −0.05407，三層 DM 全顯著 | `FINAL_REPORT.md` §2.3 |
| E-4 | MinT_wls 約束違反量 161.63 HK$m → 1e-13 | `FINAL_REPORT.md` §3.5 |
| E-5 | 新聞語料：243 快取月、17 空月（200509–200707）、42,594 條、5.6 條/日、PCA 4 維 43.6% | 本地 `stage3/data/` 實測；逐年加總 = 42,594 與報告一致 |
| E-6 | 文本注入比同寬噪音差：text_scalar +13.31%、text_emb +6.54%、visitor_mtr_text +10.59% | `FINAL_REPORT.md` 表 3-6 |
| E-7 | xLSTM 原論文無時序預測 | 外部任務清單確認（見 B-2） |

---

## F. 由本紀錄產生的三條紀律

1. **凡數字必須有片段出處**；無出處者寫 `not verified`，不套用常見引用值。
2. **凡引文必須逐字可見**；只見 paraphrase 者不加引號。
3. **凡「先例」必須先查**；提出設計前先確認是否已有 TimeXer／register token 這類工作。
