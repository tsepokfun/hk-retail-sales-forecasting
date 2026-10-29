# 2026 年 10 月 sales forecasting 研究景觀

> 資料日期：2026-10
> 取證限制：僅 `web_search`（URL 清單 + 零散片段），**無 PDF 全文抓取能力**。凡未出現在搜尋片段者標 `not found`，未以記憶補寫。
> 本文只彙整已查證結論，不含新推導。

---

## 0. 一句話現況

2026 年的前線不是「哪個大模型最強」，而是三件事同時發生：

1. 基礎模型從單變量走向「**多變量＋協變量零樣本**」（Chronos-2、TimesFM-3、TiRex-2）。
2. 榜單改成**按 domain 分開排名**——不存在總冠軍，零售／需求是獨立賽道。
3. 學界在吵**「通用時序基礎模型是不是範疇錯誤」**（ICML 2026 立場論文）。

---

## 1. 2025–2026 的模型清單

| 模型 | 機構 | 時間 | 核心主張 | arXiv / 連結 | 證據等級 |
|---|---|---|---|---|---|
| **Chronos-2** | Amazon | 2025-10 | 單變量 → 通用（多變量＋協變量零樣本） | [arXiv 2510.15821](https://browse-export.arxiv.org/abs/2510.15821)、[HF `amazon/chronos-2`](https://huggingface.co/amazon/chronos-2)、[廠商 blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) | 預印本 ＋ 廠商 blog；**參數量未證實** |
| **TimesFM 2.0 → 2.5 → 3.0** | Google Research | 2.0 見 [v2.0.1 tag](https://github.com/google-research/timesfm/tree/v2.0.1)；2.5 見[媒體](https://www.chinaz.com/ainews/21353.shtml)（更小、長上下文）；**3.0 為 2026-09** | **零樣本原生多變量** | [廠商 blog](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)、[媒體「超越所有對手但暫不能商用」](https://www.zhiding.cn/models/2026/0914/3199256.shtml)、[BigQuery 整合](https://cloud.google.com/blog/ja/products/data-analytics/bigquery-timesfm-3-0-with-multivariate-time-series-forecasting?hl=ja) | 廠商 blog ＋ 媒體；**參數量未證實** |
| **Moirai 2.0** | Salesforce | 2025-11 | **「少即是多」**：正則化策略比模型大小更重要 | [arXiv 2511.11698](https://ar5iv.labs.arxiv.org/html/2511.11698) | 預印本 |
| （Moirai-MoE，非 2.0） | Salesforce | 2024 | 混合專家版，**2026 無更新找到** | [arXiv 2410.10469](https://ar5iv.labs.arxiv.org/html/2410.10469) | 2024 工作 |
| **TiRex** | NX-AI（**xLSTM 骨幹**） | 2025，**NeurIPS 2025 收錄** | 長短視界零樣本預測 | [arXiv 2505.23719](https://ar5iv.labs.arxiv.org/html/2505.23719v1)、[論文筆記](https://github.com/xht252/paper-notes/blob/main/docs/NeurIPS2025/time_series/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md) | **同儕審查** |
| **TiRex-2** | NX-AI | **2026-07** | **推廣到多變量 ＋ 串流（streaming）** | [arXiv 2607.01204](https://arxiv-org.ezproxy.obspm.fr/pdf/2607.01204)、[repo](https://github.com/NX-AI/tirex-2) | 預印本；成績已進 fev-bench |
| **Toto** | Datadog | 2025，**NeurIPS 2025** | 可觀測性領域 TSFM；配套基準 BOOM | [NeurIPS 論文頁](https://papers.neurips.cc/paper_files/paper/2025/hash/48ea942b60e6c52b50ca90e4a40d8ac2-Abstract-Conference.html)（arXiv 2505.14766）、[廠商 blog](https://www.datadoghq.com/ja/blog/ai/toto-boom-unleashed/) | **同儕審查** |
| **Toto 2.0** | Datadog AI Research ＋ CMU | **2026-05** | **「時序預測進入 scaling 時代」** | [arXiv 2605.20119](https://browse-export.arxiv.org/abs/2605.20119) | 預印本 |
| **⭐ EXAONE Demand 1.0** | **LG AI Research** | **2026-08（媒體）／2026-09（論文）** | **明確以「需求預測」為目標的 TSFM**；媒體稱時序榜 SOTA、勝 Google 與阿里 | [arXiv 2609.30880](https://browse-export.arxiv.org/abs/2609.30880)、[HF paper](https://huggingface.co/papers/2609.30880)、[BusinessKorea](https://www.businesskorea.co.kr/news/articleView.html?idxno=274354)、[中文報導](https://www.digitaltoday.co.kr/cn/view/90484/lg-ai-research-industrial-focused-ai-strategy-gains-traction-stands-out-on-global-benchmarks) | 預印本 ＋ 媒體 |
| **Sundial** | 清華 THUML | 2025 | 128M base（由權重名稱即證）；**最新仍是 2025** | [HF `thuml/sundial-base-128m`](https://huggingface.co/thuml/sundial-base-128m)、[ICML 2025 slides](https://icml.cc/media/icml-2025/Slides/45591_NN9Y4k8.pdf) | ICML 2025 報告 |
| **Timer-S1** | 字節 | 2026 | 十億級 | [arXiv 2603.04791](https://browse-export.arxiv.org/abs/2603.04791) | 預印本 |
| **Falcon-2.0** | — | 2026 | 自稱 7 個 domain 中 4 個最佳 | [arXiv 2608.13262](https://arxiv-org.ezproxy.obspm.fr/html/2608.13262v1) | 預印本 |
| **Morpheus-2.0** | — | 2026 | — | [arXiv 2610.24321](https://export.arxiv.org/abs/2610.24321) | 預印本 |
| **CITRAS-FM** | — | 2026 | 已進 fev-bench | 見 §2 fev-bench commit | 榜單條目 |

**兩個必須記住的名字**

- **EXAONE Demand 1.0（LG，2026-09）**：**專門為需求預測設計**的 TSFM，不是通用模型順帶做零售 → 本題目最新、最直接的對手。
- **TiRex / TiRex-2（NX-AI）**：**xLSTM 骨幹打進 NeurIPS 2025 與 fev-bench 前線** → 「選 xLSTM」在 2026 有架構趨勢支持。

---

## 2. 排行榜：沒有總冠軍，只有分賽道

| 榜 | 現況 |
|---|---|
| **GIFT-Eval**（Salesforce，[arXiv 2410.10393](https://arxiv.org/html/2410.10393v2)） | **按 domain 分組給分（7 個 domain）**。可引片段：`"Falcon-2.0 achieves the lowest normalized MASE in four of seven domains: Energy (0.769), Healthcare (0…"` → 不存在單一總冠軍 |
| **fev-bench**（AutoGluon，[HF Space](https://huggingface.co/spaces/autogluon/fev-bench)） | **2026 仍活躍維護**：已加入 TiRex-2 成績（[commit fdf3780](https://huggingface.co/spaces/autogluon/fev-bench/commit/fdf37806d194662ff84499897556cc48fa5dd3b0)）、CITRAS-FM／FlowState（[commit a396ab4](https://huggingface.co/spaces/autogluon/fev-bench/commit/a396ab4dc8363454729c8f8e57fefb2d96abead0)，片段 `"TiRex-2,FlowState,0.731,0.577,0.885,0.101,0.03,0.169"`，欄名未取得） |
| **⭐ GIFT-Eval-CTX** | **2026 新增的「帶協變量／情境」子榜** — [arXiv 2610.17794](https://export.arxiv.org/pdf/2610.17794)。對零售最相關 |
| 第三方鏡像（非官方） | [tsfm.ai](https://tsfm.ai/benchmarks/gift-eval)、[Papers with Code 鏡像](https://paperswithcode.co/benchmark/gift-eval?task=time-series-forecasting) |

**誰在領先（2026）**：LG EXAONE Forecast（2026-08 報導）、Falcon-2.0（自稱 4/7 domain）、**科大訊飛认知智能國重實驗室以「情境認知時序預測智能體」登頂 GIFT-Eval（是 agent，不是純 FM）** — [link](https://cogskl.iflytek.com/archives/3503)。

→ 2026 榜單由「**工業專用模型 ＋ agent 包裝**」輪替，通用 FM 不再獨佔。

---

## 3. 零售上「基礎模型 vs GBDT／統計」：條件式有效

| 研究 | 做法 | 可得結論 |
|---|---|---|
| **《When Do Foundation Models Pay Off for Retail Demand Forecasting? A Systematic Review and Cross-Benchmark Meta-Analysis》** [Zenodo 21338004](https://zenodo.org/records/21338004) | 系統綜述＋跨基準統合。片段可見「Category A (FM papers): 17 papers, 58 rows」「Cost-accuracy Pareto frontier」「Evaluation protocol heterogeneity」 | **標題本身把問題改成「何時值得」**；預印本 |
| **《World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets》** [arXiv 2610.25170](https://export.arxiv.org/abs/2610.25170)（2026-10）、[code](https://github.com/karthik-iyer-au/retail-world-models) | 四份公開零售數據集實測 | 條件式；預印本 |
| **《Foundation or Formula? TimesFM-3 vs 古典模型》** [arXiv 2610.00589](https://browse-export.arxiv.org/abs/2610.00589)＋[預註冊協議](https://zenodo.org/records/22927193) | 預註冊對照協議 | 條件式；預印本 |
| **《Evaluation Choices Decide the Forecasting Leaderboard》** [arXiv 2609.27867](https://export.arxiv.org/pdf/2609.27867) | 生產級 marketplace 面板 | **榜單結論對評估選擇高度敏感**；預印本 |
| 零售／需求側旁證 | [快商務需求下 TSFM 何時有效](https://accesson.kr/jidb/v.17/3/89/59736)、[服裝零售預測基準測試](https://www.ebiotrade.com/newsf/2026-10/20261017085621419.htm)、[零售協同預測：不是先選模型](https://www.mdpi.com/2071-1050/18/17/8746/pdf) | 期刊 |

**2026 共識（標題層推論，非數字結論）**

- **有協變量、有階層、序列稠密** → FM 佔優。
- **稀疏／間歇性需求** → 仍是 GBDT／統計法。

**xLSTM 方向的旁證**：`"In long-term forecasting tasks, xLSTM-based models generally demonstrate superior performance compared to Transformer-based a…"` — [arXiv 2509.01187](https://backend.dev.arxiv.org/pdf/2509.01187)（預印本）。

⚠️ **上述四篇的具體勝負數字（MASE / nCRPS）皆未取得**：只取得標題與零散片段。

---

## 4. M5 → M6：M6 不是零售，2026 無零售接班競賽

| 項目 | 實況 |
|---|---|
| **M5（2020，Walmart）** | 42,840 階層序列（底部 30,490）。頂尖解法 = **LightGBM 決策樹集成 ＋ 按 store 池化訓練** — [arXiv 2102.13209](https://arxiv.org/pdf/2102.13209v3)、[Top 3% 解法 repo](https://github.com/btrotta/kaggle-m5) |
| **M6** | **不是零售，是投資／交易**。IJF 特刊 [S0169207025000652](https://www.sciencedirect.com/science/article/abs/pii/S0169207025000652)、[M6: Bridging the gap between forecasting and investment decisions](https://www.sciencedirect.com/science/article/pii/S0169207024001079)。可引片段：`"One of the key results… was the inability of participants to significantly outperform the forecasting benchmark"`。特刊導言另見 [Bath 研究入口](https://researchportal.bath.ac.uk/en/publications/introduction-to-the-m6-forecasting-competition-special-issue/) |
| **零售接班人** | **2026-10 前不存在**（已檢索 Kaggle 2025/2026、IJF 零售競賽特刊，只找到資料集） |
| **M7** | 2026-10 前**未見任何公告或論文** |

---

## 5. 2025–2026 會議與期刊論文

| 場地 | 論文 |
|---|---|
| **NeurIPS 2025** | TiRex（[2505.23719](https://ar5iv.labs.arxiv.org/html/2505.23719v1)）、Toto（[論文頁](https://papers.neurips.cc/paper_files/paper/2025/hash/48ea942b60e6c52b50ca90e4a40d8ac2-Abstract-Conference.html)）、[TiRex 論文筆記](https://raw.githubusercontent.com/zhaoyang97/Paper-Notes-en/refs/heads/main/docs/NeurIPS2025/time_series/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md) |
| **ICML 2026** | **Zeus: Towards Tuning-Free Foundation Model for Time Series Analysis** — [arXiv 列表註記 "Accepted by ICML 2026"](https://browse-export.arxiv.org/list/cs/2026-07?skip=24&show=2000) |
| **KDD 2026** | 因果視角決策條件模擬（Oral）[link](https://hub-assets-cache.baai.ac.cn/view/58061)；**《An Interpretable LSTM Encoder-Decoder with Top-Down Reconciliation for Hierarchical Retail Sales Forecasting》** [ACM 10.1145/3844275.3844292](https://dlnext.acm.org/doi/10.1145/3844275.3844292)（**會議歸屬未證實**） |
| **AAAI** | **LMGL-WD: LLM-Guided Multi-Task Graph Learning for Category-Level Warehouse Demand Prediction** [link](https://ojs.aaai.org/index.php/AAAI/article/view/38577) |
| **期刊** | 零售階層 GNN（*Knowledge-Based Systems* 2026）[S0950705126012918](https://www.sciencedirect.com/science/article/pii/S0950705126012918)；《Estimating Hierarchical Demand in Retail Operations》（*JOM*）[joom.70058](https://onlinelibrary.wiley.com/doi/pdf/10.1002/joom.70058)；**CEDAR 事件驅動需求（殘差分解）** [arXiv 2608.25871](https://browse-export.arxiv.org/abs/2608.25871) |

### 2026 的開放問題與爭論（同一批檢索所得）

| 爭論 | 文獻 |
|---|---|
| **通用時序基礎模型是「範疇錯誤」** | ICML 2026 立場論文 [arXiv 2602.05287](https://browse-export.arxiv.org/abs/2602.05287v1)（預印本） |
| **需要領域專屬基準** | ICML 2026 [TSFMs Require Explicit Domain-Level Benchmarks](https://icml.cc/virtual/2026/poster/67138) |
| **⚠️ 評估指標該換：主張以 Scaled Quantile Loss 取代 MASE 家族** | [fev-bench arXiv 2509.26468](https://arxiv-org.ezproxy.obspm.fr/html/2509.26468v4) |
| **榜單差異統計上不顯著（no champions）** | [arXiv 2502.14045](https://export.arxiv.org/abs/2502.14045) |
| **分層預測：end-to-end 還是後處理** | KU Leuven 2025-02 討論 [link](https://www.dataminingapps.com/2025/02/hierarchical-forecasting-end-to-end-or-post-hoc/) |
| **分層預測中「資訊的角色」** | [arXiv 2605.29611](https://arxiv-org.ezproxy.obspm.fr/html/2605.29611v1)（2026 預印本） |
| **分層 ＋ 共形預測（加權保證）** | [arXiv 2411.13479](https://arxiv-org.ezproxy.obspm.fr/html/2411.13479v4) |
| **冷啟動／新品** | IEEEE《Cold-Start Sales Forecasting in Fashion via Feature-Based Ensembles》[11206989](https://ieeexplore.ieee.org/document/11206989/references) |
| **促銷定價的因果（去偏 ML）** | KDD 2025 workshop [PDF](https://causal-machine-learning.github.io/kdd2025-workshop/papers/24.pdf) |
| **間歇性需求** | Amazon《Bayesian intermittent demand forecasting for large inventories》[link](https://www.amazon.science/publications/bayesian-intermittent-demand-forecasting-for-large-inventories) |
| **評估本身很難** | ICML 2025《Evaluating Forecasting is More Difficult than Other LLM Evaluations》[link](https://icml.cc/virtual/2025/49156) |

---

## 6. 誰在做（2025–2026 的實際落點）

| 團隊 | 在做什麼 |
|---|---|
| **KU Leuven** | 分層預測 end-to-end vs 後處理之爭（2025-02）[link](https://www.dataminingapps.com/2025/02/hierarchical-forecasting-end-to-end-or-post-hoc/)；M5 冠軍方法**降成本** [Lirias](https://lirias.kuleuven.be/retrieve/06b4b979-8a76-44b5-9e0c-9695d8f9e594) |
| **Monash / Rob Hyndman** | **2025–2026 未見零售分層新論文**；維護 **WAPE 指標**方法論 [Hyndsight](https://857.robjhyndman.com/hyndsight/wape.html)；重心在分層與機率預測 [作者頁](https://www.sciencedirect.com/author/7006914313/rob-j-hyndman) |
| **Makridakis / 尼科西亞** | **M6 競賽特刊導言** [Bath](https://researchportal.bath.ac.uk/en/publications/introduction-to-the-m6-forecasting-competition-special-issue/) |
| **清華 THUML** | Sundial 時序基礎模型家族（ICML 2025、權重開源） |
| **CMU** | Goswami 2025 博士論文《Towards Pragmatic Time Series Intelligence》— 從「更大模型」轉向務實評估 [PDF](https://www.ri.cmu.edu/app/uploads/2025/05/mgoswami_phd_robotics_2025.pdf) |
| **Amazon Science** | **Anchored FLoE**：業務約束下的「基礎模型＋本地模型」混合集成，用於需求預測 [link](https://www.amazon.science/publications/anchored-floe-a-business-guardrailed-ensemble-framework-of-foundation-and-local-trained-models-for-demand-forecasting) |
| **Zalando** | (R,s,Q) 補貨政策＋機率模型（*Scientific Reports* 2025）[論文](https://www.nature.com/articles/s41598-025-32537-2)、[工程 blog 2026-01](https://engineering.zalando.com/posts/2026/01/publication-replenishment-engine.html)、[庫存優化架構 2025-06](https://engineering.zalando.com/posts/2025/06/inventory-optimisation-system.html) — 從預測推到**決策** |
| **Datadog** | Toto ＋ BOOM 基準 [blog](https://www.datadoghq.com/ja/blog/ai/toto-boom-unleashed/) |
| **Uber** | 深度機率模型改造 Guidance Heatmap [blog](https://www.uber.com/pl/en/blog/enhancing-ubers-guidance-heatmap-with-deep-probabilistic-models/) |
| **Nixtla** | TimeGPT 只有文件／套件更新，**2025–2026 未找到新論文** [docs](https://www.nixtla.io/docs/introduction/about_timegpt) |
| **HKUST** | 2025-11-20 IEDA「Decision-Making in the Era of Data and AI」— **機構動向，非零售預測論文** [link](https://seng.hkust.edu.hk/news/20251120/hkust-ieda-decision-making-era-data-and-ai) |
| **香港其他院校** | **2025–2026 零售銷售預測同儕評審新作：未找到**；PolyU 僅有較舊的超市需求預測學位論文 [link](https://theses.lib.polyu.edu.hk/handle/200/1442) |
| **Walmart Global Tech / JD / Alibaba DAMO** | **2025–2026 研究出版品：未找到**（Walmart 僅商業新聞 [link](https://www.supplychain247.com/article/walmart-global-supply-chain-expansion/software)） |
| **Oxford-Man Institute** | **2025–2026 零售／nowcasting 新作：未找到**（僅較早 Eurona 工作論文 [link](https://ora.ox.ac.uk/objects/uuid:2184f32b-3a2f-4e07-a27c-80b7fbd152a8)） |
| **Princeton / Georgia Tech** | **具名零售預測團隊：未能確認** |

---

## 7. 2025–2026 可直接引用的綜述與特刊

| 項目 | 連結 | 備註 |
|---|---|---|
| **《When Do Foundation Models Pay Off for Retail Demand Forecasting?》** | [Zenodo 21338004](https://zenodo.org/records/21338004) | 最貼題；預印本 |
| **《Modeling Holiday Effect on Retail Demand Forecasting: A Systematic Review》** | [Preprints.org 2026-01](https://www.preprints.org/manuscript/202601.0840) | 節慶效應；預印本 |
| 《A Review of Machine Learning for Predicting Supply Chain Demand in Retail》 | [link](http://americaspg.com/journal/41/article/3487) | 一般背景 |
| 多模態綜述（增益不均） | [IEEE PDF](https://ieeexplore.ieee.org/ielx8/6287639/10820123/11197533.pdf) | 「有價值但增益不均」 |
| IJF 特刊「Advances in open-source forecasting software」 | [link](https://www.sciencedirect.com/special-issue/331062/advances-in-open-source-forecasting-software) | 與 R/Python 實作相關 |
| IJF 2026 徵稿「The Future of Forecasting: Bridging Theory and Practice」 | [link](https://forecasters.org/blog/2026/08/19/call-for-papers-international-journal-of-forecasting-the-future-of-forecasting-bridging-theory-and-practice/) | 2026-08-19 公告 |
| IJF 開放取用特刊截稿延期 | [link](https://forecasters.org/blog/2026/08/31/ijf-special-issue-call-for-papers-deadline-extended-open-source-forecasting/) | 2026-08-31 |
| IJF 特刊總覽 | [link](https://www.sciencedirect.com/journal/international-journal-of-forecasting/special-issues) | 持續更新 |
| M6 競賽特刊徵稿 | [PDF](https://forecasters.org/wp-content/uploads/Call-for-Proposals_The-M6-forecasting-competition-special-issue.pdf) | — |

---

## 8. 「找不到」清單（誠實記錄）

1. **各模型參數量與預訓練語料規模**：Chronos-2、TimesFM 2.5／3.0、Toto 2.0、TiRex／TiRex-2、EXAONE Demand 1.0——只取得標題層，**不予填補**。
2. **零售專屬排行榜與冠軍**：GIFT-Eval 是否把 Sales 獨立成一個 domain、該 domain 誰第一——只有「7 個 domain 分組計分」的片段，**domain 名單不完整**。
3. **TS-Arena 2026 狀態**：未檢索到 2026 年活躍證據。
4. **Chronos 官方 benchmark suite 2026 是否仍維護**。
5. **M5 之後的零售競賽**（M6-retail、Kaggle 2025/2026 零售預測賽、IJF 零售競賽特刊）。Kaggle 只搜到資料集（[demand-artifacts](https://www.kaggle.com/datasets/sonneygeorge/demand-artifacts12-16-25)）。
6. **阿里／螞蟻／微軟的專屬 TSFM 名稱**：只有「LG 勝過 Google 與阿里」的媒體敘述，**無可引用論文**。
7. **ICLR 2026 零售／需求專場論文**：未找到可證實條目。
8. **§3 四篇研究的具體勝負數字**（FM vs LightGBM 的 MASE／nCRPS 對比）——三篇最相關論文只取得標題與零散片段。

---

## 9. 對本專案的定位對照（由上述文獻推得，非新研究）

| 本專案選擇 | 2026 定位 | 可引文獻 |
|---|---|---|
| 用 **xLSTM** 而非 Transformer | **站得住** | TiRex [2505.23719]、TiRex-2 [2607.01204] |
| 用 **iTransformer** 做跨品類 | **要打折**（2025–2026 有通道依賴反證，見 `LITERATURE_MODEL_CRITIQUES.md`） | 2502.09683、Mendelowitz |
| 用 **MASE** 當主指標 | **有爭議**（fev-bench 主張改 SQL） | [2509.26468](https://arxiv-org.ezproxy.obspm.fr/html/2509.26468v4) |
| **MinT_wls 後處理調和** | **正中 2026 活爭論** | KU Leuven 2025-02 |
| **錨點 ＋ 殘差** | **有 2026 對照** | CEDAR [2608.25871](https://browse-export.arxiv.org/abs/2608.25871) |
| **按視界路由兩代模型** | **有工業先例** | Amazon Anchored FLoE |
| **不用基礎模型** | **有 ICML 2026 立場論文背書** | [2602.05287](https://browse-export.arxiv.org/abs/2602.05287v1) |
