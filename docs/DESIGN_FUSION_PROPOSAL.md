# 多模態融合設計方案

> 資料日期：2026-10
> 取證限制：融合架構文獻僅 `web_search` 片段；多數論文只取得標題與片段，**未取得結果表數值**。標 `not found` 者不得補寫。
> 本文的參數成本算術以本專案基線為分母：**xLSTM 20,496 參數 / 4,186 樣本（4.9:1）；iTransformer 26,024 參數 / 182 樣本（143:1）**。

---

## 0. 三個原則（由 143:1 的樣本比逼出）

| # | 原則 | 理由 |
|---|---|---|
| **1** | **不要加寬輸入向量，要加 token** | 每個 token 只需自己的小投影；加寬值流每欄要 32 參數 ＋ 一次性 1,088 |
| **2** | **不要餵 embedding，要餵指數** | 4 維 PCA 只解釋 43.6% 變異；低維結構化標量可解釋、可控 |
| **3** | **每個來源先過濾，再聚合** | 每月 169 條取平均 = 單條訊號權重 1/169 |

---

## 1. 五個方案（含參數代價）

| 方案 | 做法 | 參數成本 | 佔模型 | 風險 | 判斷 |
|---|---|---:|---:|---|---|
| **A. 便宜 token 融合** | 每個資料源 = 1 個 token，各用 `Linear(k→32)`（k=2–4 標量）直接進 self-attention | **≈128／源** | <1% | 低 | ✅ **首選** |
| **B. 過濾後主題指數** | 領域詞表先篩到每月 5–15 條 → 主題／情緒 → **3–5 個標量**；只在 h=6/h=12 注入 | 極低 | <1% | 低 | ✅ **與 A 並用** |
| **C. LLM 生成標量** | 每月呼叫一次 LLM，輸出 1–3 個 0–1 標量（促銷強度、旅客相關度、政策事件） | 252 次呼叫（21 年 × 12） | — | 低 | ✅ B 的替代 |
| **D. pooled / cross-learning** | 跨序列共享統計強度（iTransformer 本質如此） | 已有 | — | 低 | ✅ 選 iTransformer 當融合平台的理由 |
| **E. cross-attention 逐模態** | TimeXer 式：外生 token 只走 cross-attention | **≈12k／模態／層** | **47%** | **最高** | ❌ **143:1 下不建議** |

### 1.1 融合架構的代價對照（代理實算）

| 融合方式 | 參數成本（本專案設定下） | 佔模型比例 | 備註 |
|---|---:|---:|---|
| **early fusion**（走 iTransformer token 通道） | **≈0**（共享投影）→ 自訂投影約 128／源 | <1% | 只加寬注意力序列 |
| **late fusion**（384 維文本 → 8 維、23 條序列） | ≈ (8+1)×23 ≈ **207** | **0.8%** | 完全不過 attention |
| **cross-attention** | ≈ **12k／模態／層** | **47%** | 最高風險 |

---

## 2. 每個方案的文獻依據

### 方案 A：加異質 token

| 依據 | 內容 | 連結 |
|---|---|---|
| **TimeXer**（NeurIPS 2024） | 外生變量當**獨立 token 只走 cross-attention**，內生變量走 self-attention | [arXiv 2402.19072](https://arxiv.org/abs/2402.19072) |
| **iTransformer 原論文** | 變量 token 由**跨變量共享**的 `Linear(lookback→d_model)` 產生 → **增加變量數不改參數數**，只加寬注意力序列 | [arXiv 2310.06625](https://arxiv.org/abs/2310.06625) |
| **register token（ViT，ICLR 2024）** | 高範數離群 token 佔用容量；加少量 register token 可消除 | [arXiv 2309.16588](https://arxiv.org/abs/2309.16588) |
| **ROSE（時序 register）** | 引入 register／原型庫吸收冗餘與領域偏移 | [arXiv 2405.17478](https://arxiv.org/abs/2405.17478)（預印本） |
| **Time-LLM（ICLR 2024）** | text prototype ＋ prompt-as-prefix 把自然語言注入序列模型 | [arXiv 2310.01728](https://arxiv.org/abs/2310.01728) |
| **TOTEM（NeurIPS 2024）** | token 化時序嵌入的對照 | [arXiv 2402.16412](https://arxiv.org/abs/2402.16412) |

> **方案 A 的定位更正**：加 token 的想法**不是原創**（TimeXer 已做）。差別在於**拒絕 TimeXer 的 cross-attention 實作**（12k／模態），改用最便宜的早期融合（128／源），成本約為 TimeXer 的 **1/90**，代價是失去「外生只讀不寫」的隔離。

### 方案 B：結構化抽取

| 依據 | 內容 | 連結 |
|---|---|---|
| **Banca d'Italia QEF 1321** | 詞典法 ＋ **36 個經濟／行業主題**，語料「very large」 | [全文 PDF](https://wrap-test.warwick.ac.uk/id/eprint/166146/1/WRAP-The-power-of-text-based-indicators-in-forecasting-Italian-economic-activity-Guaitoli-22.pdf) |
| **Text Selection（Kelly, Manela, Moreira, JBES）** | 從海量文本特徵**選擇／收縮**出極少數 → 「要選不要平均」 | [DOI 10.1080/07350015.2021.1947843](https://www.tandfonline.com/doi/epdf/10.1080/07350015.2021.1947843) |
| **BERTopic 促銷分類（JSAI 2024）** | 主題數由模型決定，非人工指定 | [J-STAGE](https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_3Q1IS2a03/_pdf/-char/ja) |
| **LDA 主題 ＋ 宏觀指標** | 在 h=3,4 勝過純宏觀模型 | [里斯本大學論文庫](https://repositorio.ulisboa.pt/server/api/core/bitstreams/0d8121d9-1dd5-4878-aad7-f177d91d71f7/content) |
| 熵式文本特徵工程 | 字典／計數式抽取預測流動性變化 | [EPJ Data Science](https://link.springer.com/article/10.1140/epjds/s13688-025-00535-z) |
| Word2Prices | 央行文本嵌入做通脹預測（embedding 路線標竿） | [Econometric Society](https://jstor.econometricsociety.org/regional-activities/conference-papers/view/282/966) |

### 方案 C：LLM 產標量

| 依據 | 內容 | 連結 |
|---|---|---|
| **Fed Beige Book LLM 語氣分析** | LLM 生成語氣特徵做即時衰退預測（同類先例） | [ICAIF 2025](https://dlnext.acm.org/doi/10.1145/3768292.3770425) |
| Beige Book 文本分析（1970–2025） | 長期文本分析示範 | [FEDS 2025](https://www.federalreserve.gov/econres/feds/do-anecdotes-matter-exploring-the-beige-book-through-textual-analysis-from-1970-to-2025.htm) |

### 方案 D：pooled / cross-learning

| 依據 | 內容 | 連結 |
|---|---|---|
| global 模型系統性優於逐序列模型（同儕審查） | M4 等競賽實證 | [arXiv 2008.00444](https://browse.arxiv.org/abs/2008.00444)、[IJF](https://www.sciencedirect.com/science/article/abs/pii/S0169207021000558) |
| **M5 冠軍是跨序列 cross-learning 的 LightGBM** | 不是逐序列模型 | [M5 accuracy PDF](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf) |
| 階層序列上的遷移學習 | 以 M5 冠軍法為教師降成本並保留精度 | [KU Leuven](https://lirias.kuleuven.be/retrieve/06b4b979-8a76-44b5-9e0c-9695d8f9e594) |
| M5 增益來源 | 主要來自 **cross-learning**，而非架構本身 | [UTwente 碩論](http://essay.utwente.nl/95006/1/Fitria_MA_EEMCS.pdf)（非同儕審查） |

### 方案 E：cross-attention（不建議）

| 依據 | 內容 |
|---|---|
| TimeXer | 已證明可行，但每模態需要一整組 QKV＋FFN |
| 參數代價 | d_model=32 時 ≈ **12d² ≈ 12k 參數／層** → 本專案的 **47%** |
| 唯一直接橫向比較 early／intermediate／late | [arXiv 2406.15098](https://arxiv.org/abs/2406.15098)／[IEEE 10650421](https://ieeexplore.ieee.org/document/10650421)（預印本，SMC 2024）：**cross-modal interaction 本身主導成效，而非單純資訊量** |
| 融合範式定義（early/intermediate/late） | [COMFORTage D3.7](https://zenodo-rdm.web.cern.ch/records/18887131/files/COMFORTage_D3.7_Integrated-AI-based-Care-Model-Library-I_v1.0.pdf?download=1)（產業 AI 交付文件，**非同儕審查**） |

---

## 3. 文本編碼器：三條硬規矩

| 規矩 | 依據 | 連結 |
|---|---|---|
| **1. 凍結編碼器** | MiniLM（蒸餾、384 維）適合作凍結編碼器 | [arXiv 2002.10957](https://arxiv.org/abs/2002.10957)（預印本） |
| **2. 數百樣本下不要微調**，改用 SVD 投影把嵌入空間對齊到目標任務 | 針對 few-sample 的直接主張 | [arXiv 2004.05119](https://ar5iv.labs.arxiv.org/html/2004.05119)（**唯一直接證據，非共識**） |
| **3. 維度 ≤5–8** | **文獻查不到 PCA 降維的權威建議**；改由參數預算反推：只有 182 樣本，文本側自由參數應 ≪ 182 | 代理自家推導，非引文 |

**補充機制證據**：《When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks》[arXiv 2502.02199](https://export.arxiv.org/abs/2502.02199) — **高維嵌入在噪聲回歸上反而更差，壓縮可改善**。

**小樣本容量控制的其他依據**

| 做法 | 依據 |
|---|---|
| 凍結預訓練主幹、只訓少量參數 | One Fits All（NeurIPS 2023 Spotlight）[arXiv 2302.11939](https://arxiv.org/abs/2302.11939) |
| PEFT／低秩適配於時序基礎模型 ＋ 負載預測 | [IEEE 11535767](https://ieeexplore.ieee.org/document/11535767) |
| partial pooling／共享參數 | 見方案 D |
| weight decay／dropout／purged early stopping 在 <10k 觀測下的專門消融 | **not found** |

---

## 4. 語料升級清單

| 語料 | 目標相關度 | 可得性 | 文獻對應 |
|---|---|---|---|
| **HKEXnews 上市公司月銷／同店銷售公告** | ★★★★★ 直接講零售 | 免費、有檔案 | Attribute Embedding 的「產品屬性文本」邏輯 |
| **WiseNews（慧科）** | ★★★★☆ | **需機構授權** | **Thorsrud、Kalamara 都用報紙 → 報紙是文獻標準語料** |
| **GDELT** | ★★★☆☆ | 免費 API（日度、2013+、含語氣） | 可做大語料 ＋ 事件抽取 |
| RTHK / Now / 有線財經 RSS | ★★★☆☆ | 免費 | — |
| 政府新聞稿（現用 GovHK） | ★☆☆☆☆ | 已有 | Fed Beige Book 屬同類，但 Beige Book 是**經濟評論** |
| LIHKG / OpenRice | ★★☆☆☆ | 技術與倫理問題 | 建議列 future work |

**原則：關鍵不是「更多」，是「更貼近目標」。** 從 169 條／月的政府公告，換成 5–15 條／月的**零售相關**公告 ＋ 公司月銷披露，資訊密度高一個數量級。

---

## 5. 執行順序（不可跳步）

| 步驟 | 內容 | 依據 |
|---|---|---|
| **Step 0** | **先修驗證協定** | `FINAL_REPORT.md` §5.1 第 4 點：現行 `val_months=24` ＋ `purge_val` 丟掉 2023 全年 12 個原點、訓練只到 2021-11 → covariate shift；`rate_z` 在訓練期 155 個月中 **141 個月死鎖在 −0.470998σ**。驗證區塊只剩 13 個原點，太小 |
| **Step 1** | 加 token（方案 A） | 見 §2 |
| **Step 2** | 對**同寬度噪音對照** | 本專案已有此機制（`stage3/run_stage3.py` 的 `noise_block`），**必須保留** |
| **Step 3** | 只在驗證顯著的視界注入 | 外生唯一穩定有用的視界是 **h=6**（`FINAL_REPORT.md` §5.1） |
| **Step 4** | 天氣改用**極端事件指示變數**（如 `rain > P95`），不用連續距平 | `rain_anom_z` 在 2023-09 達 **+4.976σ**（世紀黑雨）、`temp_anom_z` 2024-04 達 **+3.250σ** → 不受控外推 |

---

## 6. 風險與「沒有先例」的部分

| 項目 | 狀態 |
|---|---|
| 把**六個來源同時**併入單一 variate-token transformer 的已發表先例 | **not found**（最接近者為 TimeXer 與 ROSE，均未達此源數）→ **可寫進報告作為 originality，同時是風險** |
| 嵌入法 vs 結構化抽取法的**受控對照實驗**（有量化勝負） | **not found** → 方案 B、C **沒有直接文獻可引**，只有間接支持 |
| 「語料量 vs 增益」的系統 ablation | **not found** |
| 「預測任務該用幾個 topic」的權威建議 | **not found** |
| 小樣本多模態融合的結果表數值 | **未取得**（工具只回標題與片段） |

---

## 7. 建議寫進報告的一句話

> 融合設計以「**加 token 不加寬度、餵指數不餵嵌入、先過濾再聚合**」三原則為約束，並以同寬度隨機噪音作為判準閘門；任何未贏過同寬噪音的注入一律不進正式配置。
