# 文本作為預測輸入：文獻對照與本專案的文本失敗診斷

> 資料日期：2026-10
> 取證限制：僅 `web_search`（URL 清單 + 零散片段），**無 PDF 全文抓取能力**。文獻的具體改善百分比多數未取得，標 `not found`。
> 本文只彙整已查證結論，不含新推導。

---

## 1. 四條研究線對照

| # | 研究線 | 文本當什麼用 | 代表文獻 | 跟本專案的關係 |
|---|---|---|---|---|
| **1** | **經濟 nowcasting（報紙文本）** | 報紙 → 主題／情緒 → nowcast 宏觀量 | 見 §2 | **有關**（文本→特徵→預測宏觀量，同一類做法） |
| **2** | **零售需求 ＋ 文本** | 評論／促銷文字 → 嵌入／主題 → 當協變量 | 見 §3 | **最直接相關** |
| **3** | **LLM-for-TS** | 文本 = **prompt／任務描述** | 見 §4 | **無關**，見 §4 的判斷 |
| **4** | **央行／機構文本** | Beige Book 等經濟評論 → 語氣／衰退 nowcast | 見 §5 | **同類但語料不同** |

---

## 2. 研究線 1：經濟 nowcasting（報紙文本）

| 文獻 | 方法 | 可得細節 |
|---|---|---|
| **Thorsrud (2020)《Words are the New Numbers: A Newsy Coincident Index of the Business Cycle》** — *Journal of Business & Economic Statistics*，DOI [10.1080/07350015.2018.1506344](https://amstat.tandfonline.com/doi/full/10.1080/07350015.2018.1506344)（2018 線上、2020 卷期；Norges Bank WP 2/2016 為前身） | 對挪威財經日報 *Dagens Næringsliv* 做 **LDA 主題模型** → 取每日主題佔比 → 餵入**動態因子模型** → 建「新聞景氣指數」→ 即時 nowcast 挪威 GDP | **改善百分比 not found**（只回期刊頁與 WP 頁）。**存在性已證實** |
| 對該文的官方討論稿 | SNB 2017 short-term forecasting 會議（Strasser） | 可用來標註「學界接受但有爭議」[link](http://sdw.central.banktunnel.eu/pub/conferences/shared/pdf/20170929_advances_in_short_term_forecasting/Paper_5_Strasser.pdf) |
| **Kalamara, Turrell, Redl, Kapetanios (+Kapadia)《Making text count: economic forecasting using newspaper text》** — BoE Staff WP 865（2020）→ *Journal of Applied Econometrics*，DOI [10.1002/jae.2907](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.2907) | 報紙文本 → 主題／情緒特徵 → 機器學習與 BMA 預測 UK GDP | 只確認到標題與期刊，**逐表數字未取得** [BoE WP865](https://www.bankofengland.co.uk/working-paper/2020/making-text-count-economic-forecasting-using-newspaper-text) |
| **Banca d'Italia QEF 1321《The power of text-based indicators in forecasting the Italian economic activity》** | **詞典法（dictionary-based）** 建文本指標 | 可引片段：`"36 economic and sector-specific topics, using a dictionary-based approach on a very lar[ge] corpus"` → **詞典法可行，但語料遠大於 42k 標題、主題數 36** [機構頁](https://www.bancaditalia.it/pubblicazioni/temi-discussione/2021/2021-1321/index.html)、[全文 PDF](https://wrap-test.warwick.ac.uk/id/eprint/166146/1/WRAP-The-power-of-text-based-indicators-in-forecasting-Italian-economic-activity-Guaitoli-22.pdf) |

**零售／消費專門**：只找到一篇碩士論文《The Predictive Value of Media-Derived Textual Features for Consumer Behaviour: Evidence from UK Retail Sales》— [link](https://apd.uw.edu.pl/diplomas/252751/)，**證據等級低**。

---

## 3. 研究線 2：零售需求預測中的文本特徵

| 文獻 | 編碼方式 | 注入方式 | 備註 |
|---|---|---|---|
| **Attribute Embedding** — *Journal of Marketing Research*，DOI [10.1177/00222429211047822](https://journals.sagepub.com/doi/suppl/10.1177/00222429211047822) | 從**消費者評論**學出階層式產品屬性表徵 | **嵌入直接作協變量**（不是接在時序模型上） | 同類注入方式 |
| 促銷文字用 **BERTopic** 分類後餵需求預測（JSAI 2024） | 主題模型，**主題數由模型決定、非人工指定** | — | [J-STAGE](https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_3Q1IS2a03/_pdf/-char/ja) |
| **《Prompt-engineered contextual embeddings for panel time-series demand forecasting: a benchmark in multi-store inventory analytics》** — *Discover Analytics* 2026 | 上下文嵌入 | **面板 ＋ 嵌入 ＋ 零售庫存** | **最貼近本專案設定的 benchmark** [Springer](https://link.springer.com/article/10.1007/s44257-026-00094-1) |
| 多模態混合證據 | — | — | 「有價值但**增益不均**，逐資料集差異大，OEM 類資料集上文字增益不顯著」[IEEE 綜述 PDF](https://ieeexplore.ieee.org/ielx8/6287639/10820123/11197533.pdf) |

---

## 4. 研究線 3：LLM-for-TS —— **文本是 prompt，不是新聞語料**

| 模型 | 文本的角色 | 連結 |
|---|---|---|
| **Time-LLM**（ICLR 2024） | 文本 = **prompt**（任務描述 ＋ 領域提示）；數值經 patch ＋ reprogramming 對齊 LLM 詞嵌入空間；**凍結 LLM、不需大規模重訓** | [arXiv 2310.01728](http://export.arxiv.org/abs/2310.01728v1) |
| **One Fits All / GPT4TS**（NeurIPS 2023） | **完全不用文本**；凍結 GPT-2 主幹、只調少量參數 | [ACM](https://dl.acm.org/doi/10.5555/3666122.3667999)、[arXiv 2302.11939](https://arxiv.org/abs/2302.11939) |
| **LLMTime / Gruver et al.**（NeurIPS 2023） | 把數值串成字串 token，零樣本預測；**無外部文本** | [arXiv 2310.07820](https://ar5iv.labs.arxiv.org/html/2310.07820) |
| **TEMPO**（ICLR 2024） | **prompt-based 分解**（趨勢／季節／殘差）作為提示 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5132940b1bced8a7b28e9695d49d435a-Abstract-Conference.html) |
| **Chronos**（Amazon） | 純數值 tokenization 大規模預訓練，**文本不參與** | [arXiv 2403.07815](https://browse-export.arxiv.org/abs/2403.07815v2) |

### ⚠️ 關鍵判斷

> 這條線上「文本」幾乎一律是 **prompt／任務描述**，不是新聞語料。本專案的新聞標題屬**另一種 regime**（輔助協變量），**不能用這批論文替新聞協變量背書**。

綜述：[IJCAI 2025](http://export.arxiv.org/abs/2505.02583)、[KDD 2025](https://dl.acm.org/doi/10.1145/3711896.3736567)。

---

## 5. 研究線 4：央行／機構文本

| 文獻 | 內容 |
|---|---|
| **《Do Anecdotes Matter? Exploring the Beige Book through Textual Analysis from 1970 to 2025》**（2025 FEDS） | Beige Book 文本分析 [link](https://www.federalreserve.gov/econres/feds/do-anecdotes-matter-exploring-the-beige-book-through-textual-analysis-from-1970-to-2025.htm) |
| **《Decoding the Beige Book: LLM-Powered Sentiment Analysis for Real-Time Recession Forecasting》**（ICAIF 2025） | LLM 語氣分析做即時衰退預測 [link](https://dlnext.acm.org/doi/10.1145/3768292.3770425) |

⚠️ **Beige Book 是「地區經濟評論」，不是政府行政公告**——與本專案的 GovHK 語料性質不同。

---

## 6. 三個判決

### 判決 1：語料**太小**，維度也**太低**——兩頭都不對

| | 文獻標準 | 本專案 |
|---|---|---|
| 主題／維度數 | **36 個**經濟與行業主題（Banca d'Italia） | **4 個 PCA** ＋ 3 個詞典標量 |
| 語料規模 | 「**very large** corpus」（同一篇） | 42,594 條標題，**5.6 條／日** |
| 來源數 | 報紙**全文**（Thorsrud、Kalamara 皆報紙） | **單一政府新聞稿流**，只有標題 |

原始裁決用語：**「維度遠低於文獻，且語料遠小於文獻，兩頭都不對。」**

### 判決 2：量錯了**對象**

| 文獻用的語料 | 本專案用的語料 |
|---|---|
| Thorsrud：財經日報 | **政府新聞稿（GovHK）** |
| Kalamara：報紙 | 同上 |
| Attribute Embedding：消費者評論 | 同上 |
| Fed：Beige Book（地區經濟評論） | 同上 |

**沒有一篇用政府行政公告。** 本專案的 `news_sent_net` 詞表是 `fall / drop / strike / protest / arrest / accident / death`（`stage3/news_features.py` L62–67）——在 GovHK 新聞稿上，這些詞主要在講**颱風、交通意外、示威、拘捕**，不是零售需求。

### 判決 3：**平均**是「特徵選擇」的反面

| | 做法 |
|---|---|
| 本專案 | `E.mean(axis=0)`（`stage3/news_features.py` L111）—— 每月約 169 條取**平均** |
| 文獻建議 | **Kelly, Manela, Moreira《Text Selection》**，*JBES*，DOI [10.1080/07350015.2021.1947843](https://www.tandfonline.com/doi/epdf/10.1080/07350015.2021.1947843) —— 正面處理「文本特徵太多、觀測太少」：用**特徵選擇／收縮從海量文本特徵挑極少數** |

**平均把 1 條相關新聞與 168 條無關新聞混成同一個向量**——這是選擇的反面。

---

## 7. 本專案的文本實測（本地證據）

### 7.1 語料覆蓋

| 實測項目 | 數值 |
|---|---|
| 快取月份 | **243** 個月（`stage3/data/news_cache/*.json`） |
| **完全沒有新聞的月份** | **17 個**（2005-09 → 2007-07） |
| 2004 / 2005 / 2006 / 2007 有新聞的月份 | 3 / 8 / **4** / 7 |
| 2006 全年標題數 | **117 條**（對比 2022 年 3,218 條） |
| 全期總標題 | **42,594 條**／21 年 ≈ 2,028 條／年 ≈ 169 條／月 ≈ **5.6 條／日** |
| 訓練期缺失月 | **37–38 個月**，補值 = 訓練均值（z=0） |
| PCA 4 維解釋變異 | **43.6%**（0.2388 + 0.0847 + 0.0672 + 0.0450） |

（逐年標題數見 `stage3/data/news_text_meta.json` 的 `coverage_by_year`；逐年加總 = 42,594，與報告一致。）

### 7.2 注入結果：**比同寬度隨機噪音更差**

`FINAL_REPORT.md` 表 3-6（MASE_ALL，越低越好）：

| 特徵組 | MASE_ALL | 同寬度噪音 | 真實 vs 噪音 | 判定 |
|---|---:|---:|---:|---|
| `none`（0 欄，正式配置） | **0.98355** | — | — | 基準（最佳） |
| `text_scalar`（3 欄，詞典法） | 1.19524 | 1.05480（3 噪音） | **+13.31%** | 顯著有害 |
| `text_emb`（4 欄，MiniLM 句向量） | 1.08961 | 1.02270（4 噪音） | **+6.54%** | 比噪音差 |
| `visitor_mtr_text`（6 欄） | 1.21501 | 1.09867（6 噪音） | **+10.59%** | 比噪音差 |

### 7.3 診斷

| 問題 | 證據 |
|---|---|
| 量錯對象（行政公告 ≠ 需求訊號） | 詞表 `fall/drop/strike/protest/arrest/accident/death` 在 GovHK 上多為颱風／意外／示威 |
| 平均稀釋訊號 | 每月 169 條 → 1 個向量；單條相關新聞權重僅 1/169 |
| 覆蓋不全 ＋ 假常數 | 17 個空月 ＋ 37–38 個月 z=0 補值，與「平均新聞月」不可區分 |
| 維度壓縮不足以救 | PCA 4 維只剩 43.6% 變異，輸入端已把訊號平均掉 |

---

## 8. 「文本劣於同寬噪音」在文獻中的定位

| 項目 | 結果 |
|---|---|
| 公開 ablation 顯示 text < same-width noise | **not found**（代理搜 `curse of dimensionality text-augmented forecasting`、`text features overfit small samples` 均未找到） |
| 機制性支持 | **《When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks》** [arXiv 2502.02199](https://export.arxiv.org/abs/2502.02199) —— **高維嵌入在噪聲回歸任務上反而更差，壓縮維度可改善** |
| 直接問「文本到底有沒有用」的央行 WP | **Burri & Kaufmann《Do daily lead texts help nowcasting GDP growth?》**（SNB）— [Semantic Scholar](https://www.semanticscholar.org/paper/Do-daily-lead-texts-help-nowcasting-GDP-growth-Burri-Kaufmann/eb80993cf2f02e6a049d3c36c1a9ffbf1a0c3087)（mixed evidence，必須引） |

→ **本專案的此項結果屬自身實驗觀察，非文獻已知現象**；報告可主張為新觀察，但必須標明。

---

## 9. 時間對齊：文獻鐵律（本專案做對了）

| 規範 | 內容 | 連結 |
|---|---|---|
| **Real-time / vintage 數據集** | 所有預測必須用**當時可得版本**（Croushore & Stark, Philadelphia Fed） | [Croushore 綜述](https://www.cnb.cz/export/sites/cnb/en/economic-research/.galleries/seminars_workshops/2025/20250320_s4_croushore_paper.pdf) |
| **Ragged edge 對齊** | 混合頻率的標準工程規範 | [CRAN `bridgr` vignette](https://cran.r-project.org/web/packages/bridgr/vignettes/ragged-edge-nowcasting.html) |
| **時間戳 = 發布時間** | 不是事件發生時間；GovHK 應以 press release 上載時刻切窗 | — |
| **本專案的做法** | `stage3/verify_text_causality.py` 用**前綴遮罩重建**（把 `> cut` 的原始資料全抹成 NaN 重建，窗口內差值必須為 **0.0**） | `stage3/verify_text_causality.py` |

→ 本專案的對齊紀律**比大部分文獻更嚴**，應引 Croushore & Stark 標定其標準地位。

---

## 10. 「找不到」清單

1. **Thorsrud 與 Kalamara 的具體改善百分比**（只抓到期刊／WP 落地頁，內文表未抽出）。
2. **「語料量 vs 增益」的系統 ablation**：無。
3. **「預測任務該用幾個 topic」的權威建議**：無。
4. **「text 特徵劣於同寬隨機噪聲」的公開 ablation 表**：無。
5. **任何以 GovHK／香港政府新聞稿為語料的預測研究**：**未找到**。
6. **「香港零售類別面板 ＋ 新聞標題」的先行研究**：**未找到** → 屬本專案的 originality，也是風險。
7. **零售銷售（非 GDP、非 SKU 需求）層級的文字 nowcasting 同儕評審論文**：只有一篇碩士論文，無期刊級證據。
