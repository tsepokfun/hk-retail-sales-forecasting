# 資料集規模對照（含已修正的錯誤）

> 資料日期：2026-10
> 取證限制：僅 `web_search`（URL 清單 + 零散片段），**無 PDF 全文抓取能力**。凡片段未出現的數字標 `not verified`，**不套用常見引用值**。
> 本文只彙整已查證結論；修正條目另見 `CORRECTIONS_LOG.md`。

---

## 1. 本專案（Hong Kong C&SD 620-67002）

| 項目 | 數值 | 出處 |
|---|---|---|
| 序列數 | **23** 個零售行業類別 | `02_DATASET.md` §4 |
| 頻率 | 月度 | — |
| 歷史長度 | **263 個月**（2004-10 → 2026-08） | `02_DATASET.md` §3.1 |
| **原始觀測點** | **6,049**（23 × 263，零缺失） | `02_DATASET.md` §8A |
| 可用樣本 | **4,761**（207 原點 × 23） | `FINAL_REPORT.md` §1.2 |
| 訓練／驗證 | **4,186 / 575**（purged 協定下為 4,186 / 299） | 同上 |
| 階層 | 4 個母行業 ＋ 15 個子行業 ＋ 4 個獨立類別（共享會計恆等式） | `02_DATASET.md` §4 |
| 目標單位 | HK$ million（名目銷售值） | — |
| 特徵層 | `features_long.csv` = 6,049 行 × **36 欄**（2,008,165 B） | `02_DATASET.md` §8A |
| xLSTM 參數／樣本 | 20,496 / 4,186 = **4.9 : 1** | `FINAL_REPORT.md` §1.2 |
| **iTransformer 參數／樣本** | 26,024 / **182 個面板窗口** = **143 : 1** | `FINAL_REPORT.md` §1.2 |

**關鍵事實**：iTransformer 的「一個樣本」= 一整個 23 × 12 的面板窗口，共 **182 個**。每個樣本攜帶 23 × 12 = **276 個數字**。

---

## 2. 逐資料集核實表

| 資料集 | 序列數 | 頻率 | 歷史長度 | 觀測點 | 協變量 | 來源 |
|---|---|---|---|---|---|---|
| **本專案** | **23** | 月 | 263 月 | **6,049** | 0（13 個測過全敗） | 本地 |
| **M5** | **42,840** 階層序列（**底部 30,490** item-store） | 日 | **1,941 天 not verified**（2011-01-29～2016-06-19 亦 not verified） | **≈59,181,090**（30,490 × 1,941，條件性；天數未核） | price/promo/calendar/SNAP **not verified** | [M5 IJF 2022](https://www.sciencedirect.com/science/article/pii/S0169207021001874)、[TU/e 論文 PDF](https://pure.tue.nl/ws/portalfiles/portal/331251027/Master_Thesis_Jasper_Oostvogel.pdf)、[GitHub](https://github.com/bagherig/walmart-forecasting)、[arXiv 2606.27863](https://arxiv.org/html/2606.27863v1) |
| **M4** | **100,000** | 6 種 | 各頻率不同 | **not verified** | 無 | [IJF/S0169207019301128](https://www.sciencedirect.com/science/article/pii/S0169207019301128)；Monthly **48,000、h=18** [ar5iv 2603.08753](https://ar5iv.labs.arxiv.org/html/2603.08753)；Daily 4,227、h=14 [arXiv 2411.04554](https://arxiv.org/html/2411.04554v1)；Yearly 23,000、Hourly 414 [arXiv 2508.07697](https://arxiv.org/html/2508.07697v4)。Quarterly 24,000／Weekly 359 **not verified**（僅見 [Zenodo Quarterly](https://zenodo.org/records/4656410)、[Weekly](https://zenodo.org/records/3898376) 頁存在） |
| **M3** | **3,003** | 年/季/月/其他 | 各頻率不同 | not verified | 無 | [Mcomp 套件 PDF](http://cran.us.r-project.org/web/packages/Mcomp/Mcomp.pdf)（片段：`"the 3003 time series from the IJF-…"`） |
| **M1（monthly 子集）** | **617**（全集 1,001） | 月 | — | **44,892** | 無 | [AutoGluon-TimeSeries arXiv 2308.05566](https://ar5iv.labs.arxiv.org/html/2308.05566v1)（表格列 `M1 Monthly \| 617 \| 44,892 \| 18`） |
| **Rossmann Store Sales**（Kaggle 2015） | **1,115 店**（店×日，非 SKU） | 日 | **942 天 not verified**（2013-01-01～2015-07-31 亦 not verified） | not verified | promo/state/school holiday **not verified** | [GitHub 專案](https://github.com/mansi-guptaa/Rossmann-Store-Sales-Prediction)（片段：`"historical sales data from 1,115 Rossmann stores"`）、[Kaggle data](https://www.kaggle.com/c/rossmann-store-sales/data) |
| **Corporación Favorita**（Kaggle 2018） | 店數、item 數 **not verified** | 日 | 2013-01-01～2017-08-15 **not verified** | **not verified** | not verified | [Kaggle 競賽頁](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)。**125,497,040 僅見[第三方問答](https://cloud.tencent.com.cn/developer/ask/sof/116087751)，不可引用** |
| **Kaggle Store Item Demand** | **500**（50 品項 × 10 店） | 日 | **5 年** | not verified | 無 | [IEEE 10677532](https://ieeexplore.ieee.org/abstract/document/10677532/references)（片段：`"sales of 50 products in 10 different stores over 5 years"`） |
| **US Census MARTS** | **13** 個聚合產業 | 月 | — | not verified | 無 | [JSSAM 論文](https://academic.oup.com/jssam/advance-article-pdf/doi/10.1093/jssam/smae019/57257862/smae019.pdf) |

---

## 3. 規模比值（由上面數字實算）

| 比較 | 計算 | 結果 |
|---|---|---|
| 按序列（M5 總階層） | 42,840 ÷ 23 | **1,862.6×** |
| 按序列（M5 底部） | 30,490 ÷ 23 | **1,325.7×** |
| 按序列（M4 monthly） | 48,000 ÷ 23 | **2,087.0×** |
| **按觀測點（M5）** | 59,181,090 ÷ 6,049 | **≈9,784×** |
| **最貼近同制度基準（M1-monthly）** | 617 ÷ 23 | **27×** |

### 可直接引用的那一行

> project = 23 series / 6,049 obs. M5 = 42,840 series (30,490 bottom-level) / ≈59,181,090 obs *(days count not verified)*.
> ratio = **1,862× by series, ≈9,784× by observations**. Nearest same-regime benchmark: M1-monthly = 617 series / 44,892 obs / h=18 (**27× by series**).

⚠️ **不是 13,700×**（見 `CORRECTIONS_LOG.md` 第 2 條）。

---

## 4. 月度／少序列的學術基準（與本專案可比者）

| 基準 | 規模 | 為何可比 | 連結 |
|---|---|---|---|
| **M1-monthly** ⭐ | **617 條月度序列 / 44,892 觀測 / h=18** | **M 系列裡最貼近「月度、K 級以下序列」的官方競賽基準** | [arXiv 2308.05566](https://ar5iv.labs.arxiv.org/html/2308.05566v1) |
| **US Census MARTS** ⭐ | **13 條月度零售產業** | **規模與本專案 23 條同級，且同為零售名目金額** → **比拿 M5 對照貼切得多** | [JSSAM](https://academic.oup.com/jssam/advance-article-pdf/doi/10.1093/jssam/smae019/57257862/smae019.pdf) |
| Kaggle Store Item Demand | 500 條日序列 / 5 年 | 「小規模多序列」常用替代基準 | [IEEE 10677532](https://ieeexplore.ieee.org/abstract/document/10677532/references) |
| M3-monthly / 澳洲 domestic tourism / 澳洲監獄人口（8 州月度） | **規模數字未取得**（僅見資料集頁存在） | — | [Nixtla prison 範例](https://nixtlaverse.nixtla.io/hierarchicalforecast/examples/australianprisonpopulation.html) |

---

## 5. 廠商宣稱的處理規模：**全部無可核實來源**

已執行的查詢字串（6 組）：`RELEX "millions of SKUs"`、`Blue Yonder "billions" SKU-store`、`SAP IBP "millions of SKUs"`、`o9 "billions of" SKU-store`、`RELEX "billions of" forecasts`、`Blue Yonder "trillions"`。

回傳全部為行銷手冊／案例頁，**片段內皆無數字**：

- [RELEX brochure](https://www.relexsolutions.com/wp-content/uploads/2017/10/relex_us_brochure_online_2017.pdf)
- [Blue Yonder AI in retail](https://now.blueyonder.com/rs/366-TWM-779/images/AI_Insights_TransformRetail.pdf)
- [Oracle Retail Demand Forecast datasheet](http://www.oracle.com/applications/retail/library/data-sheets/demand-forecast.pdf)
- [Sainsbury's / Blue Yonder 案例（1,400 店）](https://www.sityos.com/en/use-cases/how-sainsburys-cut-stockouts-across-1400-stores-blue-yonder-ai)

**報告若要寫，只能寫**：「廠商宣稱覆蓋大型零售商的數百萬 SKU／數千店，**具體每客戶規模無公開可核實數字**」，並標為 **vendor claim**。

---

## 6. 這個對照真正解釋了什麼（機制層面）

1. **「贏 LightGBM」的可推廣性極低，原因在協變量不在模型。** M5 的 GBDT 靠的是**價格／促銷／行事曆**這類高信息量欄位 × 42,840 條序列。本專案實測：13 條外生加進去**更差**（天氣距平 **+26.4%**、詞典文本 **+13.3%**，表 3-6）——23 條序列養不起 8 個欄位（+32 參數/欄 + 1,088 一次性），而天氣距平在 2023-09 達 **+4.976σ**（世紀黑雨），訓練期從未見過 → 不受控外推。
2. **本專案落在「統計方法仍有競爭力」的制度裡，不是 M5 的制度。** M5 = 大量序列 ＋ 協變量 → ML 贏；M3 月度 = 少序列、小樣本 → 統計法贏。本專案屬後者，這也是 naive 在 h=6（0.97471）、h=12（0.80900）**反贏最終模型**的原因。
3. **唯一能搬去商業規模的東西是「層級調和」，不是模型。** `MinT_wls` 把「母行業 = 子行業之和」的約束違反量由 **161.63 HK$m 壓到 1e-13**，與序列數量無關，在 M5 與任何企業預算流程裡都是剛需。

---

## 7. `not verified` 清單（誠實記錄）

1. **M5**：1,941 天與 2011-01-29～2016-06-19 日期範圍、協變量清單（price/promo/calendar/SNAP 的實際欄名）。
2. **M4**：總觀測點數；Quarterly 24,000、Weekly 359；各頻率訓練/測試切分。
3. **M3**：分頻率序列數 645/756/1428/174；horizon 18 與 8。
4. **Rossmann**：942 天、確切日期範圍、列數、協變量清單。
5. **Favorita**：店數、item 數、125,497,040 的原始出處、日期範圍、協變量。
6. **廠商規模**：全部（見 §5）。

**解鎖步驟**：上述每一項都能靠打開已在表內附上的官方頁面（Kaggle data 頁、M5 Competitors Guide、IJF 論文 PDF、Zenodo 資料集頁）逐項確認——只是本輪工具集不支援頁面抓取，**未將其視為「不存在」**。
