# iTransformer 與 xLSTM：論文原話與後續批判

> 資料日期：2026-10
> 取證限制：僅 `web_search`（URL 清單 + 零散片段），**無 PDF 全文抓取能力**。兩篇論文的 abstract 逐字原文均未取得，只能 paraphrase；凡未出現在片段者標 `not found`。
> 本文只彙整已查證結論，不含新推導。

---

## A. iTransformer：論文原話與機制

出處：Liu et al., **ICLR 2024**（清華 thuml）。[官方 PDF](https://ise.thss.tsinghua.edu.cn/~mlong/doc/iTransformer-iclr24.pdf)、[GitHub](https://github.com/thuml/iTransformer)、[arXiv 2310.06625](https://export.arxiv.org/abs/2310.06625v4)。

### A.1 機制（paraphrase，非逐字）

- 把輸入矩陣**轉置後切片**：每個 token 是「一條變量（variate）的整個 lookback 序列」，而非一個時間步。
- self-attention 因此在 **variate 維度**運作（cross-variate / cross-channel attention）。
- FFN 逐 variate token 學非線性表徵。
- 輸出頭把每個 token 的 d_model **線性投影到 horizon 長度**。
- **變量 token 由跨變量共享的投影 `Linear(lookback → d_model)` 產生** → 增加變量數**不改參數數**，只加寬注意力序列。

### A.2 唯一取到的逐字片段（p.2）

> `"With denotation of q_i, k_j ∈ R^{d_k} as the specific query and key of one (variate) token, …"`

### A.3 論文的 benchmark 制度

| 項目 | 內容 |
|---|---|
| 數據集 | 7 個公開數據集（ETTh1/h2/m1/m2、ECL、Traffic、Weather、Exchange 類） |
| horizon | **{96, 192, 336, 720}** |
| 指標 | MSE / MAE |
| 對比句（片段，後段被截斷） | `"We compare the performance of Transformer and iTransformer on all datasets in Table 7. Consistent and great promotions can be ac…"` |
| **勝過前 SOTA 幾 %** | **not found**（無法取得表格數值） |

⚠️ 這個制度是**數百通道、lookback ≥96、horizon ≥96**，與「月度、23 條序列、lookback 12」不同世界。

### A.4 批判的落點：官方 repo 內建 instance normalization（RevIN 式）**預設啟用**

[thuml/iTransformer](https://github.com/thuml/iTransformer)。這是下面所有後續批判的共同落點。

---

## B. iTransformer：後續批判與反證（五條）

| # | 文獻 | 核心主張 / 可引片段 | 對「月度、23 序列」制度的意涵 |
|---|---|---|---|
| 1 | **《Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting?》** [arXiv 2502.09683](https://ar5iv.labs.arxiv.org/html/2502.09683)（**預印本**） | 把「**通道依賴、有限 lookback、數據集過簡**」列為三大偏差來源；標題即結論 | lookback L=12、23 條序列、263 個月，正好落在其指稱的偏差維度上 |
| 2 | Mendelowitz, **《Equal Accuracy, Unequal Cost: Channel Dependence in PatchTST under Controlled Coupling》** [PDF](https://github.com/AdiMendelowitz/ci-cd-patchtst/blob/v1.0-tmlr/mendelowitz2026equalaccuracy.pdf)（**預印本**，託管於作者 GitHub） | 在**受控耦合**條件下 CD 與 CI **近乎等準**。片段：`"The empirical near-parity result above raises a natural question: how much accuracy was actually available for CD to win"` | 本專案做了同骨幹控制組，但結論與此相反（通道混合可歸因 −0.05407 MASE）；兩者需並列呈現 |
| 3 | **《There are no Champions in Supervised Long-Term Time Series Forecasting》** [arXiv 2502.14045](http://export.arxiv.org/pdf/2502.14045v1)（**預印本**） | 增刪**一個**數據集就能翻轉排名。片段：`"This experiment provides a practical sense of how much the addition/removal of a single dataset may affect concl…"` | 本專案只有單一測試年（2024），同一種病、規模更小 |
| 4 | **《Is the Attention Matrix Really the Key to Self-Attention in Multivariate Long-Term Time Series Forecasting?》** [ACL 2026](https://aclanthology.org/2026.acl-long.853.pdf)（**同儕審查**） | 變體把注意力換成逐元素式 `S = GELU(Q ⊙ K), O = S ⊙ V` 仍可運作 → **attention 矩陣本身可能不是關鍵** | 「跨品類混合」的功勞歸給 attention 可能高估了機制 |
| 5 | [arXiv 2607.19404](https://export.arxiv.org/pdf/2607.19404)（**預印本**） | 在 `"weak inter-channel correlation and rich multi-scale periodicity"` 的數據上，ablation 顯示最大增益份額來自**其他模組**（如週期性建模），非跨通道 | 本專案 23 條序列相關性高，條件相反；但提醒必須有控制組才能談歸因 |

**另有一條需存疑的實作主張**（非官方 repo、博客性質）：[TST-Degeneration README](https://raw.githubusercontent.com/Greek-Guardian/TST-Degeneration/refs/heads/main/README.md) 主張注意力退化成 MLP。

**反證缺口**：**未檢索到任何論文直接報告「iTransformer 在 20–30 條序列上輸給線性基線」** — 這是缺席，不是反證。

---

## C. xLSTM：論文原話與機制

出處：Beck et al., **NeurIPS 2024**。[NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2024/file/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Paper-Conference.pdf)、[arXiv 2405.04517](https://arxiv.org/abs/2405.04517v2)、[摘要頁](https://papers.nips.cc/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html)。

### C.1 機制（paraphrase）

| 元件 | 記憶形態 | 可否時間並行 | 說明 |
|---|---|---|---|
| **sLSTM** | 純量記憶 cell ＋ hidden-hidden 遞迴（**memory mixing**） | **否** | 保留 memory mixing 的純遞歸版本，配 conv1d |
| **mLSTM** | **矩陣記憶** `C ∈ R^{d×d}`，以 key/value 外積的 covariance 規則更新 | **是** | **無 memory mixing** |

兩者共用 **exponential gating**（輸入／遺忘門指數化）＋ normalizer state ＋ stabilizer。

可檢索到的鄰近原文片段：`"Conceptually, the closest models to xLSTM are Retention (Sun et al.)…"`

### C.2 「abandons memory mixing」的裁決

| 項目 | 結果 |
|---|---|
| 本專案筆記的說法 | 「To enable parallelization, the mLSTM abandons memory mixing」（原本當作逐字引文） |
| 外部核實 | **方向正確，但無法逐字核實** |
| 可確認的部分 | mLSTM 的並行性來自「**去掉 hidden-hidden 遞迴／memory mixing**」這條結構性理由 |
| 不可確認的部分 | 論文中**沒有** `parallel prefix sum` 字樣 |
| **裁決** | **必須寫成 paraphrase，不可加引號** |

### C.3 評估範圍：**確實零時序預測**（外部確認）

| 項目 | 內容 |
|---|---|
| 語言模型 | 15B tokens SlimPajama。片段：`"Figure 15: Method comparison on next token prediction when trained on 15B tokens from SlimPajama"`（[JKU PDF](https://epub.jku.at/obvulihs/download/pdf/13699943)） |
| 合成任務 | Nearest Neighbor Search、Cycle Nav。片段：`"Nearest Neighbor Search Task"`（[xLSTM PDF p.11](https://export.arxiv.org/pdf/2405.04517#11#12#11)） |
| **時序預測** | **零**（本專案筆記的 0 命中檢查，經外部任務清單確認） |
| perplexity 具體值 | **not found** |

---

## D. xLSTM 在時序上的後續實測

| 工作 | 內容 | 證據等級 |
|---|---|---|
| **xLSTMTime** [arXiv 2407.10240](http://export.arxiv.org/abs/2407.10240) | 把 xLSTM 搬上 ETT／ECL／Traffic／Weather 長期基準與 Transformer 系比較 | 預印本 |
| **xLSTM-Mixer** [arXiv 2410.16928](https://arxiv.org/abs/2410.16928)、[NeurIPS 2025 slides](https://neurips.cc/media/neurips-2025/Slides/118697_TcsPyYz.pdf) | 通道獨立 ＋ 跨通道 scalar-memory 混合，多變量長期預測 | **同儕審查（NeurIPS 2025）** |
| 綜述式肯定 | `"In long-term forecasting tasks, xLSTM-based models generally demonstrate superior performance compared to Transformer-based a…"` [arXiv 2509.01187](https://backend.dev.arxiv.org/pdf/2509.01187) | 預印本 |
| 零售／貨運個案 | **IPSO-xLSTM** 用於貨運量預測與倉儲配置 | 會議論文，**非月度零售小樣本** [ACM](https://dl.acm.org/doi/pdf/10.1145/3768801.3768854) |
| **TiRex / TiRex-2**（xLSTM 骨幹，基礎模型） | TiRex NeurIPS 2025；TiRex-2（2026-07）多變量＋串流 | 見 `LITERATURE_2026_LANDSCAPE.md` |

### ⚠️ 文獻互相衝突（必須明示）

- `2509.01187`：xLSTM 系在長期預測普遍勝 Transformer 系。
- `2502.09683` / `2502.14045`：這類基準排名本身受數據集選擇偏差支配。
- **兩者不能同時當作強證據。**

---

## E. `xlstm` 套件的實務坑（上游 issue 核實結果）

| 問題 | 上游狀態 |
|---|---|
| `nvcc fatal : Unknown option '-extra-device-vectorization'` | **已上游記錄** — [NX-AI/xlstm Issue #39](https://github.com/NX-AI/xlstm/issues/39) |
| `[bug] Incompatible with torch >= 2.6.0` | **已上游記錄** — [Issue #66](https://github.com/NX-AI/xlstm/issues/66) |
| **Windows 專屬 issue** | **not found**（未檢索到單獨的 Windows issue）[issues 列表](https://github.com/NX-AI/xlstm/issues) |
| **Triton 明文依賴** | **not found**（未取得上游明文）[README](https://raw.githubusercontent.com/NX-AI/xlstm/refs/heads/main/README.md) |

**本機實測的三個坑**（`00_PROJECT_INFO.md` §八）——`backend="vanilla"` 必須明寫、`mLSTMLayerConfig` 無 `backend` 參數、Triton 缺 Windows wheel——屬**本地證據**，**不可寫成「上游已知」**。

---

## F. 關鍵結論：本專案的制度沒有任何已發表評測證據

> **檢索不到任何針對「月度、20–30 條序列、每條約 250 點、階層式」的 iTransformer 或 xLSTM 評測。** 所有可得證據都在長期時序基準（數百通道、lookback ≥96、horizon ≥96）或大規模語言模型上。

### 因此報告唯一能寫的句式

| 不可寫 | 只能寫 |
|---|---|
| 「iTransformer / xLSTM 對本制度有效（或無效）」 | 「**本制度無外部證據**；兩架構的增益來源（跨通道注意力／矩陣記憶）在 n≈23、L=12 下**均未被驗證**」 |

### 方向相反的旁證

一份 UTwente 碩論分析 M5，指出主要方法增益來自 **cross-learning（跨序列共學）而非架構本身** — [PDF](http://essay.utwente.nl/95006/1/Fitria_MA_EEMCS.pdf)（**非同儕審查**）。這支持「機制」而非「架構」。

---

## G. 本專案控制組與批判 #2 的關係（需標為推論）

| 事實（有出處） | 來源 |
|---|---|
| `TimeTokenCINet` 是同 class、同嵌入寬度、同 encoder、同輸出頭、同訓練循環、同錨點，只把 attention 從 23 條序列換回 12 個時間步 | `stage0/models/itransformer.py` L55–64、L190 |
| instance normalisation **兩條臂共用**（S4 對 iTransformer 保留、雙胞胎描述為 identical） | 同上 S4 |
| 全專案實測：通道混合可歸因 **−0.05407 MASE，三層 DM 全顯著** | `FINAL_REPORT.md` §2.3 |
| 兩條臂**都拿不到** 27 個工程特徵（S1） | `itransformer.py` S1 |

**推論（非直接引用）**：由於 norm 兩臂共用，−0.05407 的差值**不可能由 instance normalisation 造成**；但 S1 使該比較成為「通道混合效果」的**下限**，不是上限。

⚠️ 此推論與批判 #2（Mendelowitz 的 CD-CI near-parity）方向相反，**必須並列呈現，不可只引對自己有利的一方**。

---

## H. 「找不到」清單

1. 兩篇論文的 **abstract 逐字原文**（工具無 HTTP fetch）。
2. iTransformer 相對前 SOTA 的 **MSE 降幅 %**。
3. xLSTM 的 **perplexity／downstream headline 數值**。
4. xLSTMTime 與 xLSTM-Mixer 的**具體 MSE/MAE 與改進 %**（只取到表格殘片如 `0.366`）。
5. 「iTransformer 在少通道／短 lookback 上失敗」的**直接實驗**（只有 2502.09683 的標題級主張）。
6. `xlstm` 的 **Windows 專屬 issue 與 Triton 明文依賴**。
7. **月度／階層式零售 ＋ iTransformer 或 xLSTM 的任何研究**。
