# 資料擴充與公平協議對決：實測結果

> 本檔案由 `stage0/run_data_and_rolling_upgrade.py report` **自動從結果 csv 生成**，不經人手抄寫。
> 生成時間：2026-10-23 16:08 ｜ 評估一律為 `harness.py` 的 1,104 點協定。

## 表 0｜對照錨點（前幾輪由鎖死 harness 實測）

| 基準 | MASE_ALL |
|---|---:|
| SeasonalNaive | 1.03029 |
| LightGBMTuned | 1.04108 |
| xLSTM (frozen 2021-11) | 0.94810 |
| xLSTM + MinT_wls (frozen) | 0.94455 |
| iTransformer (frozen 2021-11) | 0.90198 |
| ETS rolling refit | 0.85403 |
| ETS rolling + MinT_wls | 0.84927 |
| SARIMAX rolling refit | 0.88354 |
| SARIMAX rolling + MinT_wls | 0.88326 |

## 表 1｜公平 Rolling 協議：神經模型 vs 統計模型

### Rolling refit（每個測試原點重訓 + per-horizon 因果遮罩）

**`metrics_itransformer_rolling.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| itransformer_rolling_s42 | **0.94093** | 0.0916 |
| itransformer_rolling_ENS3+MinT_wls | **0.94615** | 0.0929 |
| itransformer_rolling_ENS3 | **0.94635** | 0.0927 |
| itransformer_rolling_s2024 | **0.96002** | 0.0931 |
| itransformer_rolling_s7 | **1.00947** | 0.0997 |

**`metrics_itransformer_rolling_fx45.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| itransformer_rolling_s42 | **0.95025** | 0.0921 |
| itransformer_rolling_ENS3 | **0.98043** | 0.0962 |
| itransformer_rolling_ENS3+MinT_wls | **0.98104** | 0.0964 |
| itransformer_rolling_s7 | **1.01965** | 0.1011 |
| itransformer_rolling_s2024 | **1.03479** | 0.1004 |

**`metrics_itransformer_rolling_nopartial.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| itransformer_rolling_s42 | **0.95568** | 0.0931 |
| itransformer_rolling_ENS3 | **0.96192** | 0.0943 |
| itransformer_rolling_ENS3+MinT_wls | **0.96228** | 0.0946 |
| itransformer_rolling_s2024 | **0.97406** | 0.0948 |
| itransformer_rolling_s7 | **1.02583** | 0.1013 |

**`metrics_itransformer_rolling_val24.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| itransformer_rolling_s2024 | **0.91091** | 0.0885 |
| itransformer_rolling_ENS3 | **0.92018** | 0.0896 |
| itransformer_rolling_ENS3+MinT_wls | **0.92044** | 0.0899 |
| itransformer_rolling_s42 | **0.93329** | 0.0906 |
| itransformer_rolling_s7 | **0.94244** | 0.0922 |

**`metrics_xlstm_rolling_full.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| xlstm_rolling_ENS3+MinT_wls | **0.93087** | 0.0965 |
| xlstm_rolling_ENS3 | **0.93328** | 0.0964 |
| xlstm_rolling_s42 | **0.95729** | 0.0989 |
| xlstm_rolling_s7 | **0.98704** | 0.1026 |
| xlstm_rolling_s2024 | **0.99280** | 0.1027 |

判讀：與表 0 的 `ETS rolling` 0.85403 / `SARIMAX rolling` 0.88354 直接比較。

## 表 2｜23 條序列 vs 34 條序列（+47.8% 樣本）

### 34 條序列（訓練用 34、評估鎖定原 23）

**`metrics_all_23.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| iTransformer_23_ENS3 | **0.90198** | 0.0878 |
| iTransformer_34_ENS3 | **0.92804** | 0.0902 |
| xLSTM_23_ENS3+MinT_wls | **0.94455** | 0.1003 |
| xLSTM_23_ENS3 | **0.94810** | 0.1002 |
| xLSTM_34_ENS3+MinT_wls | **0.98326** | 0.1000 |
| xLSTM_34_ENS3 | **0.98759** | 0.1004 |
| xLSTM_34_ENS3+BU | **0.99031** | 0.1004 |
| SeasonalNaive | **1.03029** | 0.1044 |
| xLSTM_34_ENS3+MinT_ols | **1.04649** | 0.1011 |

**`metrics_per_seed_23.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| iTransformer_23_s7 | **0.91178** | 0.0885 |
| iTransformer_23_s42 | **0.91499** | 0.0890 |
| iTransformer_23_s2024 | **0.91539** | 0.0895 |
| xLSTM_23_s7 | **0.95989** | 0.0985 |
| xLSTM_23_s2024 | **0.97333** | 0.1032 |
| xLSTM_23_s42 | **1.01743** | 0.1090 |

**`metrics_v23val_23.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| iTransformer_34_v23val_ENS3 | **0.91128** | 0.0883 |
| xLSTM_34_v23val_ENS3 | **0.96152** | 0.0981 |

### 穩健性：34 條的退步是「每個種子都退」還是「集成位移」？

| 家族 | 種子 | 23 條 | 34 條 | 差 |
|---|---|---:|---:|---:|
| iTransformer | s42 | 0.91499 | 0.94086 | **+0.02587** |
| iTransformer | s7 | 0.91178 | 0.94740 | **+0.03562** |
| iTransformer | s2024 | 0.91539 | 0.91899 | **+0.00360** |
| iTransformer | **平均** | 0.91405 | 0.93575 | **+0.02170**（3/3 個種子變差） |
| xLSTM | s42 | 1.01743 | 1.02394 | **+0.00651** |
| xLSTM | s7 | 0.95989 | 1.01731 | **+0.05742** |
| xLSTM | s2024 | 0.97333 | 1.05352 | **+0.08019** |
| xLSTM | **平均** | 0.98355 | 1.03159 | **+0.04804**（3/3 個種子變差） |

> 判讀：若**每個種子**都變差，則結論不依賴集成組成，可直接寫成「加子序列有害」。

## 表 3｜M1/M3 預訓練 → 微調 vs 從頭訓練

### M1+M3 月度序列預訓練 20 epochs 後微調

**`metrics.csv`**

| 模型 | MASE_ALL | WAPE_ALL |
|---|---:|---:|
| xLSTM_pretrained_ENS3+MinT_wls | **0.92354** | 0.0917 |
| xLSTM_pretrained_ENS3 | **0.92469** | 0.0916 |
| xLSTM_scratch_ENS3+MinT_wls | **0.94455** | 0.1003 |
| xLSTM_scratch_ENS3 | **0.94810** | 0.1002 |
| xLSTM_pretrained_s2024 | **0.95355** | 0.0959 |
| xLSTM_scratch_s7 | **0.95989** | 0.0985 |
| xLSTM_pretrained_s42 | **0.96120** | 0.0920 |
| xLSTM_scratch_s2024 | **0.97333** | 0.1032 |
| xLSTM_pretrained_s7 | **0.99870** | 0.1037 |
| xLSTM_scratch_s42 | **1.01743** | 0.1090 |

### 穩健性：改善是「每個種子都變好」還是「集成組成改變」？

| 種子 | scratch | pretrained | 差 |
|---|---:|---:|---:|
| _s42 | 1.01743 | 0.96120 | -0.05623 |
| _s7 | 0.95989 | 0.99870 | +0.03881 |
| _s2024 | 0.97333 | 0.95355 | -0.01978 |

- 逐種子平均差：**-0.01240** ｜ 種子平均 MASE 0.98355 → 0.97115
- 有改善的種子：**2/3**

> 判讀：若「逐種子平均差」的絕對值小於 0.02 門檻，但 ENS3 的差距大於門檻，
> 則改善**部分來自集成組成**（種子間差異變大），不可全部歸因於預訓練。

## 判讀規則（先寫死，避免事後詮釋）

1. 種子雜訊門檻 **0.02 MASE** 沿用 `FINAL_REPORT.md` §4.1 的推導；差距小於此值一律寫「方向一致，不顯著」。
2. 統計模型（ETS/SARIMAX）是**確定性**的，沒有種子；門檻只用來判讀神經模型那一側的不確定性。
3. 三張表全部在**同一 1,104 點**上結算；任何改動評估集合的結果一律不列入。
