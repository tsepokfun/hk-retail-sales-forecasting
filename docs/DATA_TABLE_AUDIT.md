# C&SD 已下載表：真實序列數、期間、重疊審計

> 產生：`stage0/audit_censtatd_tables.py` ｜ 視窗 200410–202608 ｜ 只計 `freq=='M'` 且 `figure` 可解析為數值的列

| 表號 | 標題 | sv | 序列數 | 期間數(窗內) | **窗內格數** | 期間範圍 | 被誰包含 |
|---|---|---|---:|---:|---:|---|---|
| `620-67002` | Value and value index of retail sales by t | VAL_IDX_RS | 25 | 263 | **6575** | 200410–202608 | — |
| `620-67002` | Value and value index of retail sales by t | VAL_RS | 25 | 263 | **6575** | 200410–202608 | — |
| `620-67003` | Volume index of retail sales by type of re | VOL_IDX_RS | 25 | 263 | **6575** | 200410–202608 | — |
| `510-60003` | Consumer Price Indices (October 2019 – Sep | A_CM_1920 | 14 | 260 | **3640** | 200501–202609 | — |
| `510-60003` | Consumer Price Indices (October 2019 – Sep | B_CM_1920 | 14 | 260 | **3640** | 200501–202609 | — |
| `510-60003` | Consumer Price Indices (October 2019 – Sep | CC_CM_1920 | 14 | 260 | **3640** | 200501–202609 | — |
| `510-60003` | Consumer Price Indices (October 2019 – Sep | C_CM_1920 | 14 | 260 | **3640** | 200501–202609 | — |
| `650-80001` | Visitor arrivals by nationality/region | VIS_ARR | 12 | 263 | **3156** | 200401–202608 | — |
| `510-60001A` | Consumer Price Indices (October 2019 – Sep | A_CM_1920 | 10 | 263 | **2630** | 197407–202609 | — |
| `510-60001A` | Consumer Price Indices (October 2019 – Sep | B_CM_1920 | 10 | 263 | **2630** | 197407–202609 | — |
| `510-60001A` | Consumer Price Indices (October 2019 – Sep | CC_CM_1920 | 10 | 263 | **2630** | 198010–202609 | — |
| `510-60001A` | Consumer Price Indices (October 2019 – Sep | C_CM_1920 | 10 | 263 | **2630** | 197407–202609 | — |
| `620-67012` | Value of retail sales in department stores | VAL_RS | 9 | 263 | **2367** | 200410–202608 | — |
| `410-55111A` | Seaborne cargo throughput by month | PORT_CARGO_TP | 7 | 262 | **1834** | 199701–202607 | — |
| `410-55112A` | River cargo throughput by month | PORT_CARGO_TP | 7 | 262 | **1834** | 199801–202607 | — |
| `215-16004` | Number of establishments, persons engaged  | EST | 17 | 82 | **1394** | 200603–202606 | — |
| `215-16004` | Number of establishments, persons engaged  | PE | 17 | 82 | **1394** | 200603–202606 | — |
| `215-16004` | Number of establishments, persons engaged  | VAC | 17 | 82 | **1394** | 200603–202606 | — |
| `625-68004` | Restaurant receipts by type of restaurant  | VAL_IDX_RR | 6 | 222 | **1332** | 200801–202606 | — |
| `625-68004` | Restaurant receipts by type of restaurant  | VAL_RR | 6 | 222 | **1332** | 200801–202606 | — |
| `625-68004` | Restaurant receipts by type of restaurant  | VOL_IDX_RR | 6 | 222 | **1332** | 200801–202606 | — |
| `340-45021` | Hong Kong dollar interest rates | BL_RATE | 1 | 263 | **263** | 198001–202609 | — |
| `340-45021` | Hong Kong dollar interest rates | S_DEP_RATE | 1 | 221 | **221** | 198001–202511 | — |
| `340-45021` | Hong Kong dollar interest rates | T_DEP_RATE | 5 | 263 | **1315** | 198001–202609 | — |
| `620-67011` | Value of retail sales in supermarkets by b | VAL_RS | 5 | 263 | **1315** | 200410–202608 | — |
| `340-46002` | Effective exchange rate indices for the Ho | EERI_E | 1 | 263 | **263** | 197501–202609 | — |
| `340-46002` | Effective exchange rate indices for the Ho | EERI_I | 1 | 263 | **263** | 197501–202609 | — |
| `340-46002` | Effective exchange rate indices for the Ho | EERI_T | 1 | 263 | **263** | 197501–202609 | — |
| `620-67001` | Total retail sales | VAL_IDX_RS | 1 | 263 | **263** | 200410–202608 | 620-67002 |
| `620-67001` | Total retail sales | VAL_RS | 1 | 263 | **263** | 200410–202608 | 620-67002 |
| `620-67001` | Total retail sales | VOL_IDX_RS | 1 | 263 | **263** | 200410–202608 | 620-67003 |
| `620-67021` | Seasonally adjusted series of retail sales | SA_VAL_IDX_RS | 1 | 263 | **263** | 200410–202608 | — |
| `620-67021` | Seasonally adjusted series of retail sales | SA_VOL_IDX_RS | 1 | 263 | **263** | 200410–202608 | — |
| `620-67031` | Value of online retail sales by selected t | VAL_ONLINE_RS | 3 | 80 | **240** | 202001–202608 | — |
| `625-68002` | Total restaurant receipts and restaurant p | VAL_IDX_RR | 1 | 222 | **222** | 200801–202606 | — |
| `625-68002` | Total restaurant receipts and restaurant p | VAL_RP | 1 | 222 | **222** | 200801–202606 | — |
| `625-68002` | Total restaurant receipts and restaurant p | VAL_RR | 1 | 222 | **222** | 200801–202606 | — |
| `625-68002` | Total restaurant receipts and restaurant p | VOL_IDX_RR | 1 | 222 | **222** | 200801–202606 | — |

**原始合計（未去重）：68520 格**

> 判讀規則：`被誰包含` 非空者，其序列集合是另一張表的子集，
> 納入面板時不應重複計算。實際可用擴張必須扣掉這些。
