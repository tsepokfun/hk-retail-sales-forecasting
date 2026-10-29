# Data Source Discovery — Branch: Apparel / Footwear / Fashion

Target outlet types (censtatd 620-67002): `12 Clothing, footwear and allied products`, `13 Wearing apparel`, `14 Footwear, allied products and other clothing accessories`. Window: monthly **2004-10 → 2026-08**.
All fetching via `pwsh` `Invoke-WebRequest` (+ Python `xlrd 2.0.2` / `openpyxl 3.1.5`), `Start-Sleep 3` between same-host calls.

## 1. Verified sources table

| source | exact URL | what it gives | freq | months covered | free? | known-at-time? | verification evidence |
|---|---|---|---|---|---|---|---|
| **HKO Monthly Extract** | `https://www.hko.gov.hk/cis/monthlyExtract/monthlyExtract.xml` | mean pressure, abs/mean max temp, mean temp, mean/abs min temp, dew point, RH, cloud, **rainfall mm**, **sunshine hrs** | monthly | **1884-01 → 2026-09** (143 yrs, no gaps 2004+) | yes | **yes**, no revisions | 200 `text/xml`, **169,804 B**. Columns matched vs daily API 6/6 months |
| **HKO daily temp API** | `https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP\|CLMMAXT\|CLMMINT&lang=en&rformat=json&station=HKO` | daily mean/max/min °C | daily | **1884-03-01 → 2026-08-31** | yes | yes | 200 `application/json`, **1,433,043 B**. Last row `["2026","8","31","27.7","C"]` |
| **HKO daily extract** | `https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_YYYYMM.xml` | day pressure, max/mean/min temp, dewpoint, RH, cloud, **rainfall** | monthly files | **2015-01 → present** | yes | yes | 200 `text/xml` ~2,139 B; `…201401.xml` → **404**, `…201501.xml` → 200 |
| **World Bank Pink Sheet** | `https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx` | **Cotton, A Index** ($/kg), crude oil etc. | monthly | **1960-01 → 2025-12**, n=792 continuous | yes | **no** — revised; file self-labels "Updated on January 06, 2025" | 200, spreadsheetml, **778,415 B**, magic `80,75,3,4`. Sheet `Monthly Prices`, col 54 |
| **RVD Private Retail rents/prices** | `https://www.rvd.gov.hk/doc/en/statistics/his_data_11.xls` | retail rents HK$/m²/mo + prices HK$/m², HK/Kowloon/NT | monthly, quarterly, annual sheets | monthly **1999-01 → 2025-08 (270)** but **2004 = 0 months, 2005-08 = 2 months/yr** | yes | no — appraisal-based, revised | 200 `vnd.ms-excel`, **228,352 B**, OLE2 `208,207,17,224`. xlrd: cols 1=yr 5=mo 8/11/14=rents 17/20/23=prices |
| **RVD Private Retail indices** | `https://www.rvd.gov.hk/doc/en/statistics/his_data_12.xls` | retail rental + price index (1999=100) | monthly, quarterly | monthly **1993-01 → 2025-08 (368)**; **2004 & 2005 = 0 months** | yes | no — revised | 200, **146,432 B**, same magic. Cell: `PRIVATE RETAIL - RENTAL AND PRICE INDICES (1999=100)` |
| **censtatd visitor arrivals** | `https://www.censtatd.gov.hk/api/get.php?id=650-80001&lang=en&full_series=1` | arrivals by nationality/region | **monthly** + annual | **2004 → 2026-07**, `period`=`YYYYMM` | yes | no — revised | 200 `application/json`, **936,705 B**; `freq={'Y':264,'M':3252}` |
| **censtatd visitor spend** | `…/api/get.php?id=650-80003` / `…650-80004` | destination consumption expenditure; per-capita spend | **ANNUAL ONLY** | 1998 → 2025 | yes | no | 200, **177,217 B** / **173,587 B**; `freq={'Y':504}` both — **no monthly** |
| **UN Comtrade** | `https://comtradeapi.un.org/public/v1/preview/C/M/HS?reporterCode=344&period=YYYYMM&cmdCode=61&flowCode=M` | HK trade by HS chapter (61/62/64) | monthly | **~2015-01 → present**; 2013-01 count 1, 2004-10 count 0 | yes | no — heavily revised | 2015-01: 200, count 71, 63,113 B; 2024-01: count 77, 68,535 B |
| **HKEXnews API** | `https://www1.hkexnews.hk/search/titleSearchServlet.do?…&stockId=16111&title=operational…` (+ `prefix.do` for stockId) | quarterly **operational updates** | quarterly | Anta **2016-10-27 → 2026-07-17**; Li Ning **2011-07-07 → 2026-07-15** | yes | yes | `prefix.do` 200 → `{"stockId":16111,"code":"02020"}`; servlet 200 `application/json`, 36,472 B, `recordCnt=42` |
| **Fast Retailing monthly** | `https://www.fastretailing.com/eng/ir/monthly/` | UNIQLO Japan same-store+online y/y, customer numbers | monthly, **current month only** | Sep 2026 only, **no archive** | yes | yes | 200 `text/html`, 36,983 B; `archive.html` → **404**, `/pdf/` → **403** |
| censtatd table index | `https://www.censtatd.gov.hk/data/en/all_web_tables.json` | 590 table titles | — | — | yes | — | 200, 147,110 B, 590 entries. grep `cloth\|textil\|garment\|footwear\|apparel\|wearing` → **0 hits** |

## 2. Detail

**HKO `monthlyExtract.xml`** — JSON-in-XML, `stn.data[*].yearData[*]`, 14 fields, **no `***` gaps from 2004 on**. Column mapping proved by aggregating the daily APIs to monthly means: col3 = mean daily max (2004-01: 17.5 vs 17.53; 2004-07: 30.5 vs 30.51), col4 = **mean temp** (15.8/15.82; 28.4/28.42; 16.4/16.45; 30.2/30.24; 17.5/17.55; 28.7/28.66 — 6/6), col5 = mean daily min (14.2/14.17; 26.6/26.62). col10 = **total rainfall mm** (emits `Trace`), col11 = **total sunshine hours** (Jan 136.8 / Jul 172.2 in 2004; 122.2 in a 790.3 mm Jul 2026 — coherent). **cols 12-13 UNIDENTIFIED** (`050`/`230`, `26.7`/`23.3`) — read the legend at `https://www.hko.gov.hk/en/cis/monthlyExtract.htm` before use. Fetch: `(Invoke-WebRequest '<url>' -UseBasicParsing).Content` → `json.loads(...)['stn']['data']`.

**Pink Sheet** — transposed: row 4 = headers, row 5 = units, col 0 = `YYYYMnn`, data from row 6. Cotton A Index = **col 54, $/kg, nominal USD**. URL carries a content hash — pin the file locally and record the internal "Updated on …" label as the vintage.

**RVD** — `his_data_11` = *Average Rents and Prices*, `his_data_12` = *Rental and Price Indices*, each with Monthly/Quarterly/Annual sheets. The **Monthly sheets have a hard hole across 2004-2005** — inside the target window. Quarterly sheets are the continuous fallback. RVD publishes "within the first 6 working days of every month"; the `.xls` lags the press release.

**HKEXnews** — resolve ticker first: `prefix.do?callback=cb&lang=EN&type=A&name=ANTA&market=SEHK` → `stockId 16111`; then the servlet returns `result` as a JSON **string** needing a second parse. Anta and Li Ning file real quarterly operational updates; **Giordano (1666), Esprit (595), Shenzhou (10132) → `recordCnt=0` for `title=operational`** (semi-annual only). **Bossini: `prefix.do` returns an empty `stockInfo` array → no longer a listed issuer.**

## 3. Rejected / dead ends

- **HKO rainfall/sunshine via `opendata.php`** — brute-forced 16 candidate dataTypes (`CLMRN, CLMSUN, CLMRAIN, CLMRS, CLMGRS, CLMDBT, CLMWS, CLMPRES, CLMHUM, CLMCLD, CLMVIS, CLMRH, CLMSLP, CLMWIND, CLMSD, RYES`). **All returned a 726-byte stub** `Please include valid parameters in API request`. Only `CLMTEMP`, `CLMMAXT`, `CLMMINT` are valid → rainfall/sunshine must come from `monthlyExtract.xml` (monthly) or `dailyExtract_YYYYMM.xml` (2015+).
- **HKO daily extract before 2015** — `dailyExtract_200401/201001/201401.xml` → **404**; `201501/201601` → 200. Floor = **2015-01**.
- **censtatd 590-table index has no textile/clothing table** → trade flows cannot come from the web-tables API. `410-50031/50032` cover only the *ten principal commodity divisions*, excluding textiles.
- **Legacy UN Comtrade** `https://comtrade.un.org/api/get?...` → 200 but `text/html` (GTM/JS shell), not data.
- **Comtrade monthly before ~2014** — `period=200410` → `count 0`; `201301` → count 1. **Cannot cover 2004-10.**
- **RVD guessed paths** `/en/publications/property_review.html`, `statistics.html`, `his_statistics.html`, `/doc/en/statistics/property_review_2024.pdf` → all **404**. Real page: `/en/publications/property_market_statistics.html`.
- **data.gov.hk CKAN** `package_search?q=rainfall|sunshine|temperature` → 1 / 0 / 0 datasets.
- **Fast Retailing history** — monthly page shows only the latest month; `archive.html` 404, `/pdf/` 403.
- **UNVERIFIED (never fetched):** HKTB `partnernet.hktb.com` publications portal (URL appeared only inside a censtatd table note); Zara/Uniqlo/H&M/MUJI HK store counts or footfall; HKO `monthlyExtract` cols 12-13.

## 4. Alignment recipe

All sources key on calendar month `YYYY-MM`.

- **Weather (prefer `monthlyExtract.xml`)** — the month aggregate is precomputed, so **there is no daily→monthly step and therefore no future-day leak**. Use month *M* weather as a contemporaneous regressor: fully observed when *M* ends, while retail for *M* publishes ~4-5 weeks later → strictly known-at-time.
  - If using the daily APIs instead, aggregate **only** days with `day ≤ days_in_month(M)` — never a rolling 30-day window (that pulls from *M+1*). Map `Trace` → small positive (e.g. 0.05), `***` → NaN, and require ≥90% non-missing days before emitting a monthly mean.
  - For genuine one-step-ahead forecasting of *M+1*: *M+1* weather does not exist at origin *M*. Use 1991-2020 climatological normals for that calendar month or a forecast product. **Never forward-fill observed *M+1* weather — that is a leak.**
- **Cold/hot day counts** — derive from the daily APIs (`count(min ≤ 12.0 °C)`, `count(max ≥ 33.0 °C)`) but only over elapsed days.
- **Pink Sheet cotton** — nominal USD, unadjusted. Apply a **1-month publication lag** (`Cotton[M-1]` at origin *M*); log-difference as a regressor.
- **RVD rents/indices** — publish in the first 6 working days of *M+1* → use `rent[M-1]` at origin *M*. **Handle the 2004-2005 hole explicitly**: switch to the quarterly sheet and interpolate (step or linear), or carry `NaN` and let the model drop rows — do **not** silently forward-fill, which manufactures 24 flat months.
- **650-80001 arrivals** — `period` already `YYYYMM`; released ~1 month after month end → use `arrivals[M-1]`. Prefer the **Chinese Mainland** region series and use y/y growth (the level has a 2020 break).
- **Tourism spend (650-80003/80004)** — **annual only, 1998-2025**. Join on calendar year and forward-fill within the year, but only for months at/after the annual release (~Q3 of Y+1); earlier months must carry the *previous* year's value or they leak.
- **HKEXnews operational updates** — quarterly, published **mid-month after quarter end** (Anta Q2-2026 → 2026-07-17; Li Ning Q2-2026 → 2026-07-15). Align y/y to its quarter, forward-fill across the 3 constituent months, and **only from the publication date onward** — Q2 data must not appear in April/May/June rows.
- **Comtrade** — 2-3 month lag and heavy revision → `value[M-2]` at origin *M*; treat 2015+ as the usable range.
- **Lunar New Year** — moves between late Jan and mid Feb. Do not use a Jan/Feb dummy; compute `days_from_LNY` per month and split pre/post terms.

## 5. Honest verdict

- **Biggest genuine win: HKO `monthlyExtract.xml`.** 143 years (1884→2026-09) of *monthly* rainfall, sunshine and temperatures, free, no auth, one 170 KB fetch, **zero gaps across the entire 2004-10→2026-08 window** — no other source found achieves that.
- **Second win: `650-80001` visitor arrivals *is* monthly (2004→2026-07).** The brief's fear was justified for `650-80003`/`650-80004` — those are **annual only** and must be used at annual resolution with an explicit release-lag rule, never as a naive monthly regressor.
- **Pink Sheet cotton is a clean, deep, free monthly series (1960→2025-12, 792 points)** and the only upstream *cost* signal found. But it is a global USD price: expect weak, slow correlation with HK nominal retail *value*. Use as a margin/level proxy, not a demand driver.
- **RVD retail rent is the strongest cost-side monthly series — and it is broken exactly where you need it.** 2004-2005 is absent entirely (`his_data_11` is near-empty through 2008). The "from 1989" label on the RVD page applies to the **annual** sheet, not the monthly one.
- **HKEXnews yields real quarterly operational data for exactly two names.** Giordano, Esprit and Shenzhou disclose semi-annually only, and **Bossini is no longer listed** — so an "HK-listed apparel basket" is two mainland sportswear brands plus four semi-annual reporters. Both are *mainland* demand, weakly coupled to HK retail turnover.
- **Fast Retailing is a dead end for history** — current month only, no free archive. Do not build a monthly UNIQLO feature on it.
- **Comtrade starts too late (~2015) and revises heavily** — a 2015+ cross-check on textile import prices, useless as a 2004-onward regressor.
- **Collinearity warning:** visitor arrivals, tourism spend and HK retail sales are three views of one inbound-demand shock. Keep arrivals (monthly); reduce the annual spend tables to a single per-capita feature.
- **Noise warning:** RVD rent and retail sales are driven by the same retail cycle, and rent is appraisal-based and smoothed — it will look predictive mostly because it is slow. Treat as a regime/level feature, not a driver.
- **Net:** weather (monthly, 1884+) and monthly visitor arrivals are the two sources adding *genuinely different* dimensions across the full window. Everything else is shallow (2015+), annual, gap-ridden, or a re-slice of inbound demand.
