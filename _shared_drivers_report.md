# Shared Exogenous Driver Layer — Hong Kong Retail Sales Forecasting
**Scope:** macro / tourism / transport / weather / calendar / news drivers usable as monthly regressors for a 263-month panel (2004-10 → 2026-08).
**Verification:** every URL below was fetched with `Invoke-WebRequest -Uri <url> -UseBasicParsing -TimeoutSec 45 -Headers @{ 'User-Agent'='Mozilla/5.0' }`. Raw status / Content-Type / byte length recorded. Fetch date **2026-10-13**.

**Three structural findings that reframe the whole layer:**

1. **The censtatd Monthly Digest and the TD Digest are both ROLLING-WINDOW vintages, not archives.** Confirmed twice, independently. Digest B1010002 T1008 (control-point passengers) holds only 2023-01→2026-07 in the current issue; its XLSX `sharedStrings.xml` contains the years **2025 and 2026 only**. TD `table31a.xls` likewise carries ~5 annual + ~18 monthly rows. Neither is a download — full history means ~264 monthly scrapes. Budget accordingly; do not promise depth from these.
2. **Soft-404 traps exist.** `hongkongairport.com/.../facts-figures/YYYYMM_statistics_e.pdf` returns **HTTP 200 for every month probed (200401…202606), each exactly 28279 bytes**, body title "Page Not Found". Any fetcher keying on `status == 200` silently ingests a 404 for all 263 months. Real pattern is `MonYYYY_statistics_e.pdf`.
3. **HKMA's API is behind an Aliyun anti-bot WAF.** One request succeeded (card survey, 200, 18984 B); every subsequent request returned **405** with a `Server: Tengine` / `acw_tc` cookie / `errors.aliyun.com` block page — persisting through 90/120/150 s backoffs and cookie-session replays. Treat HKMA as unreliable for a batch pipeline.

---

## 1. Verified sources

Legend for **ts?** (timestamp): `RT` = genuinely as-known-at-the-time; `CV` = current vintage (revised); `SNAP` = snapshot of today's crawl, not point-in-time.

### 1a. Macro

| source | exact URL | what it gives | freq | months covered | free | ts? | how verified |
|---|---|---|---|---|---|---|---|
| GDP + deflator + per-capita | `censtatd.gov.hk/api/get.php?id=310-31001&lang=en&full_series=1` | GDP panel | Q + Y | 1973Q3→2026Q2 (214 Q) | Y | CV | 200, application/json, n=214 Q periods |
| GDP by expenditure component | `…?id=310-31002` (current), `310-31003` (chained 2024$) | GDP components | Q + Y | 1973Q3→2026Q2 | Y | CV | 200, JSON, freq='Q' n=214 |
| Private consumption expenditure | `…?id=310-31011` / `310-31012` | PCE by component | Q + Y | 1966→2026Q2 (274 per) | Y | CV | 200, 1.49 MB, 4932 recs |
| **CXDM** | `…?id=310-31013` / `310-31014` | Consumption expenditure in domestic market | **ANNUAL ONLY** | 1966→2025 (60) | Y | CV | 200, 542 KB, **freq='Y'** — not quarterly |
| CPI (composite) | `…?id=510-60001` | CPI, m/m, y/y | M | 1974-07→2026-08 (626) | Y | CV | 200, 1.63 MB, 7920 recs |
| CPI by COICOP division | `…?id=510-60003` | CPI by division (food, clothing, durables…) | M | **2005-01**→2026-08 (260) | Y | CV | 200, 31472 recs; **starts 3 months late** |
| Unemployment / underemployment | `…?id=210-06101`, `210-06103`, `210-06401` | unemployment & underemployment rate | **M3M** | 1985-03→2026-08 (498) | Y | CV | 200 ×3; **freq='M3M' on all three — no plain monthly exists** |
| HKD interest rates | `…?id=340-45021` | HKD interest rates | M | 1980-01→2026-08 (560) | Y | CV | 200, JSON |
| Exchange rates | `…?id=340-46001` | HKD per unit FX | M | 1975-01→2026-09 (621) | Y | CV | 200, 10280 recs |
| Effective exchange rate index | `…?id=340-46002` | EERI (2020=100) | M | 1975-01→2026-09 (621) | Y | CV | 200, JSON |
| Money supply | `…?id=340-45011` / `340-45012` | money supply | **Q only** | 1997Q1→2026Q2 (117) | Y | CV | 200; freq='Q'/'Y' — no monthly |
| HK interbank market | `…?id=340-95001` | interbank aggregate | **Q only** | 2010Q1→2026Q2 (66) | Y | CV | 200; freq='Q'/'Y' |
| HKEX market statistics | `…?id=340-95003` | listed cos, market cap, ADT | **Q only** | 2010→2026Q2 (82) | Y | CV | 200; freq='Q' — **not an index** |
| Hang Seng Index | `query1.finance.yahoo.com/v8/finance/chart/%5EHSI?interval=1mo&range=25y` | monthly OHLC | M | 2001-10→2026-10 (301) | Y | CV | 200, JSON 30772 B, **0 null closes**; unofficial |
| Business receipts indices | `…?id=660-69001` (+`69011` SA) | service-industry receipts index | Q | 2005→2026Q2 (107) | Y | CV | 200, 2568 recs |
| SME diffusion indices | `…?id=330-40001` | receipts/new orders/employment diffusion | M | **2011-06**→2026-09 (184) | Y | CV | 200, 1.58 MB; short coverage confirmed |
| HKMA credit-card lending | `api.hkma.gov.hk/public/…/banking/credit-card-lending-survey` | card receivables, charge-offs | **QUARTERLY** | ~100 quarters (to 2026-Q2) | Y | CV | **200 once** (18984 B, `end_of_quarter:"2026-Q2"`, datasize=100); then WAF 405. **Confirmed quarterly — belief upheld** |

### 1b. Tourism & cross-boundary

| source | exact URL | what it gives | freq | months covered | free | ts? | how verified |
|---|---|---|---|---|---|---|---|
| Visitor arrivals by nationality | `…?id=650-80001` | arrivals by region/nationality | M | 2004-01→2026-07 (271) | Y | CV | 200, 936705 B, 3516 recs |
| Visitor arrivals by mode | `…?id=650-80002` | arrivals by transport mode | **ANNUAL** | 2004→2024 (21) | Y | CV | 200, 756 recs; freq='Y' |
| ImmD daily pax by control point | `immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv` | daily arrivals/departures by CP × pax class | D | **2021-01-01→2026-10-12** (70 months) | Y | SNAP | 200, text/csv, 3 368 213 B, 62 530 rows, 18 control points |
| ImmD festival archive | `…/statistics_passenger_traffic_festival_periods_archive.csv` | festival-period pax | D | 2019-04-04→2020-12-31 | Y | SNAP | 200, 589 506 B, 12 048 lines |
| Monthly Digest control-point pax | `censtatd.gov.hk/…/B1010002/att/B10100022026MM06B.zip` → part B10, tables T1008/T1009/T1010 | arrivals / departures / resident departures by control point | M | **2023-01→2026-07** (17 CPs) | Y | CV | 200, 309 780 B zip → 4×2 CSVs; T1008 n=306 rows, years 2023-2026 only |
| TD Table 8.2 cross-boundary pax | `td.gov.hk/datagovhk_td/mttd/resources/table82.xls` | pax arrivals/departures by control point | M | rolling vintage (~2020-2023) | Y | CV | 200, application/vnd.ms-excel, 72 192 B, OLE2 magic |
| HKIA monthly | `hongkongairport.com/iwov-resources/file/the-airport/hkia-at-a-glance/facts-figures/MonYYYY_statistics_e.pdf` | pax/cargo | M | **Oct 2016**→2026-08 | Y | CV | 200, 99 292 B, `%PDF-` (see soft-404 trap, §3) |
| HKTB PartnerNet | `partnernet.hktb.com/en/research_statistics/index.html` | arrivals detail | — | — | **gated** | — | **403**, title "Just a moment…" — Cloudflare challenge on every path |

### 1c. Transport / footfall

| source | exact URL | what it gives | freq | months covered | free | ts? | how verified |
|---|---|---|---|---|---|---|---|
| **MTR patronage** | `POST mtr.com.hk/en/corporate/investor/pf_export.php` | patronage by service line | M | **2000-01→2026-08 (320, zero gaps)** | Y | RT | 200, 25 744 B, `Content-Disposition: Patronage_20261013.csv` |
| TD Digest (HTML era) | `td.gov.hk/en/transport_in_hong_kong/transport_figures/monthly_traffic_and_transport_digest/YYYY/YYYYMM/index.html` | operator journeys, tunnels | M | 2018-01→2026-07 | Y | CV | 200, 34 562 B (202607) |
| TD Digest (PDF era) | `td.gov.hk/filemanager/en/content_<ID>/YYMM.pdf` | same | M | 2003-01→2017-12 | Y | CV | 200, `%PDF-`; **`<ID>` irregular — must scrape** |
| TD per-table XLS | `td.gov.hk/datagovhk_td/mttd/resources/table{11,21,21s,24,31a-d,32a-t}.xls` | rolling window tables | M | ~5 annual + ~18 monthly rows | Y | CV | 200, OLE2 BIFF8, **89 resources via CKAN** |

### 1d. Weather

| source | exact URL | what it gives | freq | covered | free | ts? | how verified |
|---|---|---|---|---|---|---|---|
| Daily extract (best) | `hko.gov.hk/cis/dailyExtract/dailyExtract_YYYY.xml` | 12 daily fields incl. rainfall, sunshine, RH, cloud | D | **1884-03-01**→2026-09 | Y | RT | 200; 1884=31 481 B, 2004=32 100 B, 2026=24 333 B |
| Bulk daily CSV | `data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_<CODE>_ALL.csv` | daily by element (RF, TEMP, MAXT, MINT, RH, CLD, MSLP, DEW…) | D | RF 1884-03-01; **MAXT/MINT 1884-01-01** | Y | RT | 200, RF=810 983 B, 49 492 rows, ends exactly **2026-08-31** |
| Monthly extract | `hko.gov.hk/cis/monthlyExtract/monthlyExtract.xml` | 13 monthly columns | M | 1884→2026-09 | Y | RT | 200, 169 804 B |
| Open Data API | `data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP&rformat=csv&year=YYYY&station=HKO` | monthly climate | M | 1884-03-01→ | Y | RT | 200, text/csv ~6.4 kB/yr; **only 9 dataTypes exist** |
| Tropical cyclone warnings | `hko.gov.hk/dps/wxinfo/climat/warndb/tc.dat` | signal + start/end to the minute | event | **1946-07-16→2026-09-13** | Y | RT | 200, 136 484 B, 2514 lines, 16 fields |
| Rainstorm warnings | `…/warndb/rstorm.dat` | Amber/Red/Black with start/end times | event | **1998-04-12→2026-10-04** | Y | RT | 200, 37 032 B, 977 rows, 13 fields |

### 1e. Calendar / events

| source | exact URL | what it gives | freq | covered | free | ts? | how verified |
|---|---|---|---|---|---|---|---|
| Lunar New Year dates | `hko.gov.hk/en/gts/time/calendar/pdf/files/YYYYe.pdf` | LNY day-1 date | Annual | 2004→2026, **23/23** | Y | RT | 23 × 200 with text layers; cross-checked vs GovHK gazette |
| GovHK press releases | `info.gov.hk/gia/general/YYYYMM/DD.htm` | daily release titles/links | D | **2000-10**→2026-10 | Y | **RT** | 200; 200010/03.htm = 3948 B, 27 links, real content |
| Public holidays | `gov.hk/en/about/abouthk/holiday/YYYY.htm` | general holidays | Annual | **2022→2027 only** | Y | RT | 200 for 2022+; 2004-2021 all **404** |
| Trade days | computable | see §4 | M | any | Y | — | rule-based |

### 1f. News sentiment / uncertainty

| source | exact URL | what it gives | freq | covered | free | ts? | how verified |
|---|---|---|---|---|---|---|---|
| EPU Hong Kong | `policyuncertainty.com/media/HK_EPU_Data_Annotated.xlsx` | Monthly_HKEPU_Index + 12 event labels | M | 1998-04→**2025-12** (333, contiguous) | Y | CV | 200, 204 907 B, `50 4B 03 04` |
| GEPU + 21 countries | `policyuncertainty.com/media/All_Country_Data.xlsx` | GEPU + China/Korea/Singapore/Japan | M | **1985-01→2026-07** (499) | Y | CV | 200, 130 388 B — **fresher spine than the HK file** |
| Google News RSS | `news.google.com/rss/search?q=<query>+after:YYYY-MM-DD+before:YYYY-MM-DD&hl=en-US&gl=US&ceid=US:en` | article counts | D→M | 0 items pre-2006; **usable ~2012→** | Y | SNAP | 200 on all 15 probes; item counts in §3 |

---

## 2. Full detail per source

**censtatd JSON API.** All tables share one shape: `header.title`, `header.status`, and a flat `dataSet` array of records with keys `freq`, `period`, `sv`, `svDesc`, `figure`, `sd_value`, plus one dimension pair (`OUTLET_TYPE`/`…Desc`, `REGION`, `COICOP`, `AGE`, `IND`, …). `period` is `YYYYMM` (monthly), `YYYYMM` where MM∈{03,06,09,12} (quarterly), or `YYYY` (annual). **`figure` can be the string `'***'`** — coerce before arithmetic.

```powershell
$j = (Invoke-WebRequest "https://www.censtatd.gov.hk/api/get.php?id=510-60001&lang=en&full_series=1" -UseBasicParsing).Content | ConvertFrom-Json
$j.dataSet | Where-Object { $_.freq -eq 'M' -and $_.svDesc -eq 'Index' }
```

Table discovery: `https://www.censtatd.gov.hk/data/en/all_web_tables.json` → 200, 147 110 B, **590 tables** with code + title. Download it and grep locally; it is the single best index for this project. Enumerated negative: it contains **no cross-boundary, vehicle, tunnel, traffic, ferry or retail-ridership table at all**.

**data.gov.hk CKAN API — highest-leverage discovery of this branch.** Undocumented but live:
`https://data.gov.hk/en-data/api/3/action/package_search?q=<kw>&rows=N` and `…/package_show?id=<dataset-id>`. Both returned 200 JSON. Example: `package_show?id=hk-td-tis_10-monthly-traffic-and-transport-digest` returned **89 direct `.xls` resource URLs** (keystat, table11, table21/21s, table22-24, table31a-d, table32a-t, charts). This turns "hunt for URLs" into a one-call enumeration. `package_show?id=hk-censtatd-tablechart-b1010002` gave the live Digest vintages.

**Monthly Digest vintage pattern:** `censtatd.gov.hk/en/data/stat_report/product/B1010002/att/B1010002{YYYY}MM{MM}B0{1..11}.xlsx` (also `…B.zip` for CSVs, and `…B0{1,2}` for the Excel tables split). Latest live vintage verified: **2026MM06** (2 767 129 B). 2026MM07/08/09 → 404.

**HKMA.** Base `https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/<group>/<slug>`. Group→slug mapping extracted from the docs pages: `er-ir` → hk-interbank-ir-daily / -endperiod / -periodaverage, hkd-ir-periodaverage, hkd-fer-*, er-eeri-*, renminbi-dr; `banking` → credit-card-lending-survey, residential-mortgage-survey, balance-sheet-*, loans-by-*; `money` → currency, supply-components-*, supply-adjusted. Documented call form includes a `segment` parameter, e.g. `…/er-ir/hk-interbank-ir-periodaverage?segment=hibor.fixing&offset=0`. **Every one of these returned 405 (WAF) after the first success — UNVERIFIED.** Substitute route that works: censtatd **340-45021** (HKD interest rates, monthly 1980-01→) and **340-46001** (exchange rates, monthly 1975-01→).

**HKO warnings → monthly counts.** `tc.dat` fields: `[tcID, intensity, name, signal, dir, startHHMM, startDD, startMM, startYYYY, startFlag, endHHMM, endDD, endMM, endYYYY, endFlag, durationHHMM]`. Rows with `intensity == 'MSN'` are the Strong Monsoon Signal, not a tropical cyclone — filter them. Trailer `UUUU` marks provisional rows. `rstorm.dat`: `[type, sYYYY,sMM,sDD,sHH,sMin, eYYYY,eMM,eDD,eHH,eMin, durHH,durMin]`, type ∈ {A, R, B}.

---

## 3. Rejected / dead ends

| URL / pattern | what came back |
|---|---|
| `censtatd.gov.hk/en/web_table.html?id=…` | 200, 18 547 B **JS shell** — use `/api/get.php` |
| `censtatd.gov.hk/en/EIndexbySubject.html?pcode=B1010002&scode=530` | 200, 16 516 B, JS shell; **zero** `B1010002`/`.pdf`/`.xlsx` links in raw HTML |
| `censtatd.gov.hk/en/data/stat_report/product/B1010002/att/B10100022026MM00B0100.pdf` | **404** (wrong vintage pattern) |
| `censtatd.gov.hk/data/en/stat_report/report_index.json` | **404** |
| `censtatd.gov.hk/api/post.php` | live, returns structured JSON errors, but the query schema was not cracked — **UNRESOLVED, not dead** |
| TD `…/free_publications/monthly_traffic_and_transport_digest/index.html` | **404**; real index is under `transport_figures/` |
| HKIA `…/facts-figures/YYYYMM_statistics_e.pdf` | **200 but soft-404** — 28 279 B HTML for ALL 10 months probed. Real pattern `MonYYYY_statistics_e.pdf`; `MonYYYYe.pdf` back to Oct 2016 (Sep2016 and earlier = soft-404). `/api/statistics` → same soft-404. Annual `YYYYe.pdf` 1998→2026. **PDF-only, no JSON/CSV** |
| `partnernet.hktb.com/*` | **403** Cloudflare "Just a moment…" on every path — registration-gated *and* bot-protected |
| `www.immd.gov.hk/eng/facts.html`, `/eng/statistics.html`, `/eng/message_from_us/statistics.html` | **404** (Chinese 404 page, 9018 B) |
| `www.tourism.gov.hk/english/statistics/statistics.html` | **404** |
| `www.hkex.com.hk/eng/stat/smstat/mthbull/rpt_monthly.htm` | **404** |
| `www.hsi.com.hk/eng/indexes/all-indexes/hsi` | 200 but 6738 B app shell, no data |
| data.gov.hk dataset pages via `api.data.gov.hk/v1/3/action/package_show` | **403**; use `data.gov.hk/en-data/api/…` instead |
| **GDELT REST Doc API** `api.gdeltproject.org/api/v2/doc/doc?…` | **HTTP 429 on all 3 attempts** (immediate, after 60 s, after 120 s) — reproduces the 6 prior failures. **Honest result: dead from this host** |
| GDELT bulk `data.gdeltproject.org/gdeltv2/*.export.CSV.zip` | 200 **without a key** (`lastupdate.txt` 200/319 B live). But coverage **starts 2015-02-19** (2015-02-17/18 → 404). Covering 2015-02→2026-08 ≈ **368 000 zip downloads** |
| GDELT v1 `data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip` | **404** for 2004-10-15, 2006-01-01, 2013-01-01 |
| GDELT BigQuery `gdelt-bq.gdeltv2.events` | **401 unauthenticated — needs billing** |
| HKO `en/wxinfo/currwx/tc_review.htm`, `en/informtc/tcsignal.htm`, `en/cis/monsum.htm`, `cgi-bin/hko/daily_extract.pl`, `daily_HKO_{SUN,GRS,EVAP,WIND}_ALL.csv` | **404** (all) |
| HKO `weather.php?dataType=warnsum\|warningInfo` | 200 but **2-byte** body `{}` — current-only, no history |
| `dataType=CLMRAIN` / `CLMSUN` | 534-byte "Please include valid parameters" page |

**Google News RSS stability (query `Hong Kong retail sales`).** Empty is a *valid live feed*, not a shell — 1209 B bodies carry `<lastBuildDate>Tue, 13 Oct 2026…` and zero `<item>`.

| month | status | bytes | items |
|---|---|---|---|
| 2004-10 | 200 | 1209 | **0** |
| 2005-01 | 200 | 1209 | **0** |
| 2005-06 / 2005-12 | 200 | 3474 / 2276 | 2 / 1 |
| 2007-01 / 2008-01 | 200 | 8168 / 5825 | 5 / 4 |
| 2009-01 | 200 | 1209 | **0** |
| 2011-01 | 200 | 16 829 | 14 |
| 2012-06 | 200 | 29 685 | 22 |
| 2013-01 | 200 | 49 199 | 37 |
| 2016-03 / 2020-02 / 2024-01 | 200 | ~60-69 k | 45 / 49 / 50 |

Non-monotonic (2009-01 = 0 while 2007-01 = 5): **there is no clean cutoff year.** Counts also saturate at a **~50 cap**, and `before:` is not strictly exclusive (three probes leaked one article into the following month).

---

## 4. Alignment recipe

**Monthly → direct join.** CPI (510-60001/60003), visitor arrivals (650-80001), MTR patronage, exchange rates, EERI, HKD interest rates, SME diffusion, EPU, HKO monthly extract, warning counts.

**Quarterly → never interpolate for a forecasting model.** Interpolating 2026Q2 into Apr/May/Jun requires the *whole* quarter, i.e. data through 30 June, to fill April — a 2-month look-ahead for an April-target model. Use instead, at month *t*: `last_completed_quarter(t-1)` (the most recent quarter fully ended before *t*), plus optionally a **rolling 4-quarter sum** and a quarter-index dummy for seasonality. Applicable to GDP (310-31001/2/3), PCE (310-31011/12), GCE (310-32021), business receipts (660-69001/11), money supply (340-45011), interbank (340-95001), HKEX (340-95003), HKMA card survey.

**Annual → cannot be a monthly regressor.** CXDM (310-31013/14, 310-31015) is annual through 2025 only. Any month-level use requires interpolation or forward-filling the *previous* year — the latter is look-ahead-free only if you take year *Y-1* to explain year *Y*, which then cannot distinguish months. **Verdict: use CXDM for a fixed-effects/panel-level control, or drop it.** Visitor arrivals by mode (650-80002) is annual only; use 650-80001 instead.

**M3M unemployment.** The published value is a 3-month moving average ending in the stated month: `u_t = mean(u*_{t-2}, u*_{t-1}, u*_t)`. It is look-ahead-free at origin *t* but contains **no new information beyond the underlying monthly series**, and it is serially correlated by construction (MA(2)). Do not difference it and expect white noise. It is published ~3-4 weeks after month end, so at a strict origin it may not yet be available — lag by 1 month.

**Weather: daily → monthly without leakage.** Let the forecast origin be month *M* (target 2026-08). For any feature month *m*:
1. Restrict every daily frame to `date <= last day of m`, and **assert** `max(date) <= origin_end` — do not rely on the file happening to end there.
2. Aggregate with calendar-day denominators: `rainfall_total = Σ`, `mean_temp = mean over days present`, `sunshine_total = Σ`. Counts of hot/cold days use HKO thresholds on daily min/max (≤12.0, ≥33.0, ≥28.0 °C).
3. **Warning intervals must be expanded, not summed.** `tc.dat`/`rstorm.dat` give start *and* end; expand each record to the set of dates it covers, then count **distinct dates** per (year, month). Drop any record whose **start** is after the origin.
4. Lag all weather features by ≥1 month relative to the target.

**Critical live trap (act on this).** Today is 2026-10-13, and `dailyExtract_2026.xml` **already contains a complete September 2026**; `rstorm.dat` has rows dated 2026-10-02/03/04; the last `tc.dat` record starts 2026-08-31 and ends 2026-09-04. Counting its September portion leaks one month past the 2026-08 origin. Hard-truncate at `2026-08-31`.

**Calendar.** `trading_days = days_in_month − weekend_days − general_holidays_falling_on_weekdays` (Saturday is not a trading day in HK retail). LNY day-1 is the strongest single calendar feature: **9 of 23 years fall in January, 14 in February** — encode as a month-level dummy plus days-to-LNY.

---

## 5. Coverage matrix

| driver | covers 2004-10→2026-08 fully? | if not, from when | verdict |
|---|---|---|---|
| CPI composite (510-60001) | **Yes** (1974-07→) | — | monthly regressor |
| CPI COICOP division (510-60003) | No | 2005-01 | monthly regressor (−3 months) |
| HKD interest rates (340-45021) | **Yes** (1980-01→2026-08) | — | monthly regressor |
| Exchange rates (340-46001) | **Yes** (1975-01→) | — | monthly regressor |
| EERI (340-46002) | **Yes** (1975-01→) | — | monthly regressor |
| Visitor arrivals (650-80001) | Yes (2004-01→2026-07) | tail month missing | monthly regressor |
| MTR patronage | **Yes** (2000-01→2026-08, zero gaps) | — | monthly regressor |
| HKO daily/monthly (all variables) | **Yes** (1884→) | — | monthly regressor |
| TC signal / rainstorm day counts | **Yes** (1946 / 1998→) | — | monthly regressor |
| Lunar New Year | **Yes** (23/23 verified) | — | monthly regressor |
| GovHK press releases | **Yes** (2000-10→) | — | monthly regressor (counts) |
| Trading days | **Yes** (rule) | — | monthly regressor |
| EPU Hong Kong | No | 1998-04→**2025-12** | monthly regressor, 8-month NaN tail |
| GEPU / All_Country (incl. China) | Yes (1985-01→2026-07) | tail month | monthly regressor |
| Unemployment (210-06101, M3M) | Yes (1985-03→) | — | monthly regressor (MA(2)-smoothed) |
| SME diffusion (330-40001) | No | **2011-06** | ablation only / post-2011 model |
| GDP, PCE, GCE (quarterly) | Yes (1973Q3→2026Q2) | — | monthly regressor via last-completed-quarter |
| Business receipts (660-69001) | Yes (2005→) | — | monthly regressor via last-completed-quarter |
| Money supply, HKEX, HKMA card | Yes (1997/2010/2001→) | — | quarterly-only regressor |
| Google News RSS sentiment | **No** | **~2012** | ablation only / post-2012 model |
| ImmD daily control-point pax | **No** | **2021-01-01** | ablation only |
| Monthly Digest control-point pax | **No** | **2023-01** | unusable as a full-window regressor |
| TD Digest (operator/tunnel) | Partial | 2003-01, but **264 scrapes** | usable at high cost; rolling window |
| HKIA monthly | **No** | **Oct 2016** | ablation only; PDF parsing |
| CXDM (310-31013/14) | **No** | annual only, →2025 | not a monthly regressor |
| Public holidays per-year pages | **No** | 2022 (earlier 404) | use computable rule |
| HKMA API (HIBOR, MSB series) | — | — | **UNVERIFIED** (WAF 405) — use censtatd 340-45xxx |
| GDELT (all routes) | **No** | 2015-02-19 bulk only | **unusable** |

---

## 6. Honest verdict

- **Ten drivers genuinely span the full 263 months:** CPI composite, HKD interest rates, exchange rates, EERI, visitor arrivals (650-80001, minus the tail month), **MTR patronage**, the whole HKO weather family, TC/rainstorm day counts, Lunar New Year, and GovHK press-release counts. These alone support a full-window model.
- **MTR patronage is the standout find.** One POST returns a clean 25 KB CSV, 320 consecutive months (2000-01→2026-08), zero gaps — the only transport series that covers the window, and the brief's premise that it was an unreadable JS shell was wrong twice over.
- **The brief's HKIA URL is a soft-404 trap and its "verified working for May2026" does not hold.** All ten months probed returned 200 with identical 28 279-byte HTML error pages. Real monthly coverage is PDF-only and starts **Oct 2016**.
- **CXDM is annual, not quarterly.** The brief's expectation that 310-31013/14 is the closest demand aggregate is right in spirit, but `freq='Y'` makes it unusable as a monthly regressor. PCE (310-31011/12) *is* quarterly and is the practical substitute.
- **The HKMA card survey is confirmed QUARTERLY** — the belief is upheld (one clean 200 response, `end_of_quarter: "2026-Q2"`, 100 quarters). But **every other HKMA endpoint is UNVERIFIED**: the API sits behind an Aliyun WAF that returned 405 across ~10 attempts, three backoff levels and cookie-session replays. Route around it with censtatd 340-45021/340-46001, which are monthly and go back to 1980/1975.
- **No censtatd series gives plain monthly unemployment** — 210-06101/06103/06401 are all `M3M`, a 3-month moving average, 1985-03→2026-08. Model it as MA(2)-smoothed and lag it a month for publication delay.
- **Cross-boundary passenger data cannot cover the window from any single source.** ImmD daily starts 2021-01; the Digest's control-point table starts 2023-01; TD Table 8.2 is a rolling vintage. Use **650-80001 visitor arrivals** (2004-01→2026-07) as the tourism spine and treat control-point detail as a 2021+ ablation.
- **Weather is genuinely deep and genuinely free.** HKO daily data reaches **1884-03-01**, with rainfall/sunshine/RH in one XML file, and `tc.dat`/`rstorm.dat` give exact per-month typhoon-signal-day and rainstorm-day counts back to 1946 and 1998. Cold-day and very-hot-day counts are *not* published anywhere and must be derived from MAXT/MINT.
- **A live leakage trap exists right now.** With the origin at 2026-08, `dailyExtract_2026.xml` already holds a full September 2026, and both warning files contain post-origin records. Truncate at 2026-08-31 and drop warning records starting after the origin.
- **News sentiment cannot cover the window.** Google News RSS returns **0 items** for 2004-10 and 2005-01, is non-monotonic through 2011, saturates at ~50 items, and leaks one day via `before:`. Build sentiment for **2012+** only; cover 2004-2011 with EPU plus calendar features.
- **GDELT is a dead end, stated plainly.** The REST Doc API returned **HTTP 429 on all three attempts**, including after 60 s and 120 s waits — reproducing the operator's six failures. The bulk 15-minute CSVs *do* work without a key but start **2015-02-19** and would need ~368 000 downloads; BigQuery returns **401** without billing. Reject it.
- **Two systemic traps worth encoding as fetcher guards.** (a) Reject responses whose byte length equals a known soft-404 length (HKIA = 28 279) regardless of status code. (b) Treat any "Digest"-style table as a **rolling window until proven otherwise** — both the censtatd and TD digests silently cap at ~2-4 years. The **data.gov.hk CKAN API** (`data.gov.hk/en-data/api/3/action/package_search` / `package_show`) is the single best tool for discovering these resources programmatically and is worth wiring into the pipeline before any further manual URL hunting.
