# Data Source Discovery — Branch: Luxury / Jewellery / Watches / Dept Stores / Pharmacy & Cosmetics

Outlet types `32` (jewellery/watches/valuable gifts), `30` (dept stores), `38` (optical), `39` (medicines & cosmetics).
Target: HK monthly 2004-10 → 2026-08 (263 mo). All fetches via `pwsh` `Invoke-WebRequest` (UA `Mozilla/5.0`), 3 s between same-host calls.

## 1. Verified sources table

| source | exact URL | what it gives | freq | covered (from→to) | free | as-known-at-time | how verified |
|---|---|---|---|---|---|---|---|
| **censtatd 650-80001** ⭐ | `https://www.censtatd.gov.hk/api/get.php?id=650-80001&lang=en&full_series=1` | Visitor arrivals by **12 source regions** (Chinese Mainland, Taiwan, Macao, N/SE Asia, Europe, Americas…) | **M+Y** | **2004-01 → 2026-07** (271 M) | yes | revised (not real-time) | 200, 936,705 B, `application/json`; 3516 obs = 293 periods × 12 regions |
| censtatd 650-80003 | `.../api/get.php?id=650-80003&lang=en&full_series=1` | Destination consumption expenditure by region × overnight/same-day | **Y only** | **1998 → 2025** | yes | revised | 200, 177,217 B; `freq` all `"Y"`; 504 = 28 y × 9 regions × 2 types; HK$ m |
| censtatd 650-80004 | `.../api/get.php?id=650-80004&lang=en&full_series=1` | Per-capita spending by region & visit type | **Y only** | **1998 → 2025** | yes | revised | 200, 173,587 B; `freq` all `"Y"`; HK$ |
| censtatd 650-80005 | `.../api/get.php?id=650-80005&lang=en&full_series=1` | Overnight arrivals & **length of stay** | **Y only** | **2006 → 2025** | yes | revised | 200, 17,130 B; 60 obs = 20 y × 3 series; **no region split** |
| censtatd 650-80002 | `.../api/get.php?id=650-80002&lang=en&full_series=1` | Arrivals by mode of transport × region | **Y only** | **2004 → 2024** | yes | revised | 200, 257,895 B; `freq` all `"Y"` |
| **censtatd 340-46001** ⭐ | `.../api/get.php?id=340-46001&lang=en&full_series=1` | HKD per unit FC — **17 currencies incl. CNY, JPY, KRW(WON), GBP, EUR, TWD, SGD** | **M+Y** | **1975-01 → 2026-09** | yes | revised | 200, 2,321,799 B; `sv`=`FC_AUD,…FC_CNY,FC_JPY,FC_WON,…FC_EUR`; last period `202609` |
| censtatd 340-46002 | `.../api/get.php?id=340-46002&lang=en&full_series=1` | HK$ effective exchange rate indices | **M+Y** | **1975-01 → 2026-09** | yes | revised | 200, 388,640 B; 2016 obs |
| censtatd 340-95006 | `.../api/get.php?id=340-95006&lang=en&full_series=1` | HK Gold Exchange **avg daily turnover** | **Q** | **2010 → 2026Q2** | yes | as-published | 200, 33,333 B; HK$ m / US$ m — **turnover, NOT price** |
| **World Bank Pink Sheet** ⭐ | `https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx` | **Monthly gold price**, nominal USD, col `BR`, sheet `Monthly Prices` | **M** | **1960M01 → 2024M12** | yes | as-published | 200, 765,246 B real XLSX; unzipped, parsed `sheet2.xml`: `BR5=Gold`, last `A786=2024M12 BR786=2648.01` |
| **Macau DICJ GGR** ⭐ | `https://www.dicj.gov.mo/web/en/information/DadosEstat_mensal/{YYYY}/report_en.xml` | Monthly gross gaming revenue, **MOP million**, + prior-year & accumulated | **M** | **2010 → 2026-09** | yes | as-published | 200 `text/xml`, ~2.5 KB/file; 2026 dumped: Jan 22,633 … Sep 18,063, Oct–Dec `-`; 2009→404, 2010→200 |
| **HKEXnews search API** ⭐ | `https://www1.hkexnews.hk/search/titleSearchServlet.do?sortDir=0&sortByOptions=DateTime&category=0&market=SEHK&stockId={ID}&documentType=-1&fromDate=YYYYMMDD&toDate=YYYYMMDD&title=&searchType=1&rowRange=1000&lang=en` | **All** filings per stock, JSON, with `FILE_LINK` PDF | continuous | 2007 → 2026-10 | yes | **yes** (`DATE_TIME`) | 200 `application/json`; fields `DATE_TIME,FILE_LINK,FILE_TYPE,TITLE,STOCK_CODE,…`. **Must omit** `t1code/t2Gcode/t2code` (§3) |
| HKEXnews stockId | `https://www1.hkexnews.hk/search/prefix.do?callback=cb&lang=EN&type=A&name={CODE}&market=SEHK` | Code → internal `stockId` | — | — | yes | n/a | 200; `1929→69691`, `0590→1354`, `0178→319`, `0116→1000` |
| **CTF 1929 quarterly** ⭐⭐ | stockId `69691`, e.g. `https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0723/2026072300308.pdf` | **RSV & same-store-sales growth: Chinese Mainland vs Hong Kong & Macao** | **Q** | **2012-07-11 → 2026-07-23**, 78 filings | yes | yes | servlet 632 rows, 78 match "operational data"; PDF 200, 256,420 B `%PDF`; `pdftotext -layout` → 13,359 chars: *"RSV growth +15.1% … Chinese Mainland +10.7% (83.6%) / Hong Kong & Macao +44.8% (16.4%)"*, `same store`×4, `SSSG`×2 |
| **Luk Fook 0590** | stockId `1354` | "RETAIL SALES PERFORMANCE FOR THE Nth QUARTER" + Golden Week | **Q** | **2007-07-10 → 2026-10-07**, 689 rows | yes | yes | servlet 200; Q1 FY2027 filed 16/07/2026 |
| **Sa Sa 0178** | stockId `319` | "Unaudited Sales Update" quarterly **+ Golden Week HK/Macao offline sales** | **Q** | **2007-06-28 → 2026-10-07**, 833 rows | yes | yes | servlet 200; filed 07/10/2026 (Golden Week + Q2) |
| Chow Sang Sang 0116 | stockId `1000` | 694 filings (215 monthly returns); quarterly ops not confirmed | — | 2007-06-28 → 2026-10-09 | yes | yes | servlet 200 |

## 2. Detail

**650-80001** — `figure`=persons; `freq` mixes `Y`/`M` in one payload → filter `freq=="M"`; periods are `YYYYMM` strings. Drop `Total` when using regions (collinearity). This is *the* monthly tourist-volume driver for all four outlet types.

**340-46001** — unit "HKD per unit of FC", so **CNY is HKD per CNY (~1.08)**; a rise = stronger HKD = headwind for mainland shopping. `ln()` it. Covers CNY/JPY/KRW/GBP/EUR/TWD/SGD — everything this branch needs.

**Pink Sheet** — 5 sheets; `Monthly Prices` = 786 rows, dates `YYYYMnn` in col `A`. **This URL is the 3-Jan-2025 edition, ending 2024M12** — 20 months short of 2026-08. World Bank rotates the `related/` path, so locating the current edition is a refresh task, not a blocker; splice 2025–26 from LBMA.

**DICJ** — one `<RECORD>` per month; `<DATA>` cells = month, current-yr, prior-yr, variance, then accumulated trio. Unit **MOP million** (1 HKD = 1.03 MOP). Current-year file fills in monthly. Save 17 files (2010–2026):
```powershell
1..17 | % { $y=2009+$_; (Invoke-WebRequest "https://www.dicj.gov.mo/web/en/information/DadosEstat_mensal/$y/report_en.xml" -UseBasicParsing -Headers @{'User-Agent'='Mozilla/5.0'}).Content }
```

**HKEXnews** — highest-value find for this branch: free JSON, no key, deep history, exact PDF links. Download then `pdftotext -layout` (Poppler **installed** at `…\WinGet\Packages\oschwartz10612.Poppler…\pdftotext.exe`). CTF discloses **HK & Macao separated from Chinese Mainland** — a direct quarterly tourist-sensitive probe on types 32/30, ~3 weeks after quarter-end.

## 3. Rejected / dead ends

| URL | What came back |
|---|---|
| FRED `fredgraph.csv?id=GOLDPMGBD228NLBM` | **FAILED.** IWR → "error while sending the request"; `curl.exe -sv` → TCP+TLS OK (`173.223.87.182:443`), then *"remote party requests renegotiation"* loops to timeout, HTTP:000, 0 B. `Test-NetConnection :443`=True. **UNVERIFIED — schannel/TLS-renegotiation block, not a 404.** |
| `api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/er-eeri-daily` | **405**, body = **Aliyun WAF** error page (`errors.aliyun.com/images/…`). Tried with/without `?offset=0`, `Accept: application/json`, browser UA, `Referer`, root — all 405. **UNVERIFIED — WAF-blocked.** URL+fields *are* confirmed from `apidocs.hkma.gov.hk` (`usd,gbp,jpy,cad,aud,sgd,twd,chf,cny,krw,thb,myr…`). Superseded by 340-46001. |
| `dicj.gov.mo/.../DadosEstat_mensal/2024/index.html` (and `/2024/`) | 200 but **identical size across years (11,816 / 12,037 B)** → JS/XSLT shell. Real data is the `report_en.xml` named inside `statistics_monthly.xsl`. **Textbook "200 but same size" trap.** |
| `dicj.gov.mo/.../2024/dados.xml` / `dados_en.xml` | 404 (wrong filename). |
| `hkexnews.hk/search/titlesearch.xhtml`, `.../advancedsearch/search_active_main.aspx` | 200 but 9,171 / 12,474 B JS shells, no data. |
| HKEXnews servlet **with** `&t1code=-1&t2Gcode=-1&t2code=-1` | 200 but `{"result":"[]","recordCnt":0}` for every stock, even unfiltered. **Removing that triple fixed it** → 632 rows for CTF. |
| `stooq.com/q/d/l/?s=xauusd&i=m` | 200, 796 B, *"requires JavaScript to verify your browser"* → bot challenge. |
| `partnernet.hktb.com/en/research_statistics/index.html` | **403** (retested). HKTB detail is **registration-gated**. |
| `discoverhongkong.com/eng/…`, `hktdc.com/Research` | **403.** |
| `data.stats.gov.cn/english/easyquery.htm?cn=A01` | **403.** |
| `stats.gov.cn/english/Statisticaldata/MonthlyData/` | 200 but **209 B** = meta-refresh to `data.stats.gov.cn/dg/website/page.html#/pc/national/en/monthData` → JS app. NBS monthly retail sales **not machine-readable here**. |
| `richemont.com/investors.html`, `…/investors/results-centre/` | **404.** UNVERIFIED. |
| `swatchgroup.com/en/investors`, `…/en/investor-relations` | Timeout / connection error. UNVERIFIED. |
| `hermes.com/us/en/content/Finance/` | **403.** UNVERIFIED. |
| `lvmh.com/en/investors` | 200 but **2,380,212 B Next.js app** — shell; revenue-by-region only inside results PDFs. |
| `kering.com/en/finance/` | 200, 154,341 B HTML — reachable, but group/quarterly granularity, **not HK-specific**. |
| censtatd `&period_from=202501&period_to=202506` | **Params silently ignored** — full 936,705 B series returned. Slice client-side. |
| `immd.gov.hk/eng/facts.html` | 404. |
| Hainan duty-free | **UNVERIFIED** — no working free endpoint found in time available. |

## 4. Alignment recipe

Join key `YYYY-MM`. All monthly unless stated.

| driver | transform | note |
|---|---|---|
| 650-80001 arrivals | direct, +0…+1 lag | filter `freq=="M"` |
| 340-46001 CNY/JPY/KRW | monthly mean, `ln()` | full 1975→ coverage |
| 340-46002 EERI | direct | HK$ strength control |
| Pink Sheet gold | deflate by HK CPI; 3-mo MA | **ends 2024M12** — splice or flag |
| DICJ GGR | direct, +0 | **starts 2010-01** → 63 of 263 months NaN. Truncate to 2010+ or two-sample design; **never back-cast GGR** |
| CTF / Luk Fook / Sa Sa | quarter → its 3 months | **quarterly, not monthly**; carry `available_at` |
| Golden Week / CNY (Sa Sa, Luk Fook) | event dummies | dated, irregular → good dummies |

**⚠️ The annual tables (650-80002/80003/80004/80005) cannot be monthly regressors without interpolation, and that interpolation is a look-ahead risk** — the 2025 annual figure is only known after 2025 ends, so spreading it over Jan–Dec 2025 leaks the future. If used: lag ≥1 year, or restrict to an annual auxiliary model, or drop it in favour of monthly 650-80001, which measures the same thing at the right frequency. Same warning for forward-filling the Pink Sheet tail.

**⚠️ Point-in-time.** HKEXnews and DICJ figures are as-published and revised; use the servlet's `DATE_TIME` as `available_at` (CTF ~3 weeks after quarter-end, Sa Sa/Luk Fook ~2–4 weeks). Build `(period, value, available_at)`, not a month-indexed vector. Conversely, 650-80001/340-46001 are **latest-revision snapshots** — fine for backtesting on final data, but they **would leak** in a genuine real-time forecast.

## 5. Honest verdict

- **The best new driver is HKEXnews, not a statistics agency.** `titleSearchServlet.do` is free, unauthenticated, deep-history JSON; CTF/Luk Fook/Sa Sa give quarterly HK-and-Macao operational sales that **no censtatd table contains** — a genuinely different dimension from another retail-turnover slice.
- **Two monthly drivers fully cover the window with zero interpolation:** 650-80001 (12 source regions, 2004-01→2026-07) and 340-46001 (CNY/JPY/KRW, 1975→2026-09).
- **650-80003/80004/80005 are ANNUAL** — as suspected. 650-80005 is **not** a source-market table: it is overnight arrivals + length of stay, annual, 2006–2025. Read the interpolation warning before touching any of them.
- **DICJ GGR is a valid ex-HK proxy but starts 2010-01**, costing 63/263 months. Genuinely different (Macau reacts to the same mainland-visa/tourist shocks) — accept a shorter sample or a two-period design.
- **Gold belongs to the World Bank Pink Sheet** (verified by unzipping and reading `BR5=Gold`). Its 2025–26 tail needs a splice; censtatd 340-95006 is **turnover**, a different quantity — do not substitute it for price.
- **FRED and HKMA are unreachable from this host** (TLS renegotiation loop; Aliyun WAF 405) — verified as network blocks, not bad URLs. Neither is a real loss.
- **HKTB PartnerNet is registration-gated (403)**; free source-market detail is not obtainable here, and 650-80001 is the free monthly substitute.
- **China NBS, LVMH, Richemont, Swatch, Hermès are dead ends** (403/404/JS shell/timeout); Hainan duty-free UNVERIFIED.
- **"HKEX monthly returns are share capital" is CONFIRMED empirically**: CTF's 178 monthly returns are titled *"Monthly Return of Equity Issuer on Movements in Securities"* — no sales content (Sa Sa 218, Chow Sang Sang 215). Do not use them as sales.
- **Recommended feature set:** monthly arrivals by region + HKD/CNY + gold (spliced) + Macau GGR (2010+), plus quarterly CTF/Luk Fook/Sa Sa RSV & SSS as *point-in-time* regressors, plus dated Golden Week / IVS-policy event dummies.
