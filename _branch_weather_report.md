# HKO Weather Drivers — Verified Source Report

Branch: DATA SOURCE DISCOVERY / WEATHER. Every URL was fetched on 2026-10-13 with
`Invoke-WebRequest -Uri <url> -UseBasicParsing -TimeoutSec 45 -Headers @{ 'User-Agent'='Mozilla/5.0' }`, 3 s spacing between same-host calls.

## 1. Verified sources

| Source | Exact URL | Gives | Freq | Coverage | Free | Evidence |
|---|---|---|---|---|---|---|
| **Daily Extract (BEST)** | `https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_YYYY.xml` | 12 daily elements incl. rainfall, sunshine, RH, cloud | daily | **1884-01-01 → 2026-09-30** | yes | 200; LEN 31481 (1884) / 32100 (2004) / 24333 (2026); 366 & 273 day-rows |
| **Bulk CSV (HKO)** | `https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_{RF,RH,TEMP,MAXT,MINT,CLD,MSLP,DEW,GMT,WET}_ALL.csv` | one element, all years, 5-col CSV | daily | RF 1884-03-01; MAXT/MINT 1884-01-01; RH 1947-01-01; CLD 1949-01-01; DEW 1961-01-01 — **all end 2026-08-31** | yes | RF 200, LEN 810983, 49492 rows; MAXT 840284 / 49552 |
| Open Data API | `…/weatherAPI/opendata/opendata.php?dataType={CLMTEMP,CLMMAXT,CLMMINT}&rformat=csv&year=YYYY&station=HKO` | daily temp, 5-col CSV | daily | 1884 → 2026-08 | yes | 200 `text/csv`, ~6.4 kB/yr |
| Monthly Extract | `https://www.hko.gov.hk/cis/monthlyExtract/monthlyExtract.xml` | 13 monthly elements × all years | monthly | **1884 → 2026-09** | yes | 200, LEN 169804 |
| Monthly per-element | `https://www.hko.gov.hk/cis/individual_month/monthlyElement.xml` | per-element × year × 12 months | monthly | 1884 → | yes | 200, LEN 193107 |
| Daily per-element | `https://www.hko.gov.hk/cis/individual_day/daily_YYYY.xml` | all elements per day | daily | 1884 → 2026 (Oct-Dec blank) | yes | 200; LEN 53471 / 66774 / 67067 |
| Coverage config | `https://www.hko.gov.hk/cis/hko.xml` | startYear/endYear for 24 elements | static | — | yes | 200, LEN 1308 |
| **TC signal history** | `https://www.hko.gov.hk/dps/wxinfo/climat/warndb/tc.dat` | TSV 16 fields: signal, name, start/end date+time | event | **1946-07-16 → 2026-09-13** | yes | 200, LEN 136484, 2514 lines |
| **Rainstorm history** | `https://www.hko.gov.hk/dps/wxinfo/climat/warndb/rstorm.dat` | TSV 13 fields: A/R/B, start/end date+time, duration | event | **1998-04-12 → 2026-10-04** | yes | 200, LEN 37032, 977 rows |
| Thunderstorm warnings | `…/warndb/thunder.dat` | TSV start/end | event | 1967-04-03 → | yes | 200, LEN 205929, 5843 lines |
| Other warnings | same dir: `lslip` `fire` `cold` `hot` `frost` (1991→) `nf` `.dat` | Landslip/Fire/Cold/Very Hot/Frost/NT-flooding | event | varies | yes | paths from warndb{4,6,12,13,14,11}.shtml |
| Warning DB index | `https://www.hko.gov.hk/en/cis/warndb.htm` | links to all 11 warning types | static | — | yes | 200, LEN 7224 |
| Monthly Weather Summary | `…/pastwx/mwsYYYY/mwsYYYYMM.htm` (≥2014-03); `…/pastwx/mwsYYYYMM.htm` before | **HTML prose, not PDF** | monthly | 1997-01 → 2026-09 | yes | 200, LEN 32457 (2026-08), 26960 (2026-09) |
| MWS selector | `https://www.hko.gov.hk/en/wxinfo/pastwx/mws/mws.htm` | year/month dropdown 1997-2026 | static | — | yes | 200, LEN 23549 |
| API doc | `https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf` | v1.13 Sep 2025, 48 pp | static | — | yes | 200 `application/pdf`, 556755 B |

## 2. Detail per working source

**dailyExtract_YYYY.xml** — JSON `{"stn":{"data":[{"month":N,"dayData":[[…]]}]}}`. 12 fields/day:
`[day, MSLP(hPa), maxT(°C), meanT(°C), minT(°C), meanDewPt(°C), meanRH(%), meanCloud(%), totalRainfall(mm), sunshine(h), prevailingWindDir(deg), meanWindSpeed(km/h)]`.
Semantics **proven arithmetically**: 2004-01-31 row `["31","1027.0","11.6"," 9.4"," 8.2","-1.8","46",…]`; Td at 9.4 °C / 46 % RH = −1.56 °C ≈ −1.8 ✓. Each month ends with a `Mean/Total` row (monthly actual) then a `Normal` row (1991-2020 normal). Rainfall may be `Trace` (<0.05 mm) or carry `#` = incomplete.

```powershell
(Invoke-WebRequest 'https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_2004.xml' -UseBasicParsing).Content | ConvertFrom-Json
```

**tc.dat** — TSV:
`[tcID, intensity, name, signal, dir, startHHMM, startDD, startMM, startYYYY, startFlag, endHHMM, endDD, endMM, endYYYY, endFlag, durationHHMM]`.
`dir` ∈ `X,NE,SW,NW,SE` (signal 8 only). Rows with intensity `MSN` are the **Strong Monsoon Signal**, not a TC — filter separately. Trailer `UUUU` marks the newest provisional rows. Last record: `202603 SuperT SAUDEL 1 X 1810 31 8 2026 X 010 4 9 2026 X 7800`.

**rstorm.dat** — TSV:
`[type(A=Amber,R=Red,B=Black), startYYYY, startMM, startDD, startHH, startMin, endYYYY, endMM, endDD, endHH, endMin, durHH, durMin]`.
e.g. `A 1998 4 12 5 15 1998 4 12 8 0 2 45` = Amber 1998-04-12 05:15→08:00 (2 h 45 m).

```powershell
Invoke-WebRequest 'https://www.hko.gov.hk/dps/wxinfo/climat/warndb/tc.dat' -UseBasicParsing -OutFile tc.dat
```

## 3. Rejected / dead ends

| URL | Result | Why unusable |
|---|---|---|
| `opendata.php?dataType=CLMRAIN` / `CLMSUN` | 200, `text/html`, **LEN 534** | "Please include valid parameters in API request." — dataType does not exist |
| `weather.php?dataType=clmimd&lang=en` | 200, LEN 534 | Same generic error page |
| `weather.php?dataType=warnsum` / `warningInfo` | 200, **LEN 2** | Body `{}` — current warnings only, no history |
| `en/wxinfo/currwx/tc_review.htm`, `en/informtc/tcsignal.htm`, `en/cis/wxsum.htm`, `en/cis/monsum.htm`, `en/informtc/tcDatabase.htm` | **404** | Guessed paths; real ones are `/en/cis/warndb.htm` + `/dps/.../*.dat` |
| `dailyExtract_200401.xml` | **404** | No monthly variant; year granularity only |
| `cgi-bin/hko/daily_extract.pl` | **404** | Legacy CGI removed |
| `daily_HKO_{SUN,MAX_TEMP,MIN_TEMP,GRS,GLS,EVAP,WIND,DIR}_ALL.csv` | **404** | Wrong codes; sunshine exists only at King's Park = `daily_KP_SUN_ALL.csv` |
| `data.gov.hk/api/3/action/package_show` | **404** | — |
| `api.data.gov.hk/v1/3/action/package_show` | **403 Forbidden** | CKAN API closed; scrape the dataset HTML instead |
| `pastwx/mws202608.htm` (flat) | **404** | Post-2014-03 needs `mws2026/mws202608.htm` |
| `pastwx/metob202608.htm` | **404** | `metob*` series ends 2019-10 in the sitemap |
| `en/cis/climahk.htm` | 200, 10187 B | Narrative prose only |

**JS-shell finding.** The `data.gov.hk/en-data/dataset/hk-hko-rss-*` pages return 200 with 87–576 kB of HTML. They are *not* empty shells — real download URLs are embedded, but on some pages with escaped slashes (`https:\/\/…`), which is why a naive regex finds 0 hits on `daily-total-bright-sunshine` (87422 B) yet finds hits on `daily-grass-min-temp`. Unescaping `\/`→`/` exposes `…/csvfile/KP/ALL/daily_KP_SUN_ALL.csv`. The endpoint family behind all of them is `https://data.weather.gov.hk/weatherAPI/cis/csvfile/<STN>/{<YEAR>|ALL}/daily_<STN>_<CODE>_<YEAR>_ALL.csv`.

**Enumerated negative (not a guess).** The API doc's own CSV response table (p. 34; `pdftotext -layout` line ~1496) lists **exactly** these `opendata.php` datasets: `HHOT, HLT, SRS, MRS, LHL, LTMV, CLMTEMP, CLMMAXT, CLMMINT`. There is **no** official API dataType for daily rainfall, sunshine, RH, cloud, or any warning history.

## 4. Aggregating daily → monthly without leakage

**Rule: a monthly feature for month *t* may use only observations dated ≤ the forecast origin. The HKO files DO contain post-origin data.**

With origin = 2026-08-31 (263 months, 2004-10 → 2026-08):

1. Load `dailyExtract_1884.xml … dailyExtract_2026.xml`; flatten `(year, month, day) → 12 elements`. Drop the trailing `Mean/Total` and `Normal` pseudo-rows.
2. **Hard truncate:** `df = df[df.date <= pd.Timestamp('2026-08-31')]`. Mandatory — `dailyExtract_2026.xml` already holds a complete September 2026 (32 rows) as of the 2026-10-13 scrape. The bulk CSVs happen to stop at 2026-08-31, but do **not** rely on that; assert it.
3. Aggregate per `(year, month)`: `sum` for rainfall/sunshine, `mean` for temperature/RH/cloud/pressure, `count` for derived day-counts — cold day = `minT ≤ 12.0`, very hot day = `maxT ≥ 33.0`, hot night = `minT ≥ 28.0` (HKO thresholds).
4. Warnings → counts: expand each record to the calendar dates from start to end **inclusive**, count **distinct dates** per `(year, month)`; for the origin month clip to dates ≤ origin. Drop records whose **start** > origin — e.g. rstorm rows dated 2026-10-02/03/04, and the tc.dat `SAUDEL` row starting 2026-08-31 18:10 and ending 2026-09-04, whose September portion would leak. Filter `MSN` rows out of `tc.dat` for a pure typhoon-signal-day count.
5. Feature lags must be ≥ 1 month vs the target (`shift(1)`): month *t*'s weather is contemporaneous with, not predictive of, month *t*'s sales.
6. Never read a month's `Mean/Total` row as a feature for that same month. The `Normal` row is a fixed 1991-2020 climatology — leakage-free but zero time-varying signal; use it only to build anomalies, computed identically in train and test.

## 5. Coverage matrix (target window 2004-10 → 2026-08)

| Driver | Covers 2004-2026 fully? | From | Verdict |
|---|---|---|---|
| Mean / max / min temperature | **Yes** | MAXT/MINT 1884-01-01; TEMP 1884-03-01 | USE |
| Total rainfall | **Yes** | 1884-03-01 | USE |
| Relative humidity | **Yes** | 1947-01-01 | USE |
| Sunshine duration | **Yes** | monthly 1961 (`hko.xml`); daily in `dailyExtract` | USE (via `dailyExtract`/`monthlyExtract`; HKO bulk `SUN` CSV 404s) |
| Cloud amount / MSLP / dew point | **Yes** | 1949-01-01 / 1884-03-01 / 1961-01-01 | USE |
| Cold days (minT ≤ 12) | **Yes** | derivable from MINT 1884 | USE — derived, not published |
| Very hot days (maxT ≥ 33) | **Yes** | derivable from MAXT 1884 | USE — derived, not published |
| Hot nights (minT ≥ 28) | **Yes** | derivable from MINT 1884 | USE — derived, not published |
| TC signal days | **Yes** | `tc.dat` 1946-07-16 | USE |
| Rainstorm-warning days | **Yes** | `rstorm.dat` 1998-04-12 | USE |
| Thunderstorm days | **Yes** | `thunder.dat` 1967-04-03 | USE as warning-issuance proxy (no published thunderstorm-day dataset) |
| Strong monsoon signal days | **Yes** | `tc.dat` MSN rows | USE |
| Global solar radiation | **Yes** | monthly 1978 | USE (marginal) |
| Reduced-visibility hours | **Yes** | 1997-01-01 | USE (marginal) |
| Lightning counts | **Partial** | 2005 only | SKIP — misses 2004-10…2004-12 |
| Sea temperature | Available | 1974 | marginal |
| HKO-published cold-day / hot-night / very-hot-day counts | **No such dataset** | — | Enumerated negative (see §3) |
| Monthly Weather Summary as numeric series | HTML prose only | 1997-01 | Narrative/manual checks only, not a numeric feed |

**Bottom line.** HKO daily data reaches back to **1884-03-01** (`1884,3,1,0.0,C` in `daily_HKO_RF_ALL.csv`; max/min temperature to 1884-01-01), far beyond the 2004-10 origin, and every driver above is fully covered for 2004-10 → 2026-08. Row-level warning history is downloadable and yields exact typhoon-signal-day and rainstorm-day counts per month.
