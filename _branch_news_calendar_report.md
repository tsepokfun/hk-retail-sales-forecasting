# Branch Report — News Sentiment, EPU, GDELT, Calendar/Events
Project: HK monthly retail sales forecast, 2004-10 → 2026-08 (263 months).
All fetches run from `pwsh` with
`Invoke-WebRequest -Uri <url> -UseBasicParsing -TimeoutSec 45 -Headers @{'User-Agent'='Mozilla/5.0'}`
(3 s between same-host requests; 60 s / 120 s back-off for GDELT). Every row below was actually fetched.

---

## 1. Verified-sources table

| Source | Exact URL | What it gives | Freq | Months covered | Free | How verified |
|---|---|---|---|---|---|---|
| EPU HK index | `https://www.policyuncertainty.com/media/HK_EPU_Data_Annotated.xlsx` | Monthly HK news-based EPU index | Monthly | **1998-04 → 2025-12** (333 rows, contiguous) | Yes | 200, 204,907 B, magic `50 4B 03 04`; parsed with openpyxl |
| EPU Global+countries | `https://www.policyuncertainty.com/media/All_Country_Data.xlsx` | GEPU_current, GEPU_ppp + 21 country cols incl. China/Korea/Singapore/Japan | Monthly | **1985-01 → 2026-07** (499 rows) | Yes | 200, 130,388 B; sheet `EPU`, dims `A1:Z526` |
| EPU China (standalone) | `https://www.policyuncertainty.com/media/China_Policy_Uncertainty_Data.xlsx` | China news-based EPU | Monthly | 1995-01 → **2019-04** (292 rows) — **stale** | Yes | 200, 9,090 B; sheet `China EPU Index` |
| Google News RSS | `https://news.google.com/rss/search?q=…&hl=en-US&gl=US&ceid=US:en` | Article counts + headlines per month | Daily, query per month | ~2012 → present (see §4) | Yes | 16 month-queries; item counts via `[regex]::Matches($c,'<item>')` |
| GovHK press releases | `https://www.info.gov.hk/gia/general/YYYYMM/DD.htm` | Full daily release list (bilingual, includes monthly retail sales releases) | Daily | **2000 → present** | Yes | 2000/2003/2004/2005/2006/2007/2008 + 2020/2021/2022/2023 all 200 w/ real content |
| HKO Gregorian–Lunar calendar | `https://www.hko.gov.hk/en/gts/time/calendar/pdf/files/YYYYe.pdf` | Authoritative lunar↔Gregorian table | Annual PDF | **1901 → 2100** | Yes | 23/23 PDFs 2004–2026 downloaded, text layer extracted |
| GovHK general holidays | `https://www.gov.hk/en/about/abouthk/holiday/YYYY.htm` | Gazetted general holidays | Annual | **2022 → 2027 only** | Yes | 2022–2026 = 200; 2004/2006/2008/2010/2012/2015/2017/2018/2019/2020/2021 = **404** |
| GDELT v2 bulk | `http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip` | 15-min event/mention files, no key | 15-min | **2015-02-19 → present** | Yes | `lastupdate.txt` 200/319 B live; 2020 file 200/162,032 B valid zip |

---

## 2. Detail per working source

### A. EPU Hong Kong — **VERIFIED, primary driver**
The page `hk_monthly.html` (200, 15,422 B) contains exactly two data hrefs:
`media/HK_EPU_Data_Annotated.xlsx` and `media/HK_EPU_Paper.pdf`.
The file has **three sheets**:

| Sheet | Dims | Layout |
|---|---|---|
| `HKEPU` | `A1:B334` | `A=Date` (real Excel date, 1st of month), `B=Monthly_HKEPU_Index` (float, mean≈100 by construction). 333 data rows, **1998-04 → 2025-12, no gaps**. Observed range 23.011 – 433.046. |
| `Annotation` | empty | — |
| `Events` | `A1:B13` | 12 hand-annotated events (Asian Financial Crisis, 911, SARS, GST proposal, Lehman, European Debt Crisis, US downgrade, RMB fixing, Brexit, HK CE election) — free event labels. |

Units: index, not percent. Fetch + parse one-liner:
```powershell
Invoke-WebRequest 'https://www.policyuncertainty.com/media/HK_EPU_Data_Annotated.xlsx' -OutFile hk.xlsx -UseBasicParsing -Headers @{'User-Agent'='Mozilla/5.0'}
python -c "import openpyxl;ws=openpyxl.load_workbook('hk.xlsx',data_only=True)['HKEPU'];print([r for r in ws.iter_rows(min_row=2,max_row=3,values_only=True)])"
```
**Gap to note: last month is 2025-12, so the final 8 months of the project window (2026-01 → 2026-08) will be NaN at build time.** The `All_Country_Data.xlsx` file runs one month later (2026-07) and is the better spine if you want max recency.

### B. Google News RSS — see §4 for the stability verdict.

### C. GovHK press release archive — **earliest year = 2000**
`200010/03.htm` → 200, 3,948 B, 27 `<a>` tags, real releases
(`"Public Order Ordinance Explained"`, `"MPFA's 'Push Campaign' set for launch"`).
`200310/03.htm` → 200, 6,437 B. `200410/15.htm` → 200, 6,843 B / 34 links.
2005 (8,340 B), 2006 (3,202 B), 2007 (5,264 B), 2008 (8,738 B) all 200 with content.
Title check on 2007: `<title>HKSAR Government Press Releases 15-10-2007</title>`.
**Verdict: full coverage of 2004-10 → 2026-08.** Two encoding traps: 2007/2008 pages declare
`charset=BIG5`; modern pages use `/gia/general/YYYYMM/DD/PYYYYMMDDNNNNN.htm`.

### D. GDELT — see §3 (honest failure + working workaround).

### E. Calendar
**LNY dates are extractable programmatically** from HKO PDFs (no JSON/CSV/TXT exists — see §3).
Layout differs by vintage (pre-2011 ≈ 20 KB, 2011+ ≈ 200–290 KB) and the January `Lunar`
marker sometimes sits on its own text line, so map each `Lunar` word to a day column by
**maximum x-overlap** against the header row `1…31`, then take the marker whose ordinal is `1st`.

**Trading days rule (computable, no fetch):** `trading_days = days_in_month − weekend_days − general_holidays_falling_on_weekdays`.
Saturday is not a trading day for the securities market, so only weekday-falling holidays reduce the count.
Fully computable 2004-2026 from the LNY table plus the fixed-date holiday list.

---

## 3. Rejected / dead ends (raw)

**GDELT Doc API — HTTP 429, three attempts, honestly failed.**
```
FAIL [gdelt-doc-artlist]  HTTP=429 | 429 (Too Many Requests)     <- mode=artlist, after 60s wait
FAIL [gdelt-timelinevol]  HTTP=429 | 429 (Too Many Requests)     <- mode=timelinevol, after 120s wait
```
Reproduces the operator's 6 failures. **The REST API is not usable from this host.**

**GDELT bulk files — WORKS, no key.**
```
OK [gdelt-lastupdate]      200 | text/plain | bytes=319
   54446 6f18b1b0… http://data.gdeltproject.org/gdeltv2/20261013031500.export.CSV.zip
OK [gdelt-bulk-csv]        200 | application/zip | bytes=162032   (magic 50 4B 03 04)
```
Boundary probe: `20150217000000` → **404**, `20150218000000` → **404**, `20150219000000` → **200, 362,610 B**.
So **GDELT v2 bulk starts 2015-02-19** — it cannot cover 2004-10 → 2015-02 at all.
GDELT v1 path `http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip` → **404** on 2004-10-15, 2006-01-01, 2013-01-01.
Practical cost: 4 files/hour × 24 × 365 × ~10.5 y ≈ **368,000 zips** for 2015-02→2026-08. Infeasible for this project.

**GDELT BigQuery — needs billing, confirmed.**
```
POST https://bigquery.googleapis.com/bigquery/v2/projects/gdelt-bq/queries
  => HTTP 401  (no auth).  gdelt-bq.gdeltv2.events requires a GCP project with billing.
```

**Other 404s:** `media/GlobalEPU_Data.xlsx` (404); `media/China_Mainland_Policy_Uncertainty_Data.xlsx` (404);
`https://www.gov.hk/en/about/abouthk/holiday/2004.htm` (404, and 2006/2008/2010/2012/2015/2017/2018/2019/2020/2021 likewise);
`consumptionvoucher.gov.hk/en/` (connection failure, no HTTP status);
HKO `…/calendar/text/T2004e.txt`, `…/calendar/json/2004e.json`, `…/calendar/files/2004e.csv` → all **404**
(only PDFs exist; `conversion.htm` has no form/select/JSON — no hidden endpoint behind it).

**Discrete events — evidence level** (all fetched from `info.gov.hk/gia/general/<date>.htm`):
- **2021-08-01** voucher: *"Consumption Voucher Scheme has disbursed first voucher to about 5.5 million registrants today"* (200, 13,018 B). ✔
- **2023-02-03**: *"Full resumption of normal travel between Hong Kong and Mainland"* (200, 15,213 B). ✔
- **2022-01-04**: *"…announces plan for expanding 'vaccine bubble'"* (200, 16,237 B). ✔
- **2020-03-25** non-resident ban: only related items on that date (inbound-traveller specimen centres); ban announced at the 2020-03-23 press conference (200, 14,430 B, 29 releases). **Partial — exact wording not located.**
- **2008-09 GFC (Lehman)**: dated `Sept 2008` in the EPU `Events` sheet — month, not day.
- **2019 protests, voucher amounts/instalment schedule, 2023-24 "night vibes"/mega events: UNVERIFIED** — TODO.

---

## 4. Google News RSS stability (query `Hong Kong retail sales`)

| Month | HTTP | Bytes | `<item>` count | Earliest pubDate | Latest pubDate |
|---|---|---|---|---|---|
| 2004-10 | 200 | 1209 | **0** | — | — |
| 2005-01 | 200 | 1209 | **0** | — | — |
| 2005-06 | 200 | 3474 | 2 | 2005-06-03 | 2005-06-29 |
| 2005-12 | 200 | 2276 | 1 | 2005-12-13 | 2005-12-13 |
| 2006-01 | 200 | 3403 | 2 | 2006-01-06 | 2006-01-12 |
| 2006-06 | 200 | 2917 | 2 | 2006-06-23 | 2006-06-25 |
| 2007-01 | 200 | 8168 | 5 | 2007-01-01 | 2007-01-23 |
| 2008-01 | 200 | 5825 | 4 | 2008-01-10 | 2008-01-24 |
| 2009-01 | 200 | 1209 | **0** | — | — |
| 2010-01 | 200 | 3608 | 2 | 2010-01-28 | 2010-02-01 |
| 2011-01 | 200 | 16829 | 14 | 2011-01-04 | 2011-01-31 |
| 2012-06 | 200 | 29685 | 22 | 2012-06-06 | 2012-06-29 |
| 2013-01 | 200 | 49199 | **37** | 2013-01-03 | 2013-01-31 |
| 2016-03 | 200 | 60371 | 45 | 2016-03-01 | 2016-04-01 |
| 2020-02 | 200 | 69533 | 49 | 2020-02-03 | 2020-02-28 |
| 2024-01 | 200 | 64733 | 50 | 2024-01-02 | 2024-02-01 |

The 1,209-byte response is a **valid, live, empty feed**, not a JS shell —
`<generator>NFE/5.0</generator> … <lastBuildDate>Tue, 13 Oct 2026 03:06:40 GMT</lastBuildDate>` with zero `<item>` elements.

**Verdict.** Stable and usable, but it does **not** cover 2004-2026 evenly:
- **2004-10 and 2005-01 return 0 items.** The operator's 2013-01 baseline reproduces (37 items here vs 44 then — same effect, different day).
- Coverage is **sparse and non-monotonic** 2005–2011 (1–14 items), and **2009-01 returns 0 while 2007-01 returned 5** — Google's index for this query is simply thin there. Do not assume a clean year cutoff.
- **Earliest non-trivial count: 2006** (2 items). **Earliest month with a usable ≥20-item sample: 2012.**
- Two construction hazards: counts appear **capped near 50** (45/49/50 for 2016/2020/2024), so the index saturates post-~2016 and *levels are not comparable* across eras; and `before:` is **not strictly exclusive** — 2016-03, 2024-01 and 2010-01 each leak one article into the next month (latest pubDate = 2016-04-01 / 2024-02-01 / 2010-02-01).

**Recommendation:** build the news-sentiment layer **2012-01 → 2026-08** and use it as a *level-free* regressor (z-score within regime, or counts normalised per era), leaving 2004-10 → 2011-12 to EPU + calendar drivers.

---

## 5. Lunar New Year 2004-2026 (HKO PDFs, 23/23)

| Year | LNY | Month | Year | LNY | Month |
|---|---|---|---|---|---|
| 2004 | Jan 22 | **Jan** | 2016 | Feb 8 | Feb |
| 2005 | Feb 9 | Feb | 2017 | Jan 28 | **Jan** |
| 2006 | Jan 29 | **Jan** | 2018 | Feb 16 | Feb |
| 2007 | Feb 18 | Feb | 2019 | Feb 5 | Feb |
| 2008 | Feb 7 | Feb | 2020 | Jan 25 | **Jan** |
| 2009 | Jan 26 | **Jan** | 2021 | Feb 12 | Feb |
| 2010 | Feb 14 | Feb | 2022 | Feb 1 | Feb |
| 2011 | Feb 3 | Feb | 2023 | Jan 22 | **Jan** |
| 2012 | Jan 23 | **Jan** | 2024 | Feb 10 | Feb |
| 2013 | Feb 10 | Feb | 2025 | Jan 29 | **Jan** |
| 2014 | Jan 31 | **Jan** | 2026 | Feb 17 | Feb |
| 2015 | Feb 19 | Feb | | | |

**9 Jan / 14 Feb.** Independent cross-check from the GovHK gazette pages:
2024 → *"Lunar New Year's Day 10 February Saturday"*; 2023 → *"The second day of Lunar New Year 23 January"* (⇒ day 1 = 22 Jan); 2025 → *"Lunar New Year's Day 29 January"*. All agree.

Derived features: `lny_in_january` (binary), `days_from_month_start_to_lny`, and the
**Jan/Feb sales-split indicator** — the single strongest calendar feature here, since a January
LNY concentrates the pre-holiday spending spike into January while a February LNY shifts it out.

---

## 6. Coverage matrix

| Driver | Covers 2004-10 → 2026-08 fully? | If not, from when | Verdict |
|---|---|---|---|
| EPU HK (`HK_EPU_Data_Annotated.xlsx`) | No — ends **2025-12** | 1998-04 | **USE.** Best single sentiment proxy; 8 trailing months NaN |
| EPU Global/country (`All_Country_Data.xlsx`) | No — ends **2026-07** | 1985-01 | **USE** as spine; has China/Korea/Singapore |
| EPU China (standalone) | No — ends **2019-04** | 1995-01 | **REJECT** — stale; use `All_Country` China col |
| Google News RSS | **No** | usable 2012-01; thin 2006-2011; dead ≤2005 | **USE FROM 2012.** Level-free only; ~50 cap |
| GovHK press releases | **Yes** | 2000 | **USE.** Full-window text/news layer, free |
| HKO lunar calendar | **Yes** | 1901 | **USE.** LNY + lunar-date features |
| GovHK general holidays (per-year pages) | **No** | 2022 | **PARTIAL** — 2004-2021 needs gazette/ordinance or the computable rule |
| Trading days / weekends | **Yes** | computable | **USE.** Derived, no fetch |
| GDELT (all routes) | No | v2 bulk 2015-02-19; REST/BigQuery unusable | **REJECT** — REST 429 ×3; bulk ~368k files & wrong window; BigQuery 401 |
| Discrete event dummies | Mostly | — | **PARTIAL** — §3 evidence levels |

**Bottom line:** whole-window coverage comes from GovHK press releases + HKO calendar + EPU;
Google News RSS sentiment only from ~2012; GDELT is out.
