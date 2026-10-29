# Data Source Discovery — Branch: Other Retail / Online Retail / Logistics / Fuels / Books / Chinese Drugs

Scope: HK, monthly 2004-10 → 2026-08 (263 months). Target outlet types: `36 Books, newspapers, stationery and gifts`, `37 Chinese drugs and herbs`, `40 Other consumer goods NEC`, `9 Fuels`, plus censtatd `620-67031` online retail.

All fetches performed with:
`Invoke-WebRequest -Uri <url> -UseBasicParsing -TimeoutSec 45 -Headers @{ 'User-Agent'='Mozilla/5.0' }`

---

## 1. Verified sources table

| source | exact URL | what it gives | frequency | months covered (from→to) | free? | as-known-at-the-time? | how verified |
|---|---|---|---|---|---|---|---|
| censtatd 620-67031 | `https://www.censtatd.gov.hk/api/get.php?id=620-67031&lang=en&full_series=1` | Online retail sales value, 3 series | **M** (+Y) | **202001→202608 (80 monthly)** + annual 2020–2025 | yes | yes (vintage) | 200, `application/json`, 230000 b, header title "Value of online retail sales by sel…" |
| censtatd 720-87002 | `…/api/get.php?id=720-87002&lang=en&full_series=1` | % companies with e-commerce sales/purchases/e-delivery; e-commerce % of receipts | **Y only** | 2000→2025 (26 points) | yes | no | 200, json, 32382 b; `freq=Y n=104 periods=26` |
| censtatd 410-55111A | `…/api/get.php?id=410-55111A&lang=en&full_series=1` | Seaborne cargo throughput **by month** | **M** | **199701→202606 (354)** + annual 1993–2025 | yes | yes | 200, json, **2,269,294 b**; `freq=M n=4872 periods=354` |
| censtatd 410-55112A | `…/api/get.php?id=410-55112A&lang=en&full_series=1` | River cargo throughput by month | **M** | **199801→202606 (342)** | yes | yes | 200, json; `freq=M n=4704 periods=342` |
| censtatd 410-55290 | `…/api/get.php?id=410-55290&lang=en&full_series=1` | Container throughput ('000 TEUs) + YoY% | **Q** | 199903→202606 (110 q) + annual 1997–2025 | yes | yes | 200, json, 911010 b; `freq=Q n=1980 periods=110` |
| censtatd 410-92011 | `…/api/get.php?id=410-92011&lang=en&full_series=1` | Inward/outward cargo movements **by air** ('000 t) | **Q** | 199803→202512 (112 q) + annual 1998–2025 | yes | yes | 200, json, 889810 b; `freq=Q n=2016 periods=112` |
| censtatd 410-50031 | `…/api/get.php?id=410-50031&lang=en&full_series=1` | Imports by 10 principal commodity divisions (incl. petroleum) | **M** | **199301→202608 (404)** | yes | yes | 200, json; `freq=M n=8888 periods=404`; `sv=VAL_IM` = YoY % change only |
| censtatd 635-92076 | `…/api/get.php?id=635-92076&lang=en&full_series=1` | Postal services: letters (M articles) + **parcels ('000)** | **Q** | **202103→202606 (22 q)** + annual 2021–2025 | yes | yes | 200, json, 62798 b; 8 Q series; source field = "Hongkong Post" |
| censtatd 915-91104 | `…/api/get.php?id=915-91104&lang=en&full_series=1` | Monthly oil/coal net-import quantity **+ HK$/litre unit values** | **M** | **201301→202608 (164)** + annual 2013–2025 | yes | yes | 200, json, 607158 b; `svDesc: HK$/litre, Kilolitres, Tonnes`; `freq=M n=1968 periods=164` |
| censtatd 915-91106 | `…/api/get.php?id=915-91106&lang=en&full_series=1` | Sales of oil products by type of user | **Q** | 201303→202606 (41 q) + annual 2013–2025 | yes | yes | 200, json, 1086592 b; `freq=Q n=2214 periods=41` |
| censtatd 915-91201 | `…/api/get.php?id=915-91201&lang=en&full_series=1` | Electricity & gas consumption by user type | **M** | monthly (back to 1979 per family table) | yes | yes | 200, json, 1,628,358 b |
| censtatd 510-60003 | `…/api/get.php?id=510-60003&lang=en&full_series=1` | CPI at COICOP **division** level (14 divisions) | **M** | 200501→202608 (260) | yes | yes (revisable) | 200, json, 9,099,860 b; `freq=M n=29120 periods=260` |
| censtatd 510-60001A | `…/api/get.php?id=510-60001A&lang=en&full_series=1` | CPI at commodity/service **section** level (10 sections) | **M** | index non-null **198010→202608** (626 periods listed) | yes | yes | 200, json, **21,962,337 b** |
| Marine Dept | `https://www.mardep.gov.hk/filemanager/en/share/fact/pdf/portstat_1_m_a1.pdf` | Monthly port/container throughput | M | current month + history | yes | yes | 200, `application/pdf`, 177897 b, magic `%PDF-` |
| Marine Dept | `…/filemanager/en/share/fact/pdf/portstat_1_b2.pdf` | Container throughput by handling location & seaborne/river | M/Q | — | yes | yes | 200, `application/pdf`, 131509 b, `%PDF-` |
| HKMPB | `https://www.hkmpb.gov.hk/document/HKP_KTCT-stat.pdf` | HK Port + Kwai Tsing container throughput | M | latest | yes | yes | 200, `application/pdf`, 248718 b, `%PDF-` |
| EIA | `https://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls` | Europe Brent spot price, monthly, US$/bbl | **M** | 1987→**9/2026** | yes | **yes (vintage!)** | 200, `application/vnd.ms-excel`, 55296 b; contains `Brent`, `Monthly`, `9/2026`, `Release Date: 10/7/2026` |
| Consumer Council | `https://oil-price.consumer.org.hk/en` | HK auto-fuel pump price comparison | daily/current | current only | yes | no | 200, `text/html`, 69700 b |
| Consumer Council | `https://online-price-watch.consumer.org.hk/opw/?lang=en` | Supermarket online price watch | current | current | yes | no | 200, `text/html`, 227814 b |

---

## 2. Full detail per source

**620-67031 (online retail).** Only 80 monthly points, 2020-01→2026-08, plus 6 annual. 3 series. Starts at the COVID e-commerce break — no pre-2020 baseline exists in this table. Revisions: none material; the table is a series-break table.

**720-87002 (e-commerce penetration).** ANNUAL, 2000→2025, 26 points, 104 records (4 indicators). Unit `%`. This is the **only long e-commerce penetration series** found. It is a survey of companies (establishment survey), so it is a slow structural covariate, not a monthly driver.

**410-55111A / 410-55112A.** Seaborne + river cargo by month, 354 + 342 monthly points — the **longest monthly logistics series in this branch**, spanning the entire 2004-10→2026-08 target window. Units: thousand tonnes. Aggregation: `GROUP`/`PMDesc` empty; series are the seaborne/river split.

**410-55290 (containers).** Quarterly only at censtatd (1999Q1→2026Q2). For MONTHLY container throughput use the Marine Dept PDF (`portstat_1_m_a1.pdf`) — PDF-only, requires tabula/pdfplumber extraction.

**410-92011 (air cargo).** Quarterly 1998Q3→2025Q4. **This is the durable air-cargo series**; HKIA's own monthly PDFs are not deep (see §3).

**635-92076 (postal).** Source field literally reads `Hongkong Post`. 8 quarterly series × 22 quarters: Local / Posted outside HK / From outside HK / In transit × {Million articles, Thousands parcels}. **Parcels are the direct e-commerce throughput proxy** — but only from 2021Q1.

**915-91104 (oil).** The single most useful fuel series: monthly net-import quantity AND **HK$/litre unit values**, 164 months. Closest thing to a monthly HK fuel deflator.

**915-91106 (oil sales by user).** Quarterly only. Units Kilolitres / Tonnes.

**510-60003 / 510-60001A (CPI).** COICOP division level = 14 divisions; section level = 10 sections. **Neither exposes "Fuels", "Books, newspapers and stationery" or "Chinese drugs" as a standalone monthly index.** See §3.

**EIA RBRTEm.xls.** `Data 1: Europe Brent Spot Price FOB (Dollars per Barrel)`, monthly, header row carries `Release Date: 10/7/2026`, latest period `9/2026`. Binary .xls — needs `pandas.read_excel`. EIA publishes it as a **vintage-stamped release**, so as-known-at-the-time reconstruction is partially possible from the release-date metadata.

---

## 3. Rejected / dead ends

| URL tried | what came back |
|---|---|
| `hongkongairport.com/iwov-resources/file/airport-authority/statistics/{202605,202001,201501,201001,200410}_statistics_e.pdf` | **200 but `text/html`, 28279 b IDENTICAL for all five months.** `<title>Page Not Found - Hong Kong International Airport</title>`. **Soft-404 — the brief's "verified working for May2026" claim is FALSE.** |
| `hongkongairport.com/.../facts-figures/Jan2026_statistics_e.pdf` | 200, **`application/pdf`, 100325 b, `%PDF-`** — REAL |
| same pattern `Jan{2016,2017,2018,2019,2020,2021,2022,2023,2024}` | all 200 `text/html` **28279 b** = soft-404. **Archive depth ≈ recent months only.** |
| `hkiacargo.com/en/monthly-aggregated-air-cargo-statistic` | 200, `text/html`, 77174 b — **SPA shell** (bundles `assets/hoisted.BsFUCb0Y.js`, `assets/page.V2R8AmkL.js`); 0 `.json`, 0 `.csv`, 0 `2019`, 0 `2024` occurrences. No data. |
| `hongkongairport.com/en/business-information/statistics.html` | 500 Internal Server Error |
| `fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU` | connection error / timeout at 45 s (blocked from this host) |
| `fred.stlouisfed.org/data/{DCOILBRENTEU,POILBREUSDM}.txt` | failed |
| `stooq.com/q/d/l/?s=cl.f&i=m` | 200 but 796 b JS-challenge page |
| `data.stats.gov.cn/easyquery.htm?cn=A01` (China NBS) | **403 Forbidden** |
| `stats.gov.cn/english/PressRelease/...` | 404 |
| `api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/banking/credit-card-lending-survey` (+`?offset=0`) | **405 Not Allowed, empty body** — despite being the **verbatim URL printed in HKMA's own docs**. Blocked from this environment. |
| `api.hkma.gov.hk/public/market-data-and-statistics/…` (all other paths tried) | 405 |
| `hkma.gov.hk/eng/.../payment-systems/`, `.../credit-card-lending-survey/`, `.../open-api/` | 404 |
| `thedocs.worldbank.org/.../CMO-Historical-Data-Monthly.xlsx` | 404 |
| `consumer.org.hk/en/oil-price-comparison`, `/en/price-watch` | 404 (correct path is `oil-price.consumer.org.hk`) |
| `hongkongpost.hk/en/about_us/statistics/` | 404. Working: `hongkongpost.hk/en/about_us/corp_info/publications/fact_sheet/index.html` (200, 14707 b) |
| `cmchk.org.hk/pc/pceng/index.htm` | 404. Root `cmchk.org.hk` = 200 but only 4670 b |

**Enumeration basis for the negatives:** `all_web_tables.json` was fetched in full (590 tables, 147110 b) and filtered on `online|e-commerce|fuel|oil|petrol|book|newspaper|printing|printed|Chinese medicine|drugs|herbs|postal|cargo|container|payment|credit card|CPI|Consumer Price|electricity|gas`. Separately, **all** `510-*` tables (6 total), **all** `620-*` tables (8 total) and **all** `410-*` tables (34 total) were listed unfiltered. **There is no censtatd table giving monthly print-media, stationery, book, or Chinese-medicine trade detail.** `410-50031` stops at 10 commodity divisions and contains no printed-matter or medicinal division.

---

## 4. Alignment recipe

- **Hard join:** all censtatd `/api/get.php?id=…&full_series=1` responses are a flat JSON array; `period` is `YYYYMM` (M), `YYYYQq`→`YYYYMM` at quarter-start (Q), or `YYYY` (Y). Reshape on (`period`, `sv`/`PMDesc`) → wide. No join keys beyond the month — pure time-axis merge.
- **620-67031 online** → inner-join, **loses 183 of 263 months** (2020-01 onward only). Model as a post-2020 regime dummy / short-window model, not a full-sample regressor.
- **720-87002** → annual; forward-fill within year or linearly interpolate to monthly; flag as interpolated.
- **410-55111A / 410-55112A / 410-50031 / 915-91104** → direct monthly inner-join across all 263 months. Log or YoY-difference before use (strong trend).
- **410-55290 / 410-92011 / 635-92076 / 915-91106** → quarterly; convert to monthly by repeating quarter value (or MIDAS / interpolate). Note parcel data starts 2021Q1 → **loses all pre-2021 months**.
- **EIA RBRTEm.xls** → monthly, join on month; convert US$/bbl × HKD/USD (censtatd 410-51001-family or HKMA) to approximate local fuel cost.
- **915-91104 HK$/litre** → best candidate **deflator for outlet type 9 Fuels**. `620-67003` (volume index of retail sales by outlet type) already gives the retail-side deflator — cross-check the two.
- **Marine Dept / HKMPB PDFs** → PDF-only; requires `pdfplumber`/`tabula` to reach the time axis. Only use if monthly container throughput is essential.

---

## 5. Honest verdict

- The single most important correction: **the HKIA `YYYYMM_statistics_e.pdf` pattern is a soft-404**, not a working source. Five different months returned byte-identical 28279-byte HTML "Page Not Found" pages. Any pipeline built on that pattern would silently ingest a 404 page. Do not use it.
- HKIA's *real* monthly PDF path (`facts-figures/MonYYYY_statistics_e.pdf`) works for recent months but 404s at every year ≤2024 tested — so **HKIA's own portal cannot supply a 2004→2026 air-cargo history.**
- Therefore **air cargo must come from censtatd `410-92011` (quarterly, 1998Q3→2025Q4)**, and the true monthly logistics backbone for this branch is **seaborne + river cargo (`410-55111A`, `410-55112A`), 354 / 342 monthly points** — these cover the whole target window and are the best logistics drivers found.
- **On the 80-month online series: no longer monthly substitute was found.** `620-67031` is 2020-01→2026-08 and that is the entire monthly online history in HK official statistics. The only longer e-commerce measure is `720-87002`, which is **annual, 2000→2025** — usable as a slow structural covariate with interpolation, never as a monthly driver. China NBS (the obvious external substitute) returned **403** from this environment. **Conclusion: any monthly online-retail modelling is structurally restricted to 2020-01+ (80 months).** Options: (a) restrict the online sub-model to 2020+; (b) use `720-87002` annual penetration as an interpolated covariate on the full sample and treat the monthly online series as validation only; (c) treat `635-92076` parcels (2021Q1+) as an independent monthly-ish check on the online channel.
- **The CPI deflator hope fails at the category level.** censtatd's finest public monthly CPI is COICOP *division* (14 groups, `510-60003`, 2005-01+). There is **no standalone monthly CPI for "Fuels", "Books, newspapers and stationery", or "Chinese drugs"** — those live inside `Housing, water, electricity, gas and other fuels`, `Recreation, sport and culture`, and `Health` respectively. For outlet-type 9 Fuels the practical deflator is `915-91104`'s **HK$/litre unit value** (monthly, 164 months).
- **Fuels is the best-served category in this branch:** monthly unit values (915-91104), quarterly sales by user (915-91106), monthly electricity/gas (915-91201), monthly Brent (EIA, to 9/2026), plus a live Consumer Council pump-price comparison site.
- **Books/stationery and Chinese drugs are the worst-served.** No censtatd monthly series at any level of the 590-table index isolates either, and `410-50031` has only 10 commodity divisions with no print or medicinal division. These two outlet types will have to be modelled from their own retail sales series plus generic macro drivers — **expect them to be the weakest-fitting categories, and say so rather than manufacturing a proxy.**
- **Payment statistics were NOT verified.** HKMA's documented credit-card-lending endpoint returns **405 with an empty body** from this environment even when copied verbatim from HKMA's own API docs. The docs confirm the series is **quarterly** (`end_of_quarter` in `yyyy-qq`, e.g. `1997-Q1`) — so even once reachable it is quarterly, not monthly. **Treat HKMA payments as UNVERIFIED / blocked**, and if a monthly payment volume series is needed, that is an open item requiring either a different network path or a data.gov.hk mirror.
- **Revisions:** censtatd retail/cargo/CPI series are revisable; `620-67031` annual figures back-fill the monthly. `510-60001A/60003` carry an explicit rebasing note (base Oct2019–Sep2020=100) and pre-Oct-2020 YoY was computed on the then-current base — **do not splice YoY across 2020-10 without care.** EIA's .xls is a stamped release, which is the one source here that gives a genuine as-known-at-the-time vintage.
