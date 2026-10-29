# Branch report — TRANSPORT & FOOTFALL DRIVERS

Every URL was requested with `Invoke-WebRequest -Uri <url> -UseBasicParsing -TimeoutSec 45 -Headers @{'User-Agent'='Mozilla/5.0'}`. Status / CT / bytes / body prefix are quoted from actual responses. Nothing here is inferred.

## 1. Verified sources

| Source | Exact URL | Gives | Freq | Covered | Evidence |
|---|---|---|---|---|---|
| **MTR patronage CSV** ⭐ | `POST https://www.mtr.com.hk/en/corporate/investor/pf_export.php` | MTR patronage by service | Monthly | **Jan 2000 → Aug 2026 (320 rows)** | 200, 25744 B, `Content-Disposition: attachment; filename="Patronage_20261013.csv"` |
| MTR patronage HTML | `POST https://www.mtr.com.hk/en/corporate/investor/patronage.php` | same, HTML | Monthly | same | 200, 168122 B, 324 `<tr>` |
| **TD Digest index** | `https://www.td.gov.hk/en/transport_in_hong_kong/transport_figures/monthly_traffic_and_transport_digest/index.html` | Archive index, 2 URL eras | Monthly | 2003-01 → 2026-07 | 200, 60281 B, title `Transport Department - Monthly Traffic and Transport Digest` |
| TD Digest, HTML era | `…/monthly_traffic_and_transport_digest/YYYY/YYYYMM/index.html` | per-table XLS links | Monthly | **2018-01 → 2026-07** | `202607`→200, 34562 B, title `Transport Department - July 2026`, 9 tables |
| TD Digest, PDF era | `https://www.td.gov.hk/filemanager/en/content_<ID>/YYMM.pdf` | Whole digest | Monthly | **2003-01 → 2017-12** | `0301`→200/4001521 B; `0401`→200/1262515 B; `1701`→200/3444027 B; all `%PDF-` |
| TD Table 2.1 | `…/content_<ID>/table21.xls` | PT journeys **by operator** | Monthly+annual | rolling window (§2.3) | 200, 118272 B, `application/vnd.ms-excel`, HEX `D0 CF 11 E0 A1 B1 1A E1` |
| TD Table 3.1(a) | `…/content_<ID>/table31a.xls` | **Cross Harbour Tunnel** flow | Monthly+annual | rolling window | 200, 59904 B, OLE2 |
| TD Table 2.4 | `…/content_<ID>/table24.xls` | Cross-harbour passengers | Monthly+annual | rolling window | 200, 64512 B, OLE2 |
| HKIA monthly (new) | `…/facts-figures/MonYYYY_statistics_e.pdf` | HKIA pax/cargo/ATM | Monthly | **Feb 2024 → Aug 2026** | `Aug2026`→200/99292 B/`%PDF-` |
| HKIA monthly (old) | `…/facts-figures/MonYYYYe.pdf` | same | Monthly | **Oct 2016 → Jan 2024** | `Oct2016`→200/15999 B/`%PDF-` |
| HKIA annual | `…/facts-figures/YYYYe.pdf` | Annual | Annual | **1998 → 2026** | `1998e`→200/19456 B; `2026e`→200/156976 B |
| censtatd index | `https://www.censtatd.gov.hk/data/en/all_web_tables.json` | 590-table code+title index | — | — | 200, `application/json`, 147110 B, **590** parsed |
| FA100065 | `…/FA100065/att/B72104FB2021XXXXB01.zip` | PT patronage | **Annual only** | 2010 → 2020 | 200, 3409 B, `PK`, 4 CSVs |
| data.gov.hk CKAN | `https://data.gov.hk/en-data/api/3/action/package_search?q=patronage` | Discovery | — | — | 200, 4224 B, `count=1` |

## 2. Detail

**2.1 MTR — the brief was wrong twice over: it is neither a JS shell nor a dead end.** `GET patronage.php` → 200/text/html/18976 B. Grep of raw HTML: `json`=0, `ajax`=0, `xlsx`=0, `iwov-resources`=0 — but `<table>`=3, `<td>`=100, `<tr>`=12, `<form name="MyForm" action="patronage.php#search" method="post">`. The table **is** server-rendered; the page just defaults to Jan–Aug 2026. The period `<select>` holds **78 options = 12 months + years 2000…2026**, i.e. the full searchable range.

```powershell
# all data, as CSV
$b=@{EXPORT_MONTH_FRM='01';EXPORT_YEAR_FRM='2000';EXPORT_MONTH_TO='08';EXPORT_YEAR_TO='2026';dummy='EXPORT';BUTTON='EXPORT'}
Invoke-WebRequest 'https://www.mtr.com.hk/en/corporate/investor/pf_export.php' -Method POST -Body $b -UseBasicParsing -OutFile mtr.csv
# HTML panel: swap pf_export.php -> patronage.php and dummy/BUTTON -> 'dummy'/'SEARCH' with SEARCH_* keys
```

**Units.** Header line 1 is literally `"Patronage figures (unit in thousands)"`. Column pairs: `Month | Domestic Service {Monthly Total, Average Weekday} | Airport Express {Monthly Total, Average Daily} | Cross-boundary {Monthly Total, Average Daily} | Intercity, Light Rail & Bus {Monthly Total, Daily Average} | HSR {Monthly Total, Daily Average}`.

- *Monthly Total* = trips that month, thousands. Aug 2026 Domestic `135,062` ≈ 135.06 M trips.
- *Average Weekday/Daily* = per-day, thousands. `135,062 / 4,599 ≈ 29.4` days ✔ consistent.
- Footnote (2) HSR began 2018-09-23 → `-` before. Footnote (3) Intercity Through Train suspended since 2020-01-30 (hence the merged grouping).
- 327 lines, **320 month rows**; last `"January 2000","66,073","2,271","865","28","-","-","-","-","-","-"`.

**2.2 TD Digest.** The brief's URL (`…/free_publications/monthly_traffic_and_transport_digest/index.html`) is a **404**. Real home is under `transport_figures`. Both eras link from that one index: **2018-01→2026-07** HTML at `…/YYYY/YYYYMM/index.html`; **2003-01→2017-12** PDF at `/filemanager/en/content_<ID>/YYMM.pdf` (plus `2001all.pdf` 3444379 B, `2002all.pdf` 3403535 B). The `<ID>` is **irregular per month** — 2018-01=`content_4880`, 2019-06=`content_4942`, 2020-06=`content_4975`, 2026-07=`content_5413` — so it must be scraped per page, never constructed.

Tables (2026-07 page, `.xls` + usually `.pdf`): `keystat`, `table11`; **`table21`** PT journeys by operator, `table21s` avg daily by operator, `table22`, `table23` MTRC & franchised buses, **`table24`** cross-harbour passengers, **`table31a` Cross Harbour Tunnel**, `table31b` Eastern, `table31c` Western, `table31d` ferries, `table31s` summary, `table32a…32t` (20 more tunnels/bridges), charts 21–27/31/71–72/81–87. Whole report: `content_<ID>/2607_e.pdf` → 200, 2432430 B, `%PDF`.

**2.3 ⚠ The TD `.xls` is a ROLLING WINDOW, not a full history.** Row-label dump of `table31a.xls`: annual rows `2021…2025`, then monthly rows `2025-02…2025-12`, `2026-01…2026-07`. `table21.xls` is identical in shape. **One month's `.xls` gives ~5 annual + ~18 monthly rows only** — so 2004→2026 monthly needs ~264 page scrapes (index → content ID → tableNN.xls). Budget for it. Latest TD month is **2026-07** (2026 index lists `202601…202607`).

**2.4 Formats.** Legacy BIFF8 `.xls`. Excel COM is **not registered here** (`80040154 REGDB_E_CLASSNOTREG`) but **python 3.13.5 + xlrd 2.0.2 reads them fine** (pandas 2.3.3 present). Chinese cells are Big5 mojibake; English header rows are clean:
- `Table 3.1(a) : Cross Harbour Tunnel` — `Private Cars / Taxis / Motor Cycles` × `{South-bound, North-bound, Two-way Total}` (41×20).
- `Table 2.1 : Public Transport Passenger Journeys by Public Transport Operator` — `KMB`, `Citybus`, `Franchise (HK & XHT)`, `Franchise (U & NT)`, `Franchise (Airport & N. Lantau)`, `Sub-total` (49×27).

## 3. Rejected / dead ends

| URL | Response | Why unusable |
|---|---|---|
| `…/facts-figures/YYYYMM_statistics_e.pdf` for `200401, 200801, 201201, 201601, 202001, 202401, 202603–202606` | **200 for ALL TEN**, each **exactly 28279 B**, magic `<!DOC` | **SOFT-404 TRAP.** Body title = `Page Not Found - Hong Kong International Airport`, 3× `page404`. A status-only pipeline would silently ingest 404 HTML for every month. The brief's "202605 verified working" does not hold. |
| `MonYYYYe.pdf` for `Jul2016, Aug2016, Sep2016` | 200 / 28279 B / `<!DOC` | same soft-404 → **monthly archive starts Oct 2016** |
| `MonYYYYe.pdf` for `Jun2016, Jan2016, Dec2015, Jun2015, Jan2015, Jun2014, Jan2014, Jan2013, Jan2010` | 200 / 28279 B / `<!DOC` | no pre-2016 files under this pattern |
| `https://www.hongkongairport.com/api/statistics` | 200, 28279 B | same soft-404 → **no HKIA JSON API** |
| `…/js/facts.js` | 200, **65 B** | trivial |
| `…/hkia-at-a-glance/statistics.page` | **404** | absent |
| TD `…/free_publications/…digest/index.html`, `…/digest/`, `…/index_t.html` | **404, 404, 404** | brief's TD URL wrong |
| `https://www.td.gov.hk/filemanager/en/content_1849/1801.pdf` | **404** | guessable IDs fail |
| `…/content_5413/table31a.pdf` | HTML "revamped website" redirect notice | that table is XLS-only in 2026-07 |
| `https://www.mtr.com.hk/en/corporate/investor/patronage.html` | 200, 158797 B, `MTR Website Temporarily Not Available` | dead variant |
| `https://www.tih.hk/` (KMB parent) | **FAIL — SSL could not be established** | HTTPS broken; `http://www.tih.hk/` → 200, 9821 B, `Transport International Holdings Ltd 載通國際控股有限公司` |
| `https://www.nwff.com.hk/` | **connection refused** | NWFF defunct; successor `https://www.sunferry.com.hk/en/` → 200, 41120 B |
| `https://www.ksf.com.hk/` | DNS fail | — |
| `https://www.hktramways.com/en/` | 200, **148 B** | empty shell |
| `https://www.kmb.hk/` | 200, **2669 B** | shell, no stats |
| `https://www.citybus.com.hk/` | 200, 117910 B | marketing homepage only |
| `https://www.censtatd.gov.hk/api/post.php` | 200 + JSON error | **UNRESOLVED, not dead.** 3 schemas rejected: raw JSON `{"query":[…]}` → `"Query is not defined"`; `?id=650-80002`+raw JSON → same; form field `query={…}` → `"Table ID is not defined, Lang code is not defined"`. Endpoint live, schema not cracked. |

**censtatd grep (590 tables, unfiltered list scanned).** Filter `transport|passenger|vehicle|tunnel|traffic|ferry|railway|bus|tram|visitor` → **15 hits**. Relevant: `410-50021/2` (trade by mode of transport), `635-74002…74008` (transport-sector companies), `635-93002` (air transport capacity), `650-80001` (visitor arrivals by nationality), **`650-80002` (visitor arrivals by mode of transport)**, `650-93021` (HKIA transit pax). **No monthly local-transport ridership table exists in censtatd.** data.gov.hk: `q=patronage`→1; `q="passenger journeys"`→**0**; `q=transport`→35.

## 4. Coverage matrix

| Driver | Covers 2004-10 → 2026-08? | From | Verdict |
|---|---|---|---|
| **MTR patronage (CSV)** | **YES — complete, no gaps** | Jan 2000 | **Full-range monthly regressor** |
| **TD Table 2.1** PT journeys by operator | Data exists 2003-01→2026-07 but not in one file | 2003-01 | Monthly regressor **if** ~264 pages scraped |
| **TD Tables 3.1(a)(b)(c)** cross-harbour tunnels | same caveat | 2003-01 | Monthly regressor (same scrape) |
| **TD Table 2.4** cross-harbour passengers | same caveat | 2003-01 | Monthly regressor (same scrape) |
| HKIA monthly pax/cargo/ATM | **NO** | **Oct 2016** (soft-404 before) | **Ablation only / post-2016 model** |
| HKIA annual | Annual only | 1998 | Annual control |
| censtatd transport tables | **NO** monthly ridership exists | — | **Unusable**; `650-80002` possible tourism-side ablation |
| FA100065 patronage | **NO** — annual, ends 2020 | 2010 annual | Ablation only |
| KMB / Citybus / Sun Ferry / Star Ferry / Tramways sites | **NO** machine-readable monthly series | — | **Unusable** — only via TD Tables 2.1/3.1 and TIH reports |

**Bottom line.** Exactly one transport driver covers 2004-2026 monthly: **MTR patronage** — and it is the cleanest source in the branch (one POST → ready CSV, Jan 2000 → Aug 2026). TD is far richer in principle (operator-level journeys, cross-harbour tunnel vehicle counts, cross-harbour passengers; 2003-01 → 2026-07) but the rolling-window `.xls` makes it a ~264-request scrape, not a download. HKIA is monthly only from Oct 2016 and PDF-only. censtatd and the operators contribute nothing monthly.

## 5. Artifacts

`_mtr_patronage.csv` (25744 B, 320 rows, Jan 2000→Aug 2026) · `_mtr_patronage_post.html` (168122 B, 324 `<tr>`) · `_mtr_patronage_shell.html` (18976 B, 7-month shell) · `_all_web_tables.json` (147110 B, 590 tables) · `_hkia_202605_response.html` (28279 B soft-404 proof)
