"""Fetch bibliographic metadata + abstracts from OpenAlex for the literature
positioning table. Saves raw JSON so nothing is transcribed by hand."""
from __future__ import annotations

import json
import sys
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "results" / "_lit"
OUT.mkdir(exist_ok=True)

QUERIES = [
    "M5 accuracy competition results findings and conclusions",
    "M5 competition LightGBM top performing methods machine learning",
    "retail sales forecasting gradient boosting versus deep learning comparison",
    "A methodology for applying k-nearest neighbor to time series forecasting",
    "k-nearest neighbours time series forecasting competitive accuracy simple method",
    "Forecasting theory and practice Petropoulos machine learning data requirements",
    "Evaluating time series forecasting models performance estimation methods small samples",
]


def abstract_text(inv):
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch(q):
    url = ("https://api.openalex.org/works?search="
           + urllib.parse.quote(q) + "&per-page=5")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    allres = {}
    for q in QUERIES:
        try:
            j = fetch(q)
        except Exception as e:                      # noqa: BLE001
            print(f"FAIL {q}: {e}")
            continue
        allres[q] = j
        (OUT / (q[:40].replace(" ", "_").replace("/", "_") + ".json")).write_text(
            json.dumps(j, indent=1))
        print("=" * 96)
        print("QUERY:", q)
        for w in j.get("results", []):
            print(f"  TITLE : {w.get('title')}")
            print(f"  DOI   : {w.get('doi')}   YEAR {w.get('publication_year')}")
            src = (w.get("primary_location") or {}).get("source") or {}
            print(f"  VENUE : {src.get('display_name')}")
            ab = abstract_text(w.get("abstract_inverted_index"))
            if ab:
                print("  ABS   : " + textwrap.shorten(ab, 1400))
            else:
                print("  ABS   : <absent in OpenAlex>")
            print()
    (OUT / "_all.json").write_text(json.dumps(allres, indent=1))
    print("saved raw JSON under", OUT)


if __name__ == "__main__":
    main()
