"""
Stage 3 / Part 1 — MULTI-DIMENSIONAL TEXT EMBEDDING & CATEGORY-SEMANTIC ROUTING.

WHAT PROBLEM THIS REPLACES
--------------------------
`news_features.py` turned every GovHK headline of a month into ONE vector
(mean of per-title MiniLM embeddings), compressed it with PCA(4), and fed the
SAME 4 numbers to all 23 retail categories.  Two things are wrong with that:

  * it mixes orthogonal semantic streams (interest-rate policy, inbound-tourism
    reviews, property/car/big-ticket chatter, grocery/weather chatter) into one
    monthly average, so a tourism-dominated month looks the same to a
    supermarket as to a jewellery store;
  * it has no notion of WHICH text is relevant to WHICH category.

This module implements the three-part replacement:

  A. MULTI-DIMENSIONAL TEXT STREAMS.  Each headline is soft-assigned to one of
     four orthogonal semantic dimensions (macro/policy, tourism & cross-border
     sentiment, durables/auto/property, staples & department retail).  The four
     streams are handled separately from that point on.

  B. CATEGORY-SEMANTIC ATTENTION ROUTING (the core innovation).  Every retail
     category j has a text prototype  P_j  (j = 1..23).  Inside a (month, dim)
     cell the routing weight of headline i for category j is

         w_{i,j} = softmax_i ( <emb(d_i), emb(P_j)> / tau )

     (softmax taken OVER HEADLINES i, for each category j separately), and the
     category's own text embedding for that month/dimension is the weighted mean

         E_{t,j,k} = sum_i  p_{i,k} w_{i,j} emb(d_i)  /  sum_i p_{i,k} w_{i,j}

     with p_{i,k} the soft dimension membership of headline i.  E_{t,j,k} is
     then reduced to D_TEXT=4 PCA scores (fit on period <= 202312 only) and
     shipped together with three intensities: routed sentiment, routed retail
     salience, and the dimension's news volume.  => 7 numbers per
     (category, dimension, month), 28 per (category, month).

  C. HORIZON-AWARE MULTIMODAL FUSION.
     * xLSTM path  -- `XLSTMForecasterFiLMHorizon` (from models/xlstm_exog.py)
       fed a SERIES-SPECIFIC exogenous cube instead of the old period-broadcast
       block, plus an L1 penalty on the horizon gates, so the h=1 head can stay
       on pure momentum while h=3/6/12 absorb the category's text semantics.
     * iTransformer path -- `ITransformerTextToken`: the 23 sales variate tokens
       get 4 additional TEXT-MODALITY tokens in the same inverted self-attention
       (23 + 4 = 27 tokens, ZERO new parameters in the encoder trunk).

LEAKAGE RULES (enforced, not asserted)
--------------------------------------
  1. A month's features are built from headlines whose `month` == that month.
     All four GovHK streams are published inside their own month.
  2. Every fitted object (MiniLM space is frozen; PCA, z-score mu/sigma) is fit
     on `period <= 202312`.
  3. `prefix_mask_test()` re-derives the whole feature tensor from a
     date-truncated headline table and asserts the in-window values are
     BIT-IDENTICAL (max |diff| must be exactly 0.0).

USAGE
    python multidim_text_model.py --stage build            # text -> feature tensor
    python multidim_text_model.py --stage verify            # causality + loop equivalence
    python multidim_text_model.py --stage fit --arms xlstm_routed it_routed
    python multidim_text_model.py --stage report            # ablation table + CSVs
    python multidim_text_model.py --stage all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent            # stage3/
PROJ = ROOT.parent
STAGE0 = PROJ / "stage0"
for p in (str(ROOT), str(STAGE0), str(STAGE0 / "eval"), str(STAGE0 / "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval import harness as H                    # noqa: E402
from models import xlstm_direct as X             # noqa: E402
from models import xlstm_exog as XE              # noqa: E402
from models import itransformer as IT            # noqa: E402

import torch                                     # noqa: E402
import torch.nn as nn                            # noqa: E402

DATA = ROOT / "data"
RES = ROOT / "results"
MD = DATA / "multidim"                           # cache dir
PRED = RES / "multidim" / "preds"
META = RES / "multidim" / "meta"

TRAIN_END = H.TRAIN_END                          # "202312"
VAL_MONTHS = 24                                  # champion purge window
TAU = 0.10                                       # routing temperature
TAU_DIM = 0.15                                   # dimension-membership temperature
D_TEXT = 4                                       # PCA dims per stream
L1_LAMBDA = 1e-3
SEEDS = (42, 7, 2024)
EPOCHS = 100

# ---------------------------------------------------------------------------
# the 23 retail categories -- codes, names, hierarchy, text prototypes
# ---------------------------------------------------------------------------
CAT_ROWS = [
    ("2",  "Food, alcoholic drinks and tobacco (other than supermarkets)", None),
    ("3",  "Fishery, livestock and poultry, fresh or frozen", "2"),
    ("4",  "Fruits and vegetables, fresh", "2"),
    ("5",  "Bread, pastry, confectionery and biscuits", "2"),
    ("6",  "Other food not elsewhere classified", "2"),
    ("7",  "Alcoholic drinks and tobacco", "2"),
    ("8",  "Supermarkets", None),
    ("9",  "Fuels", None),
    ("12", "Clothing, footwear and allied products", None),
    ("13", "Wearing apparel", "12"),
    ("14", "Footwear, allied products and other clothing accessories", "12"),
    ("18", "Consumer durable goods", None),
    ("19", "Motor vehicles and parts", "18"),
    ("23", "Furniture and fixtures", "18"),
    ("51", "Electrical goods and other consumer durables nec", "18"),
    ("30", "Department stores", None),
    ("32", "Jewellery, watches and clocks, and valuable gifts", None),
    ("35", "Other consumer goods", None),
    ("36", "Books, newspapers, stationery and gifts", "35"),
    ("37", "Chinese drugs and herbs", "35"),
    ("38", "Optical shops", "35"),
    ("39", "Medicines and cosmetics", "35"),
    ("40", "Other consumer goods not elsewhere classified", "35"),
]

# Semantic anchors.  Deliberately written the way a GovHK / news sentence about
# that category would read, in English, because the encoder space is English.
CAT_PROTOTYPES = {
    "2":  "Food, alcoholic drinks and tobacco, groceries, fresh provisions, "
          "rice and cooking oil prices, food safety recall",
    "3":  "Fresh fish, livestock, poultry, frozen pork and beef supply, "
          "wet market meat prices, seafood imports",
    "4":  "Fresh fruits and vegetables supply, vegetable prices, imported produce, "
          "agricultural wholesale market",
    "5":  "Bakery, bread, pastry, confectionery, biscuits, cakes, cake shops",
    "6":  "Processed food, sauces, instant noodles, dairy products, eggs, "
          "other food not elsewhere classified",
    "7":  "Alcoholic drinks, beer, wine and spirits, liquor licence, "
          "tobacco and cigarettes, duty on alcohol",
    "8":  "Supermarkets, daily groceries, household food staples, chain grocery "
          "store, everyday shopping basket",
    "9":  "Fuel, petrol, diesel, LPG, liquefied petroleum gas, motor fuel prices, "
          "electricity and gas tariffs",
    "12": "Clothing, footwear and allied products, apparel retail, fashion shops, "
          "garment sales, shoes and accessories",
    "13": "Wearing apparel, fashion clothing, garments, uniforms, textile and "
          "clothing retail",
    "14": "Footwear, shoes, sneakers, handbags, other clothing accessories, "
          "leather goods",
    "18": "Consumer durable goods, big ticket household appliances, durable "
          "household equipment, electronics purchase",
    "19": "Motor vehicles, car registration, private car sales, fuel, transport, "
          "electric vehicle first registration tax",
    "23": "Furniture and fixtures, sofa, beds, home furnishing, interior "
          "decoration, renovation",
    "51": "Electrical goods, household appliances, refrigerators, air "
          "conditioners, washing machines, consumer electronics",
    "30": "Department stores,百货 company, multi-brand department store counters, "
          "cosmetics counters and fashion floors",
    "32": "Jewellery, watches, clocks, luxury gifts, gold price, mainland tourist "
          "shopping, jewellery and watch retail",
    "35": "Other consumer goods, miscellaneous personal and household goods retail",
    "36": "Books, newspapers, stationery, magazines, greeting cards and gifts, "
          "book fair",
    "37": "Chinese drugs and herbs, traditional Chinese medicine, herbal tea, "
          "dried seafood and ginseng",
    "38": "Optical shops, spectacles, contact lenses, eye examination, eyewear",
    "39": "Medicines and cosmetics, pharmacy, skincare, beauty products, "
          "personal care and toiletries",
    "40": "Other consumer goods not elsewhere classified, pet supplies, toys, "
          "sports goods, musical instruments",
}

CAT_NAMES = {c: n for c, n, _ in CAT_ROWS}
CAT_PARENT = {c: p for c, _, p in CAT_ROWS}

# ---------------------------------------------------------------------------
# the four orthogonal semantic dimensions (innovation A)
# ---------------------------------------------------------------------------
DIM_KEYS = ["macro", "tourism", "durables", "staples"]
DIM_LABEL = {
    "macro":    "Dim1 Macro & Policy",
    "tourism":  "Dim2 Tourism & Cross-Border",
    "durables": "Dim3 Durables, Auto & Property",
    "staples":  "Dim4 Staples & Department Retail",
}
DIM_PROTOTYPES = {
    "macro": "Interest rate, inflation, consumer price index, tax, budget, "
             "consumption voucher, government policy, border control, "
             "unemployment, economy and GDP",
    "tourism": "Visitor arrivals, tourists, inbound travel, flights and airport, "
               "border crossing, mainland shoppers, hotel occupancy, shopping "
               "experience and travel reviews",
    "durables": "Property market, housing transactions and mortgage, car "
                "registration, motor vehicle, electrical appliances, furniture, "
                "renovation, big ticket durable purchase",
    "staples": "Supermarket, food prices, fresh food supply, seasonal promotion, "
               "weather and typhoon, daily consumption, department store and "
               "grocery basket",
}

# ---------------------------------------------------------------------------
# dimension lexicon -- the ADMISSION GATE and the membership prior.
#
# WHY A LEXICON IS NEEDED ON TOP OF THE EMBEDDING
# -----------------------------------------------
# The GovHK daily archive is mostly generic government notices ("Fatal traffic
# accident in Kowloon Bay", "Entries invited for drawing competition").  A
# 384-d MiniLM cosine alone assigns 50% of that corpus to whichever dimension
# prototype happens to share a stop-word flavour, which makes the "four
# orthogonal streams" claim false in practice.  Measured: with a pure-cosine
# assignment only 26% of headline/dimension pairs worth keeping survive.
#
# So a headline is ADMITTED to the router only when it carries at least one
# dimension keyword (15.2% of the 42,594 headlines do), and the membership
# p_{i,k} mixes the keyword count with the cosine score:
#
#     score_{i,k} = kw_{i,k}/0.5 + <e_i, D_k>/0.30
#
# A headline with no hit contributes NOTHING to any routed embedding, which is
# the honest treatment of "a government press release about a drawing
# competition" -- it is not weak evidence, it is no evidence.
# ---------------------------------------------------------------------------
DIM_LEXICON = {
    "macro": (r"interest rate|inflation|consumer price|composite cpi|\bcpi\b|"
              r"\btax\b|taxation|budget|rates concession|voucher|unemploy|\bgdp\b|"
              r"econom|monetary|exchange rate|exchange fund|fiscal|deficit|"
              r"subsid|relief|recession|stimulus|retail sales|gross domestic"),
    "tourism": (r"visitor|tourist|tourism|inbound|arrival|travell?er|flight|"
                r"airport|airline|hotel|occupancy|cruise|shopper|cross-boundary|"
                r"cross border|border|immigration|holiday|itinerary|travel agent"),
    "durables": (r"propert|mortgage|housing|private flat|apartment|\bcars?\b|"
                 r"vehicle|motor|registration|appliance|electrical|furniture|"
                 r"renovat|land sale|developer|real estate|furnishing|"
                 r"air conditioner|refrigerator"),
    "staples": (r"supermarket|grocery|groceries|\bfood\b|vegetable|pork|beef|"
                r"poultry|\bfish\b|rice|cooking oil|beverage|typhoon|weather|"
                r"rainfall|temperature|department store|cosmetic|toiletr|"
                r"fresh food|market stall|price of"),
}
DIM_RE = {k: re.compile(v, re.I) for k, v in DIM_LEXICON.items()}
TAU_KW = 0.5          # keyword-count scale in the membership prior
TAU_COS = 0.30        # cosine scale in the membership prior

# sentiment lexicon (same words as news_features.py so the two are comparable)
POS = ["rise", "rises", "rose", "up", "growth", "grow", "grows", "grew", "increase",
       "increases", "record", "surge", "surges", "boost", "boosts", "improve",
       "improves", "improved", "recovery", "recovers", "rebound", "rebounds",
       "strong", "stronger", "gain", "gains", "expand", "expands", "high",
       "higher", "welcome", "welcomes", "launch", "launches", "success",
       "successful", "approve", "approves", "approved", "support", "supports"]
NEG = ["fall", "falls", "fell", "down", "drop", "drops", "dropped", "decline",
       "declines", "declined", "decrease", "decreases", "slump", "slumps",
       "weak", "weaker", "loss", "losses", "cut", "cuts", "low", "lower",
       "concern", "concerns", "warning", "warns", "protest", "protests",
       "strike", "strikes", "accident", "injured", "death", "deaths", "fraud",
       "arrest", "arrests", "seized", "recall", "recalls", "deficit", "debt"]
POS_RE = re.compile(r"\b(" + "|".join(POS) + r")\b", re.I)
NEG_RE = re.compile(r"\b(" + "|".join(NEG) + r")\b", re.I)


# ===========================================================================
# PART A/B -- text -> category-routed feature tensor
# ===========================================================================
def load_headlines() -> pd.DataFrame:
    d = pd.read_csv(DATA / "news_headlines.csv",
                    dtype={"date": str, "month": str, "title": str})
    d["month"] = d["month"].astype(str)
    return d


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.clip(n, 1e-9, None)


def encode_titles(titles: list[str], cache: Path) -> np.ndarray:
    """MiniLM embeddings of every headline, cached (embeddings are per-text, so
    caching them cannot leak: the causality rule is about which TEXTS we read)."""
    if cache.exists():
        E = np.load(cache)
        if E.shape[0] == len(titles):
            print(f"  [cache] headline embeddings {E.shape} <- {cache.name}")
            return E
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    E = model.encode(titles, batch_size=256, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
    E = E.astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    print(f"  encoded {len(titles):,} headlines -> {E.shape} in {time.time()-t0:.1f}s ({dev})")
    return E


def encode_prototypes() -> tuple[np.ndarray, np.ndarray]:
    """(23 x 384) category prototypes, (4 x 384) dimension prototypes."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cats = [CAT_PROTOTYPES[c] for c, _, _ in CAT_ROWS]
    dims = [DIM_PROTOTYPES[k] for k in DIM_KEYS]
    P = model.encode(cats + dims, batch_size=32, show_progress_bar=False,
                     convert_to_numpy=True).astype(np.float32)
    return _l2(P[:len(cats)]), _l2(P[len(cats):])


def _sentiment(titles: list[str]) -> np.ndarray:
    out = np.zeros(len(titles), dtype=np.float32)
    for i, t in enumerate(titles):
        p = len(POS_RE.findall(t))
        q = len(NEG_RE.findall(t))
        out[i] = (p - q) / (p + q) if (p + q) else 0.0
    return out


def _routing_tensor(master: pd.DataFrame, periods: list[str], series: list[str],
                    Pcat: np.ndarray, Pdim: np.ndarray,
                    keep: np.ndarray | None = None):
    """Raw (un-fitted) routed quantities.

    Returns
    -------
    raw  : (P, S, 4, 384)  E_{t,j,k}, the category's own text embedding
    ints : (P, S, 4, 3)    routed sentiment, routed salience, dimension volume
    route: (4, 23)         mean routing mass per (dimension, category)
    """
    keep = np.ones(len(master), dtype=bool) if keep is None else keep
    # admission gate: a headline that carries no dimension keyword is not weak
    # evidence about any stream, it is no evidence.
    keep = keep & (master["kw_total"].to_numpy() > 0)
    ms = master.loc[keep].reset_index(drop=True)
    V = _l2(np.asarray(list(ms["emb"].to_numpy()), dtype=np.float32))   # (n,384)
    sent = ms["sent"].to_numpy(np.float32)
    KW = ms[[f"kw_{k}" for k in DIM_KEYS]].to_numpy(np.float32)         # (n,4)

    nP, nS, nK, nV = len(periods), len(series), len(DIM_KEYS), V.shape[1]
    raw = np.zeros((nP, nS, nK, nV), dtype=np.float32)
    ints = np.zeros((nP, nS, nK, 3), dtype=np.float32)
    route = np.zeros((nK, nS), dtype=np.float64)
    adm = np.zeros(nP, dtype=np.int32)
    n_rows = 0

    month_idx = {m: i for i, m in enumerate(periods)}
    mi = np.array([month_idx.get(m, -1) for m in ms["month"]], dtype=np.int64)

    # NOTE: every matmul below is PER MONTH, not one big (n x 384) product.
    # BLAS may partition a large product differently when the number of rows
    # changes, which perturbs the last bits of a row even when that row's input
    # is identical.  Per-month products keep each month's arithmetic exactly
    # reproducible under a headline mask, which is what makes the causality
    # test below able to demand a difference of EXACTLY 0.0.
    for t in range(nP):
        sel = np.where(mi == t)[0]
        if len(sel) == 0:
            continue
        adm[t] = len(sel)
        Vt = V[sel]
        Sc = Vt @ Pcat.T                                 # (m, 23) category match
        Sd = Vt @ Pdim.T                                 # (m, 4)  dimension match
        sal = np.clip(Sc.max(axis=1), 0.0, 1.0).astype(np.float32)
        Z = KW[sel] / TAU_KW + Sd / TAU_COS              # blended membership
        Z -= Z.max(axis=1, keepdims=True)
        Pd = np.exp(Z)
        Pd /= Pd.sum(axis=1, keepdims=True)              # (m, 4) p_{i,k}
        W = np.exp((Sc - Sc.max(axis=0, keepdims=True)) / TAU)   # softmax OVER i
        W /= W.sum(axis=0, keepdims=True)                        # (m, 23)
        st, sa = sent[sel], sal
        for k in range(nK):
            wk = W * Pd[:, [k]]                          # (m, 23) routed mass
            tot = wk.sum(axis=0)                         # (23,)
            safe = np.clip(tot, 1e-8, None)
            raw[t, :, k, :] = (wk.T @ Vt) / safe[:, None]
            ints[t, :, k, 0] = (wk.T @ st) / safe        # routed sentiment
            ints[t, :, k, 1] = (wk.T @ sa) / safe        # routed salience
            ints[t, :, k, 2] = float(np.log1p(Pd[:, k].sum()))   # dimension volume
            route[k] += tot
            n_rows += 1
    route /= max(n_rows, 1)
    return raw, ints, route, adm


def routed_features(master: pd.DataFrame, periods: list[str], series: list[str],
                    keep: np.ndarray | None = None,
                    fitted: dict | None = None,
                    verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Build the (P, S, 28) category-routed feature tensor.

    master : rows = headlines, columns = [month, title, sent, emb]
    keep   : boolean mask of headlines to READ (the prefix-mask test uses it)
    fitted : frozen {"Pcat","Pdim"} (+ optionally {"pca","scaler"}); when the
             latter two are present they are REUSED, which is what makes the
             causality test compare text content only -- re-fitting the PCA on a
             truncated table would move the projection and make the comparison
             meaningless.
    """
    fitted = {} if fitted is None else fitted
    train_mask = np.array([p <= TRAIN_END for p in periods])
    raw, ints, route, adm = _routing_tensor(master, periods, series,
                                            fitted["Pcat"], fitted["Pdim"],
                                            keep=keep)

    nP, nS = raw.shape[0], raw.shape[1]
    feats = np.zeros((nP, nS, len(DIM_KEYS) * 7), dtype=np.float32)
    pca_info = {}
    from sklearn.decomposition import PCA
    for k, key in enumerate(DIM_KEYS):
        R = raw[:, :, k, :].reshape(-1, raw.shape[3])           # (P*S, 384)
        tr = raw[train_mask, :, k, :].reshape(-1, raw.shape[3])
        if fitted.get("pca") is None or key not in fitted["pca"]:
            pca = PCA(n_components=D_TEXT, random_state=0).fit(tr)
            fitted.setdefault("pca", {})[key] = pca
        else:
            pca = fitted["pca"][key]
        feats[:, :, k * 7:k * 7 + 4] = pca.transform(R).reshape(
            nP, nS, D_TEXT).astype(np.float32)
        feats[:, :, k * 7 + 4:k * 7 + 7] = ints[:, :, k, :]
        pca_info[key] = {"explained_variance_ratio":
                         [round(float(x), 4) for x in pca.explained_variance_ratio_]}

    if fitted.get("scaler") is None:
        mu = np.zeros(feats.shape[2], dtype=np.float64)
        sd = np.ones(feats.shape[2], dtype=np.float64)
        for c in range(feats.shape[2]):
            col = feats[train_mask, :, c]
            m, s = float(col.mean()), float(col.std(ddof=0))
            if not np.isfinite(s) or s == 0.0:
                s = 1.0
            mu[c], sd[c] = m, s
        fitted["scaler"] = {"mu": mu, "sd": sd}
    mu, sd = fitted["scaler"]["mu"], fitted["scaler"]["sd"]
    feats = (feats - mu[None, None, :]) / sd[None, None, :]
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    info = {"pca": pca_info,
            "scaler_mu": [round(float(v), 6) for v in mu],
            "scaler_sd": [round(float(v), 6) for v in sd],
            "admitted_headlines_per_month": {"min": int(adm[adm > 0].min()),
                                            "median": int(np.median(adm[adm > 0])),
                                            "max": int(adm.max()),
                                            "months_with_zero": int((adm == 0).sum())},
            "route_weight_mean": route.round(6).tolist()}
    return feats, info


COLNAMES = [f"d{k+1}_{s}" for k in range(len(DIM_KEYS))
            for s in ("e1", "e2", "e3", "e4", "sent", "sal", "vol")]


def prepare_master() -> pd.DataFrame:
    """headlines + embeddings + lexicon sentiment + dimension keyword counts.

    Duplicate (month, title) rows are collapsed: the archive re-publishes some
    notices verbatim (e.g. the daily "Effective Exchange Rate Index"), and an
    un-deduplicated row would otherwise collect 5x the routing mass of an
    equally-relevant headline that happened to be published once.
    """
    d = load_headlines()
    n0 = len(d)
    d = d.drop_duplicates(subset=["month", "title"]).reset_index(drop=True)
    titles = d["title"].astype(str).tolist()
    E = encode_titles(titles, MD / "headline_emb.npy")
    master = pd.DataFrame({"month": d["month"].to_numpy(), "title": titles,
                           "sent": _sentiment(titles)})
    master["emb"] = list(E)
    for k in DIM_KEYS:
        master[f"kw_{k}"] = [len(DIM_RE[k].findall(t)) for t in titles]
    master["kw_total"] = master[[f"kw_{k}" for k in DIM_KEYS]].sum(axis=1)
    print(f"  headlines: {n0:,} -> {len(master):,} unique (month, title); "
          f"{int((master['kw_total'] > 0).sum()):,} "
          f"({100*(master['kw_total'] > 0).mean():.1f}%) carry a dimension keyword")
    return master


def build_all(force: bool = False, verbose: bool = True) -> dict:
    """Fit the routing/ PCA / scaler on history and write the feature tensor."""
    MD.mkdir(parents=True, exist_ok=True)
    out_npz = MD / "routed_features.npz"
    if out_npz.exists() and not force:
        z = np.load(out_npz, allow_pickle=True)
        if verbose:
            print(f"  [cache] {out_npz.name}  E={z['E'].shape}")
        return {"E": z["E"], "periods": list(z["periods"]), "series": list(z["series"]),
                "info": json.loads(str(z["info"]))}

    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    periods, series, F, Y, KF = X.build_tensors(feat)
    master = prepare_master()
    Pcat, Pdim = encode_prototypes()
    fitted = {"Pcat": Pcat, "Pdim": Pdim}
    t0 = time.time()
    E, info = routed_features(master, periods, series, fitted=fitted, verbose=verbose)
    info["fit_sec"] = round(time.time() - t0, 1)
    info["tau"] = TAU
    info["tau_dim"] = TAU_DIM
    info["d_text"] = D_TEXT
    info["encoder"] = "sentence-transformers/all-MiniLM-L6-v2"
    info["pca_fit_period_max"] = TRAIN_END
    info["columns"] = COLNAMES
    info["n_headlines"] = int(len(master))
    np.savez_compressed(out_npz, E=E, periods=np.array(periods),
                        series=np.array(series), info=json.dumps(info))
    (MD / "routed_meta.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"  routed tensor {E.shape}  ({len(periods)} months x {len(series)} "
              f"series x {E.shape[2]} features)   {info['fit_sec']}s")
    return {"E": E, "periods": periods, "series": series, "info": info}


def flat_features(E: np.ndarray) -> np.ndarray:
    """Control: identical width, NO routing.  Every category receives the
    cross-category mean of the routed block, i.e. dimension-split text with the
    category-specific part averaged away."""
    m = E.mean(axis=1, keepdims=True)                # (P, 1, 28)
    return np.repeat(m, E.shape[1], axis=1)


def lumped_features(periods: list[str], series: list[str]) -> np.ndarray:
    """The old Stage-3 representation, re-expressed as a series-specific cube:
    the four month-level PCA columns broadcast to all 23 categories.

    Exists so that the 'old lumped text' arm can be re-run through the SAME
    (horizon-gated) model class as the new arms.  Without it, the old-vs-new
    comparison would also change the injection point and the parameter count,
    and the routing claim would be confounded with both.
    """
    lump = pd.read_csv(DATA / "news_text_wide.csv", index_col=0, dtype={0: str})
    lump.index = lump.index.astype(str)
    cols = [f"news_pc{i+1}" for i in range(4)]
    pc = lump.reindex(periods)[cols].fillna(0.0).to_numpy(np.float32)
    return np.repeat(pc[:, None, :], len(series), axis=1)


def dimension_stats(verbose: bool = True) -> dict:
    """How much of the corpus each semantic dimension owns, with examples.

    Written to `stage3/results/multidim_dimension_stats.csv` so the report can
    quote a measured number instead of an impression.
    """
    master = prepare_master()
    Pcat, Pdim = encode_prototypes()
    adm = (master["kw_total"].to_numpy() > 0)
    ms = master.loc[adm].reset_index(drop=True)
    V = _l2(np.asarray(list(ms["emb"].to_numpy()), dtype=np.float32))
    KW = ms[[f"kw_{k}" for k in DIM_KEYS]].to_numpy(np.float32)
    Z = KW / TAU_KW + (V @ Pdim.T) / TAU_COS
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(axis=1, keepdims=True)                 # (n, 4) membership
    hard = P.argmax(axis=1)
    rows = [{"dimension": "ALL", "label": "admitted (>=1 keyword)",
             "headlines": int(adm.sum()),
             "share_pct": round(100.0 * adm.mean(), 2),
             "soft_mass_pct": 100.0}]
    for k, key in enumerate(DIM_KEYS):
        m = hard == k
        rows.append({"dimension": key, "label": DIM_LABEL[key],
                     "headlines": int(m.sum()),
                     "share_pct": round(100.0 * m.sum() / max(adm.sum(), 1), 2),
                     "soft_mass_pct": round(100.0 * float(P[:, k].sum() / len(P)), 2)})
        if verbose:
            print(f"  {key:9s} {int(m.sum()):6d} headlines "
                  f"({100*m.sum()/max(adm.sum(),1):5.2f}% of admitted, "
                  f"{100*m.sum()/len(master):5.2f}% of corpus)")
            for e in ms.loc[m, "title"].head(3).tolist():
                print(f"      e.g. {e[:96]}")
    by_year = (pd.DataFrame({"year": ms["month"].str[:4],
                             "dim": [DIM_KEYS[i] for i in hard]})
               .groupby(["year", "dim"]).size().unstack(fill_value=0))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "multidim_dimension_stats.csv", index=False)
    by_year.to_csv(RES / "multidim_dimension_by_year.csv")
    # routing sanity: which dimension does each category lean on most
    meta = json.loads((MD / "routed_meta.json").read_text(encoding="utf-8"))
    R = np.array(meta["route_weight_mean"], dtype=float)
    lift = R / R.mean(axis=1, keepdims=True)
    lean = {c: DIM_KEYS[int(np.argmax(lift[:, j]))]
            for j, (c, _, _) in enumerate(CAT_ROWS)}
    if verbose:
        print("  strongest dimension per category (lift): "
              + "  ".join(f"{c}:{lean[c]}" for c in ["8", "2", "30", "32", "12",
                                                     "19", "23", "51", "9"]))
    return {"rows": rows, "lean": lean}


# ===========================================================================
# PART B (cont.) -- causality: prefix-mask test
# ===========================================================================
def prefix_mask_test(cut: str = "202212", verbose: bool = True) -> dict:
    """Re-derive the whole feature tensor from a DATE-TRUNCATED headline table.

    The fitted objects (PCA, z-scores) are deliberately the ones fit on
    period <= 202312: masking out later headlines must not move them, and the
    in-window feature values must be BIT-IDENTICAL.  Any non-zero difference
    means a future headline reached a past month.
    """
    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    periods, series, F, Y, KF = X.build_tensors(feat)
    master = prepare_master()
    Pcat, Pdim = encode_prototypes()

    f1 = {"Pcat": Pcat, "Pdim": Pdim}
    E_full, _ = routed_features(master, periods, series, fitted=f1, verbose=False)
    # the SAME fitted PCA / scaler objects, now applied to a truncated table:
    # only the text content differs between the two runs.
    frozen = {"Pcat": Pcat, "Pdim": Pdim,
              "pca": f1["pca"], "scaler": f1["scaler"]}
    keep = (master["month"] <= cut).to_numpy()
    E_mask, _ = routed_features(master, periods, series, keep=keep, fitted=frozen,
                                verbose=False)
    win = np.array([p <= cut for p in periods])
    d = np.abs(E_full[win] - E_mask[win]).max()
    d_full = np.abs(E_full - E_mask)
    later_changed = int((d_full[~win].max(axis=(1, 2)) > 0).sum())
    res = {"cut": cut, "n_headlines_kept": int(keep.sum()),
           "n_headlines_total": int(len(keep)), "n_months_in_window": int(win.sum()),
           "max_abs_diff_in_window": float(d),
           "months_after_cut_that_changed": later_changed,
           "PASS": bool(d == 0.0)}
    if verbose:
        print("PREFIX-MASK CAUSALITY TEST")
        for k, v in res.items():
            print(f"  {k:34s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ===========================================================================
# PART C1 -- xLSTM: Category-Routed Horizon FiLM (+ L1 gate sparsity)
# ===========================================================================
def build_tensors_routed(feat: pd.DataFrame, E: np.ndarray,
                         standardize: bool = True):
    """Like xlstm_exog.build_tensors_exog, but the exogenous block is
    SERIES-SPECIFIC: row (t, j) of E belongs to series j only.

    That is the whole point of routing -- the old block was (P, 1, d) broadcast
    over 23 series, so every category saw the same text numbers.
    """
    periods, series, F, Y, KF = X.build_tensors(feat)
    n_numeric = F.shape[2] - 1
    assert E.shape[0] == len(periods) and E.shape[1] == len(series), E.shape
    R = np.nan_to_num(E.astype(np.float64), nan=0.0)
    if standardize:
        tr = np.array([p <= TRAIN_END for p in periods])
        mu = R[tr].mean(axis=(0, 1), keepdims=True)
        sd = R[tr].std(axis=(0, 1), keepdims=True)
        sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
        R = (R - mu) / sd
    R = R.astype(np.float32)
    F2 = np.concatenate([F[:, :, :n_numeric], R, F[:, :, n_numeric:]], axis=2)
    assert F2.shape[2] == n_numeric + E.shape[2] + 1, F2.shape

    # ---- audit (same standard as xlstm_exog.build_tensors_exog) -----------
    # The numeric block legitimately contains NaN in the lag warm-up months, so
    # "no NaN anywhere" is the wrong test.  What must hold is that the ROUTED
    # block does not cost one otherwise-usable origin.
    ctx = X.CONTEXT

    def usable(FF):
        return {i for i in range(len(periods))
                if i - ctx + 1 >= 0
                and not np.isnan(FF[i - ctx + 1:i + 1]).any()}

    ok_num, ok_all = usable(F), usable(F2)
    lost = sorted(ok_num - ok_all)
    if lost:
        raise ValueError(
            f"the routed block would drop {len(lost)} otherwise-usable origin(s): "
            f"{[periods[i] for i in lost]}")
    return periods, series, F2, Y, KF


def train_routed(F, Y, KF, periods, series, cutoff_period, open_heads=(3, 6, 12),
                 l1=L1_LAMBDA, epochs=EPOCHS, lr=5e-4, batch=128,
                 weight_decay=5e-3, val_months=VAL_MONTHS, seed=42,
                 verbose=True):
    """`xlstm_direct.train_model`, byte-for-byte except for ONE added term:

        loss = MSE + l1 * mean(|W_exog_gate|)

    (L1 sparsity on the horizon gate).  With l1=0 the two functions are
    identical, which `verify_loop_equivalence()` checks numerically instead of
    asking the reader to trust the copy.
    """
    ctx = X.CONTEXT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pidx = {p: i for i, p in enumerate(periods)}
    cut = pidx[cutoff_period]
    first = ctx - 1
    val_from = cut - val_months
    val_end = cut - max(X.HEADS)                    # purged
    tr_i = list(range(first, val_from))
    va_i = list(range(val_from, val_end + 1))
    if not va_i:
        raise ValueError("purge left no validation origins")
    sidx = list(range(len(series)))
    Xtr, Str, Ctr, Ttr = X.make_samples(tr_i, sidx, F, Y, KF, context=ctx)
    Xva, Sva, Cva, Tva = X.make_samples(va_i, sidx, F, Y, KF, context=ctx)
    n_numeric = len(H.FEATURE_COLS)
    n_exog = F.shape[2] - 1 - n_numeric

    X.set_seed(seed)
    model = XE.XLSTMForecasterFiLMHorizon(
        len(series), F.shape[2] - 1, n_numeric, n_exog,
        open_heads=tuple(open_heads), context=ctx, use_anchor=True,
        use_calendar=False).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  train samples {len(Xtr):,}  val {len(Xva):,}  device {dev}")
        print(f"    origins train {periods[tr_i[0]]}..{periods[tr_i[-1]]}   "
              f"val {periods[va_i[0]]}..{periods[va_i[-1]]}")
        print(f"    params {n_par:,}   exog cols {n_exog}   open heads {list(open_heads)}"
              f"   l1 {l1}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t = torch.from_numpy(Xtr); Str_t = torch.from_numpy(Str)
    Ctr_t = torch.from_numpy(Ctr); Ttr_t = torch.from_numpy(Ttr)
    Xva_t = torch.from_numpy(Xva).to(dev); Sva_t = torch.from_numpy(Sva).to(dev)
    Cva_t = torch.from_numpy(Cva).to(dev); Tva_t = torch.from_numpy(Tva).to(dev)
    n = len(Xtr_t)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb = Xtr_t[idx].to(dev); sb = Str_t[idx].to(dev)
            cb = Ctr_t[idx].to(dev); yb = Ttr_t[idx].to(dev)
            opt.zero_grad()
            loss = ((model(xb, sb, cb) - yb) ** 2).mean()
            if l1 and l1 > 0:
                loss = loss + l1 * model.exog_gate.weight.abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = float(((model(Xva_t, Sva_t, Cva_t) - Tva_t) ** 2).mean())
        if vloss < best:
            best, best_ep = vloss, ep
            best_state = {k2: v.detach().cpu().clone()
                          for k2, v in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 25 == 0 or ep == epochs):
            print(f"    ep {ep:3d}  train {tot / n:.4f}  val {vloss:.4f}"
                  f"{'  *' if ep == best_ep else ''}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    aw = model.anchor_weights()
    gv = model.gate_values()                     # (n_heads, n_series, embed)
    gate_abs = np.abs(gv)
    meta = {"n_params": int(n_par), "best_val_loss": float(best),
            "best_epoch": int(best_ep), "n_train_samples": int(len(Xtr)),
            "n_val_samples": int(len(Xva)), "open_heads": list(open_heads),
            "closed_heads": [h for h in X.HEADS if h not in open_heads],
            "l1": float(l1), "n_exog": int(n_exog),
            "anchor_weights": {str(h): float(aw[k]) for k, h in enumerate(X.HEADS)},
            "gate_absmean_by_head": {str(h): float(gate_abs[k].mean())
                                     for k, h in enumerate(X.HEADS)},
            "gate_sparsity_by_head": {str(h): float((gate_abs[k] < 1e-2).mean())
                                      for k, h in enumerate(X.HEADS)},
            "train_origins": [periods[tr_i[0]], periods[tr_i[-1]]],
            "val_origins": [periods[va_i[0]], periods[va_i[-1]]]}
    if verbose:
        print(f"  best val {best:.4f} @ ep {best_ep}   anchor "
              + "  ".join(f"h{h}={aw[k]:.3f}" for k, h in enumerate(X.HEADS)))
        print("  gate |mean| by head: "
              + "  ".join(f"h{h}={meta['gate_absmean_by_head'][str(h)]:.4f}"
                          for h in X.HEADS)
              + "   (closed heads are structurally 0)")
    return model, meta


def verify_loop_equivalence(verbose: bool = True) -> dict:
    """`train_routed(l1=0)` must reproduce `train_model(model_factory=...)`
    EXACTLY.  Otherwise the 'only difference is the text block' claim is false."""
    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    d = build_all(verbose=False)
    periods, series, F2, Y, KF = build_tensors_routed(feat, d["E"])
    n_numeric = len(H.FEATURE_COLS)
    n_exog = F2.shape[2] - 1 - n_numeric

    def factory(n_series, n_feat_total):
        return XE.XLSTMForecasterFiLMHorizon(
            n_series, n_feat_total, n_numeric, n_exog, open_heads=(3, 6, 12),
            context=X.CONTEXT, use_anchor=True, use_calendar=False)

    kw = dict(epochs=3, verbose=False)
    m1, _ = X.train_model(F2, Y, KF, periods, series, TRAIN_END,
                          purge_val=True, model_factory=factory, seed=123, **kw)
    m2, _ = train_routed(F2, Y, KF, periods, series, TRAIN_END, l1=0.0,
                         seed=123, **kw)
    p1 = X.predict_at(m1, F2, KF, periods, series, H.ORIGINS[0])
    p2 = X.predict_at(m2, F2, KF, periods, series, H.ORIGINS[0])
    dmax = float(max(np.abs(p1[h] - p2[h]).max() for h in X.HEADS))
    res = {"max_abs_diff_reference_vs_copy": dmax,
           "identical": bool(dmax == 0.0), "PASS": bool(dmax == 0.0)}
    if verbose:
        print("TRAINING-LOOP EQUIVALENCE (l1=0 copy vs xlstm_direct.train_model)")
        for k, v in res.items():
            print(f"  {k:34s} {v}")
    return res


def fit_xlstm(arm: str, E: np.ndarray, seed: int, open_heads: tuple, l1: float,
              verbose: bool = True):
    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(STAGE0 / "data" / "panel_wide.csv", index_col=0)
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(STAGE0 / "data" / "scaling.csv", dtype={"series": str})
    periods, series, F2, Y, KF = build_tensors_routed(feat, E)
    tag = f"{arm}_s{seed}"
    print("\n" + "=" * 78, flush=True)
    print(f"  {tag}   cube {F2.shape}   open_heads={open_heads}  l1={l1}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    model, meta = train_routed(F2, Y, KF, periods, series, TRAIN_END,
                               open_heads=open_heads, l1=l1, seed=seed,
                               verbose=verbose)
    rows = []
    for o in H.ORIGINS:
        pr = X.predict_at(model, F2, KF, periods, series, o)
        if pr is None:
            continue
        for h in X.HEADS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            yhat = H.inverse_transform(pr[h], np.asarray(series), scaling)
            for s, yp in zip(series, yhat):
                rows.append((tag, o, h, s, float(wide.loc[tgt, s]), float(yp)))
    df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                     "y_true", "y_pred"])
    meta.update({"tag": tag, "arm": arm, "seed": seed, "family": "xlstm",
                 "fit_sec": round(time.time() - t0, 1),
                 "n_exog_cols": int(E.shape[2])})
    PRED.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED / f"{tag}.csv", index=False)
    (META / f"{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  saved {tag}: {len(df):,} rows, {meta['fit_sec']}s", flush=True)
    return df, meta


# ===========================================================================
# PART C2 -- iTransformer with multimodal TEXT-MODALITY tokens
# ===========================================================================
class ITransformerTextToken(IT.ITransformerNet):
    """23 sales variate tokens + N_TEXT text-modality tokens in ONE inverted
    self-attention.  The encoder trunk is untouched, so the extra tokens cost
    ZERO trunk parameters; only the (small) text-token projection is new.

    Token content for dimension k is

        [ routed embedding (D_TEXT), routed sentiment, routed salience,
          dimension volume,  routing weights over the 23 categories ]

    so a sales token can read BOTH what the dimension is about and how strongly
    that dimension is routed to itself.  The last block is what makes the token
    category-aware without a per-category token.
    """

    def __init__(self, context: int = 12, d_tok: int = 30, n_text: int = 4, **kw):
        super().__init__(context=context, **kw)
        self.n_text = int(n_text)
        self.text_proj = nn.Linear(int(d_tok), self.d_model)
        nn.init.zeros_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    def forward(self, Z: torch.Tensor, T: torch.Tensor | None = None) -> torch.Tensor:
        B, S, L = Z.shape
        if self.instance_norm:
            mu = Z.mean(dim=-1, keepdim=True)
            sd = Z.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            Zn = (Z - mu) / sd
        else:
            sd = torch.ones_like(Z[:, :, -1:])
            Zn = Z
        tok = self.embed(Zn)                                  # (B, S, d)
        if T is not None and self.n_text:
            tok = torch.cat([tok, self.text_proj(T)], dim=1)  # (B, S+4, d)
        h = self._encode(tok)[:, :S]                          # sales tokens only
        delta = self.proj(h) * sd
        return self.anchor_terms(Z) + delta


def it_origin_plan(periods: list[str], F: np.ndarray, context: int = 12):
    """identical to run_sota_baselines.origin_plan (purged, VAL_MONTHS=24)"""
    pidx = {p: i for i, p in enumerate(periods)}
    cut = pidx[TRAIN_END]
    val_end = cut - max(H.HORIZONS)
    val_from = cut - VAL_MONTHS
    usable = {}
    for i in range(len(periods)):
        lo = i - context + 1
        if lo < 0:
            continue
        usable[i] = not bool(np.isnan(F[lo:i + 1]).any())
    tr = [i for i in range(context - 1, val_from) if usable.get(i, False)]
    va = [i for i in range(val_from, val_end + 1) if usable.get(i, False)]
    if not va:
        raise ValueError("purge left no validation origins")
    assert periods[min(va[-1] + max(H.HORIZONS), len(periods) - 1)] <= TRAIN_END
    return tr, va


def build_text_tokens(E: np.ndarray, periods: list[str], series: list[str],
                      mode: str = "routed") -> np.ndarray:
    """(P, 4, 30) per-month text tokens.

    mode="routed" : dimension-k content + its routing weights
    mode="pooled" : the OLD lumped representation -- one monthly vector, no
                    dimension split and no routing, so the four tokens are
                    IDENTICAL and carry zeros in the routing block
    """
    nP, nS, nC = E.shape
    tok = np.zeros((nP, 4, 7 + nS), dtype=np.float32)
    if mode == "pooled":
        # reuse the stage-3 lumped PCA text columns (period-broadcast)
        lump = pd.read_csv(DATA / "news_text_wide.csv", index_col=0,
                           dtype={0: str})
        lump.index = lump.index.astype(str)
        pc = lump.reindex(periods)[[f"news_pc{i+1}" for i in range(4)]] \
                 .fillna(0.0).to_numpy(np.float32)
        sc = lump.reindex(periods)[["news_log_count"]].fillna(0.0) \
                .to_numpy(np.float32)[:, 0]
        for k in range(4):
            tok[:, k, 0:4] = pc
            tok[:, k, 4] = 0.0
            tok[:, k, 5] = 0.0
            tok[:, k, 6] = sc
            tok[:, k, 7:] = 0.0
        return tok
    # routed: per-dimension content, mean over categories, + routing weights
    for k in range(4):
        blk = E[:, :, k * 7:(k + 1) * 7]                     # (P, S, 7)
        tok[:, k, 0:7] = blk.mean(axis=1)
        tok[:, k, 7:] = blk[:, :, 5]                         # routed salience block
    # normalise the routing block per (month, dim) so it sums to 1
    w = np.clip(tok[:, :, 7:], 0.0, None)
    s = w.sum(axis=2, keepdims=True)
    tok[:, :, 7:] = w / np.clip(s, 1e-8, None)
    return tok


def train_it_text(model, Z: np.ndarray, T: np.ndarray, tr_i, va_i,
                  context: int = 12, epochs: int = EPOCHS, lr: float = 5e-4,
                  weight_decay: float = 5e-3, batch: int = 32,
                  seed: int = 42, verbose: bool = True):
    """`itransformer.train_transformer` plus a text-token tensor."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    if T is None:
        T = np.zeros((len(Z), 0), dtype=np.float32)
    Wtr, Ytr = IT.build_windows(Z, tr_i, context)
    Wva, Yva = IT.build_windows(Z, va_i, context)
    Ttr = torch.tensor(T[tr_i], dtype=torch.float32)
    Tva = torch.tensor(T[va_i], dtype=torch.float32)
    n = len(Wtr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Wva_t, Yva_t = Wva.to(dev), Yva.to(dev)
    Tva_t = Tva.to(dev) if model.n_text else None
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb, yb = Wtr[idx].to(dev), Ytr[idx].to(dev)
            tb = Ttr[idx].to(dev) if model.n_text else None
            opt.zero_grad()
            out = model(xb, tb) if model.n_text else model(xb)
            loss = ((out - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vo = model(Wva_t, Tva_t) if model.n_text else model(Wva_t)
            vloss = float(((vo - Yva_t) ** 2).mean())
        if vloss < best:
            best, best_ep = vloss, ep
            best_state = {k2: t.detach().clone() for k2, t in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 25 == 0 or ep == epochs):
            print(f"    ep {ep:3d}  train {tot / n:.4f}  val {vloss:.4f}"
                  f"{'  *' if ep == best_ep else ''}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {"best_val_loss": float(best), "best_epoch": int(best_ep),
            "n_train_origins": int(n), "n_val_origins": int(len(Wva)),
            "n_params": int(sum(p.numel() for p in model.parameters()
                                if p.requires_grad)),
            "trunk_params": int(sum(p.numel() for p in model.encoder.parameters())
                                + sum(p.numel() for p in model.embed.parameters())
                                + sum(p.numel() for p in model.proj.parameters())),
            "n_tokens": int(Z.shape[1] + getattr(model, "n_text", 0)),
            "n_series_tokens": int(Z.shape[1])}
    if hasattr(model, "anchor_weights"):
        meta["anchor_weights"] = {str(h): float(w) for h, w in
                                  zip(IT.HEADS, model.anchor_weights())}
    return model, meta


def fit_it(arm: str, E: np.ndarray, seed: int, mode: str, n_text: int,
           verbose: bool = True):
    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(STAGE0 / "data" / "panel_wide.csv", index_col=0)
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(STAGE0 / "data" / "scaling.csv", dtype={"series": str})
    periods, series, F, Y, KF = X.build_tensors(feat)
    Z = Y.astype(np.float64)
    tr_i, va_i = it_origin_plan(periods, F)
    T = build_text_tokens(E, periods, series, mode=mode) if n_text else None
    d_tok = T.shape[2] if n_text else 0
    tag = f"{arm}_s{seed}"
    print("\n" + "=" * 78, flush=True)
    print(f"  {tag}   series={len(series)}  text tokens={n_text}  d_tok={d_tok}  "
          f"tokens/sample={len(series)+n_text}  train origins {len(tr_i)}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if n_text:
        model = ITransformerTextToken(context=12, d_tok=d_tok, n_text=n_text,
                                      d_model=32, n_heads_attn=4, layers=2,
                                      dropout=0.1, use_anchor=True)
    else:
        model = IT.ITransformerNet(context=12, d_model=32, n_heads_attn=4,
                                   layers=2, dropout=0.1, use_anchor=True)
        model.n_text = 0
    model, meta = train_it_text(model, Z, T, tr_i, va_i, seed=seed, verbose=verbose)
    rows = []
    for o in H.ORIGINS:
        i = periods.index(o)
        W = torch.tensor(Z[i - 11:i + 1, :].T[None, :, :], dtype=torch.float32)
        Tw = torch.tensor(T[i][None], dtype=torch.float32) if n_text else None
        dev = next(model.parameters()).device
        with torch.no_grad():
            if n_text:
                out = model(W.to(dev), Tw.to(dev))
            else:
                out = model(W.to(dev))
            lv = out[0].cpu().numpy()
        for k, h in enumerate(H.HORIZONS):
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            yhat = H.inverse_transform(lv[:, k], np.asarray(series), scaling)
            for s, yp in zip(series, yhat):
                rows.append((tag, o, h, s, float(wide.loc[tgt, s]), float(yp)))
    df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                     "y_true", "y_pred"])
    meta.update({"tag": tag, "arm": arm, "seed": seed, "family": "itransformer",
                 "fit_sec": round(time.time() - t0, 1), "text_mode": mode})
    PRED.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED / f"{tag}.csv", index=False)
    (META / f"{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  saved {tag}: {len(df):,} rows, {meta['fit_sec']}s, "
          f"params {meta['n_params']:,} (trunk {meta['trunk_params']:,})", flush=True)
    return df, meta


# ===========================================================================
# PART D -- arms, ENS3 metrics, ablation table
# ===========================================================================
ARMS = {
    # ---- (a) no text at all ------------------------------------------------
    "xlstm_none":      {"family": "xlstm", "kind": "reuse", "cell": "none"},
    "it_none":         {"family": "itransformer", "kind": "fit", "n_text": 0,
                        "mode": None},
    # ---- (b) the OLD, one-lump-for-all-23 text -----------------------------
    "xlstm_pooled":    {"family": "xlstm", "kind": "reuse", "cell": "text_emb"},
    "it_pooled":       {"family": "itransformer", "kind": "fit", "n_text": 4,
                        "mode": "pooled"},
    # ---- (c) multi-dimensional text ---------------------------------------
    "xlstm_flat":      {"family": "xlstm", "kind": "fit", "exog": "flat",
                        "open_heads": (1, 3, 6, 12), "l1": L1_LAMBDA},
    "xlstm_routed":    {"family": "xlstm", "kind": "fit", "exog": "routed",
                        "open_heads": (3, 6, 12), "l1": L1_LAMBDA},
    # ---- controls that isolate the two design choices ----------------------
    "xlstm_lumped_hz": {"family": "xlstm", "kind": "fit", "exog": "pooled4",
                        "open_heads": (1, 3, 6, 12), "l1": L1_LAMBDA},
    "xlstm_routed_nol1": {"family": "xlstm", "kind": "fit", "exog": "routed",
                          "open_heads": (3, 6, 12), "l1": 0.0},
    "xlstm_routed_allH": {"family": "xlstm", "kind": "fit", "exog": "routed",
                          "open_heads": (1, 3, 6, 12), "l1": L1_LAMBDA},
    "it_routed":       {"family": "itransformer", "kind": "fit", "n_text": 4,
                        "mode": "routed"},
}

ARM_ORDER = ["xlstm_none", "xlstm_pooled", "xlstm_lumped_hz", "xlstm_flat",
             "xlstm_routed", "xlstm_routed_nol1", "xlstm_routed_allH",
             "it_none", "it_pooled", "it_routed"]

ARM_LABEL = {
    "xlstm_none":         "(a) xLSTM  no text",
    "xlstm_pooled":       "(b) xLSTM  Stage-3 old lumped text (bottom-injected PCA4)",
    "xlstm_lumped_hz":    "(b2) same lumped text, horizon-gated (same class as ours)",
    "xlstm_flat":         "(c3) 4-dim text, NO routing (broadcast)",
    "xlstm_routed":       "(c1) 4-dim text + routing, h=1 mask (pre-specified)",
    "xlstm_routed_nol1":  "control: (c1) with L1 = 0",
    "xlstm_routed_allH":  "(c) 4-dim text + routing, all heads open (OURS)",
    "it_none":            "(a') iTransformer 23 tokens, no text",
    "it_pooled":          "(b') iTransformer +4 identical old-text tokens",
    "it_routed":          "(c') iTransformer +4 routed dim tokens (OURS)",
}


def _reuse_pooled_arms() -> dict:
    """Pull the stage-3 'none' and 'text_emb' cells out of the existing
    predictions file so those arms cost nothing to re-run."""
    f = RES / "stage3_predictions_all.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype={"series": str, "origin": str})
    out = {}
    for cell, arm in (("none", "xlstm_none"), ("text_emb", "xlstm_pooled")):
        g = d[d["cell"] == cell].copy()
        if g.empty:
            continue
        for seed, gg in g.groupby("seed"):
            out[f"{arm}_s{int(seed)}"] = gg[["model", "origin", "h", "series",
                                             "y_true", "y_pred"]].copy()
    return out


def load_arm_preds(arm: str) -> pd.DataFrame:
    """concatenate the per-seed prediction files of one arm"""
    frames = []
    for seed in SEEDS:
        f = PRED / f"{arm}_s{seed}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, dtype={"series": str, "origin": str}))
    if frames:
        return pd.concat(frames, ignore_index=True)
    reuse = _reuse_pooled_arms()
    frames = [v for k, v in reuse.items() if k.startswith(arm + "_s")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ens3(df: pd.DataFrame) -> pd.DataFrame:
    """seed ensemble = mean of the per-seed predictions (same recipe as ENS3)"""
    k = ["origin", "h", "series"]
    agg = df.groupby(k, as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    agg["model"] = "ENS3"
    return agg[["model", "origin", "h", "series", "y_true", "y_pred"]]


def report(verbose: bool = True) -> pd.DataFrame:
    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(STAGE0 / "data" / "panel_wide.csv", index_col=0)
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)

    rows = []
    for arm in ARM_ORDER:
        d = load_arm_preds(arm)
        if d.empty:
            print(f"  [missing] {arm}")
            continue
        for seed in sorted(d["model"].unique()):
            one = H.metrics_from_frame(d[d["model"] == seed].assign(model=arm),
                                       naive_mae)
            for _, r in one.iterrows():
                rows.append({"arm": arm, "family": ARMS[arm]["family"],
                             "seed": seed, "h": r["h"], "MASE": r["MASE"],
                             "WAPE": r["WAPE"], "RMSE": r["RMSE"], "MAE": r["MAE"],
                             "bias_pct": r["bias_pct"], "n_points": r["n_points"]})
        e = ens3(d)
        one = H.metrics_from_frame(e.assign(model=arm), naive_mae)
        for _, r in one.iterrows():
            rows.append({"arm": arm, "family": ARMS[arm]["family"],
                         "seed": "ENS3", "h": r["h"], "MASE": r["MASE"],
                         "WAPE": r["WAPE"], "RMSE": r["RMSE"], "MAE": r["MAE"],
                         "bias_pct": r["bias_pct"], "n_points": r["n_points"]})
    m = pd.DataFrame(rows)
    out = RES / "multidim_text_results.csv"
    m.to_csv(out, index=False)
    if verbose:
        e = m[m["seed"].astype(str) == "ENS3"]
        piv = e.pivot_table(index="arm", columns="h", values="MASE")
        piv = piv.reindex([a for a in ARM_ORDER if a in piv.index])
        # NOTE: `h` is stored as int64 with the string sentinel "ALL" mixed in, so
        # pandas builds a mixed-type column index (ints + str).  Comparing that
        # index against the strings "1","3",… silently selected nothing but
        # "ALL" and the printed table lost four of its five columns.  Normalise
        # the axis to str before selecting.
        piv.columns = piv.columns.astype(str)
        cols = [c for c in ["1", "3", "6", "12", "ALL"] if c in piv.columns]
        if len(cols) != 5:
            raise RuntimeError(f"horizon columns not found in pivot: {list(piv.columns)}")
        print("\n" + "=" * 100)
        print("  MASE by arm x horizon   (ENS3, lower is better)")
        print("=" * 100)
        print(piv[cols].round(5).to_string())
        print("\n  WAPE (ALL):")
        w = e[e["h"].astype(str) == "ALL"].set_index("arm")["WAPE"]
        print("  " + "  ".join(f"{a}={w[a]:.5f}" for a in ARM_ORDER if a in w.index))

        # ---- the required (a) vs (b) vs (c) table -------------------------
        print("\n" + "=" * 100)
        print("  ABLATION: (a) no text  vs  (b) stage-3 lumped  vs  (c) ours")
        print("=" * 100)
        for fam, (a, b, c) in {"xlstm": ("xlstm_none", "xlstm_pooled", "xlstm_routed"),
                               "itransformer": ("it_none", "it_pooled", "it_routed")}.items():
            sub = e[e["arm"].isin([a, b, c])]
            if sub.empty:
                continue
            t = sub.pivot_table(index="arm", columns="h", values="MASE")
            t = t.reindex([a, b, c])
            t["Δ ours-(a)"] = t["ALL"] - t.loc[a, "ALL"]
            t["Δ ours-(b)"] = t["ALL"] - t.loc[b, "ALL"]
            print(f"\n  --- {fam} ---")
            print(t.round(5).to_string())
        print(f"\n  wrote {out}")
    return m


# ===========================================================================
# driver
# ===========================================================================
def run_arms(arms: list[str], seeds=SEEDS, verbose: bool = True):
    d = build_all(verbose=verbose)
    E = d["E"]
    flats = {"flat": flat_features(E),
             "pooled4": lumped_features(d["periods"], d["series"])}
    for arm in arms:
        cfg = ARMS[arm]
        if cfg["kind"] == "reuse":
            print(f"  [reuse] {arm} <- stage3 cells")
            continue
        for seed in seeds:
            tag = f"{arm}_s{seed}"
            if (PRED / f"{tag}.csv").exists():
                print(f"  [skip] {tag}")
                continue
            if cfg["family"] == "xlstm":
                src = E if cfg["exog"] == "routed" else flats[cfg["exog"]]
                fit_xlstm(arm, src, seed, tuple(cfg["open_heads"]), cfg["l1"],
                          verbose=verbose)
            else:
                fit_it(arm, E, seed, cfg["mode"], cfg["n_text"], verbose=verbose)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["build", "verify", "stats", "fit", "report", "all"])
    ap.add_argument("--arms", nargs="+", default=ARM_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    v = not a.quiet

    if a.stage in ("build", "all"):
        print("=" * 78)
        print("  STAGE 1  multi-dimensional text streams + category routing")
        print("=" * 78)
        build_all(force=a.force, verbose=v)
    if a.stage in ("stats", "all"):
        print("\n" + "=" * 78)
        print("  STAGE 2b  semantic-dimension composition of the corpus")
        print("=" * 78)
        dimension_stats(verbose=v)
    if a.stage in ("verify", "all"):
        print("\n" + "=" * 78)
        print("  STAGE 2  leakage + equivalence checks")
        print("=" * 78)
        prefix_mask_test(verbose=v)
        verify_loop_equivalence(verbose=v)
    if a.stage in ("fit", "all"):
        print("\n" + "=" * 78)
        print("  STAGE 3  fitting arms")
        print("=" * 78)
        run_arms(a.arms, seeds=tuple(a.seeds), verbose=v)
    if a.stage in ("report", "all"):
        print("\n" + "=" * 78)
        print("  STAGE 4  ablation table")
        print("=" * 78)
        report(verbose=v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
