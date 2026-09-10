"""universe.py v3 - committed CSV is source of truth; Wikipedia optional."""
import os
import pandas as pd

CSV_PATH = "universe_nifty100.csv"
WIKI_N100 = "https://en.wikipedia.org/wiki/NIFTY_100"
WIKI_SENSEX = "https://en.wikipedia.org/wiki/BSE_SENSEX"
UA = {"User-Agent": "CPlusScorer/1.0 (educational quant dashboard)"}

EMERGENCY_SENSEX30 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","BHARTIARTL","SBIN","LT",
    "ITC","AXISBANK","HCLTECH","MARUTI","SUNPHARMA","TITAN","ULTRACEMCO",
    "BAJFINANCE","NESTLEIND","M&M","ADANIENT","ADANIPORTS","TATAMOTORS",
    "POWERGRID","NTPC","ONGC","COALINDIA","JSWSTEEL","TATASTEEL","TECHM",
    "HINDALCO","BAJAJFINSV",
]

def _wiki_symbols(url):
    try:
        import requests
        html = requests.get(url, headers=UA, timeout=20).text
        for tbl in pd.read_html(html):
            cols = [str(c).lower() for c in tbl.columns]
            for key in ("symbol", "ticker"):
                if key in cols:
                    col = tbl.columns[cols.index(key)]
                    syms = [str(s).strip().upper() for s in tbl[col].dropna()]
                    syms = [s for s in syms if s and len(s) <= 15]
                    if len(syms) >= 20:
                        return syms
    except Exception as e:
        print(f"WIKI UNAVAILABLE (fine - using CSV): {e}")
    return []

def fetch_universe():
    if os.path.exists(CSV_PATH):
        fb = pd.read_csv(CSV_PATH)
        fb["symbol"] = fb["symbol"].str.upper().str.strip()
        n100 = fb["symbol"].tolist()
        s30 = fb.loc[fb["sensex"] == 1, "symbol"].tolist() if "sensex" in fb.columns else []
        w100, w30 = _wiki_symbols(WIKI_N100), _wiki_symbols(WIKI_SENSEX)
        if len(w100) >= 50: n100 = w100
        if len(w30) >= 20: s30 = w30
    else:
        n100 = _wiki_symbols(WIKI_N100) or EMERGENCY_SENSEX30
        s30 = _wiki_symbols(WIKI_SENSEX) or []
    if not s30:
        s30 = [s for s in EMERGENCY_SENSEX30 if s in n100] or EMERGENCY_SENSEX30
    uni = sorted(set(n100) | set(s30))
    return pd.DataFrame({
        "symbol": uni,
        "ticker": [f"{s}.NS" for s in uni],
        "in_n100": [s in n100 for s in uni],
        "in_s30":  [s in s30 for s in uni],
    })
