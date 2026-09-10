"""universe.py - keeps SENSEX + NIFTY 100 constituents fresh automatically."""
import pandas as pd

WIKI_N100   = "https://en.wikipedia.org/wiki/NIFTY_100"
WIKI_SENSEX = "https://en.wikipedia.org/wiki/BSE_SENSEX"

def _wiki_symbols(url):
    try:
        for tbl in pd.read_html(url):
            cols = [c.lower() for c in tbl.columns]
            for key in ("symbol", "ticker"):
                if key in cols:
                    col = tbl.columns[cols.index(key)]
                    return [s.strip().upper() for s in tbl[col].astype(str)]
    except Exception as e:
        print(f"WIKI FETCH FAILED {url}: {e}")
    return []

def fetch_universe():
    n100 = _wiki_symbols(WIKI_N100)
    s30  = _wiki_symbols(WIKI_SENSEX)
    if not n100:                      # offline fallback
        fb = pd.read_csv("fallback_nifty100.csv")
        n100 = fb["symbol"].tolist()
        s30  = fb.loc[fb.sensex, "symbol"].tolist()
    uni = sorted(set(n100) | set(s30))
    return pd.DataFrame({
        "symbol": uni,
        "ticker": [f"{s}.NS" for s in uni],
        "in_n100": [s in n100 for s in uni],
        "in_s30":  [s in s30 for s in uni],
    })
