"""C+ Scoring Engine (Robust V2.4) - SENSEX + NIFTY 100, non-BFSI.
Hardened against Yahoo Finance API unit glitches:
  - EV/EBITDA sanity check + recompute from raw EV & EBITDA
  - FCF Yield unit-mismatch guard with neutral fallback
  - CCC fix for companies with no Cost Of Revenue row (services firms)
"""
import os, time, json, datetime as dt
import numpy as np, pandas as pd, yfinance as yf
from universe import fetch_universe

RISK_FREE, EQUITY_RISK_PREMIUM = 0.065, 0.070
FALLBACK_COST_OF_DEBT, FALLBACK_TAX, SLEEP = 0.09, 0.25, 1.5
BFSI_SECTOR = "Financial Services"

WEIGHTS = {
    "profit_yoy": .08, "profit_3y": .07, "sales_yoy": .05, "sales_3y": .05,
    "roce": .10, "roic_wacc": .07, "roe": .05, "opm": .03,
    "rel_pe": .07, "rel_ev": .05, "fcf_yield": .08,
    "ccc": .05, "fcf_sales": .05, "accruals": .05,
    "de": .04, "int_cov": .03, "pledge": .02, "momentum": .06,
}
HIGHER_BETTER = {"profit_yoy","profit_3y","sales_yoy","sales_3y","roce",
                 "roic_wacc","roe","opm","fcf_yield","fcf_sales","int_cov","momentum"}

def row(df, name):
    try:
        return df.loc[name].dropna() if name in df.index else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def ok(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))

def plausible(v, lo, hi):
    return ok(v) and lo <= v <= hi

def fmt_pct(v): return f"{v:.0%}" if ok(v) else "n/a"
def fmt_x(v):   return f"{v:.1f}x" if ok(v) else "n/a"

def fetch(sym, pledge_override):
    t = yf.Ticker(sym); info = t.info
    fin, bs, cf = t.financials, t.balance_sheet, t.cashflow
    rev, ni = row(fin,"Total Revenue"), row(fin,"Net Income")
    ebit = row(fin,"EBIT") if "EBIT" in fin.index else row(fin,"Operating Income")
    ebitda_s = row(fin,"EBITDA")
    intr, tax, pretax = row(fin,"Interest Expense"), row(fin,"Tax Provision"), row(fin,"Pretax Income")
    cogs = row(fin,"Cost Of Revenue")
    ocf, capex = row(cf,"Operating Cash Flow"), row(cf,"Capital Expenditure")
    eq, debt = row(bs,"Stockholders Equity"), row(bs,"Total Debt")
    cash, inv = row(bs,"Cash And Cash Equivalents"), row(bs,"Inventory")
    recv, pay = row(bs,"Receivables"), row(bs,"Payables And Accrued Expenses")
    assets = row(bs,"Total Assets")
    g = lambda s: float(s.iloc[0]) if len(s) else np.nan
    m = {}
    # --- Growth ---
    m["profit_yoy"] = ni.iloc[0]/ni.iloc[1]-1 if len(ni)>=2 else np.nan
    m["profit_3y"]  = (ni.iloc[0]/ni.iloc[3])**(1/3)-1 if len(ni)>=4 else np.nan
    m["sales_yoy"]  = rev.iloc[0]/rev.iloc[1]-1 if len(rev)>=2 else np.nan
    m["sales_3y"]   = (rev.iloc[0]/rev.iloc[3])**(1/3)-1 if len(rev)>=4 else np.nan
    # --- Profitability ---
    ce = g(debt)+g(eq)-g(cash)
    m["roce"] = g(ebit)/ce if ce else np.nan
    tax_rt = g(tax)/g(pretax) if g(pretax) else FALLBACK_TAX
    roic = g(ebit)*(1-tax_rt)/ce if ce else np.nan
    mcap = info.get("marketCap") or np.nan
    e, d = mcap or 0.0, g(debt) or 0.0
    re_ = RISK_FREE + (info.get("beta") or 1.0)*EQUITY_RISK_PREMIUM
    rd = (g(intr)/d) if d else FALLBACK_COST_OF_DEBT
    wacc = (e/(e+d)*re_ + d/(e+d)*rd*(1-tax_rt)) if (e+d) else re_
    m["roic_wacc"] = roic - wacc
    m["roe"] = g(ni)/g(eq) if g(eq) else np.nan
    m["opm"] = g(ebit)/g(rev) if g(rev) else np.nan
    # --- Valuation (HARDENED) ---
    m["pe"] = info.get("trailingPE") or np.nan
    evr = info.get("enterpriseToEbitda") or np.nan
    if not plausible(evr, 0, 60):                       # API glitch guard
        ev_raw, eb0 = info.get("enterpriseValue"), g(ebitda_s)
        evr = ev_raw/eb0 if (ev_raw and eb0 and eb0 > 0) else np.nan
        evr = evr if plausible(evr, 0, 60) else np.nan
    m["ev"] = evr
    fcf = np.nan
    if len(ocf):
        o, cx = g(ocf), g(capex)
        if ok(o): fcf = o + (0.0 if not ok(cx) else cx)
    if not ok(fcf): fcf = info.get("freeCashflow") or np.nan
    fy = fcf/mcap if (ok(fcf) and ok(mcap) and mcap) else np.nan
    if not plausible(fy, -0.20, 0.35):                  # unit-mismatch guard
        alt = (info.get("freeCashflow") or np.nan)/mcap if (ok(mcap) and mcap) else np.nan
        fy = alt if plausible(alt, -0.20, 0.35) else np.nan
    m["fcf_yield"] = fy
    # --- Cash flow & forensics (CCC FIXED for services firms) ---
    cogs_v = g(cogs)
    if not ok(cogs_v) or cogs_v <= 0: cogs_v = g(rev)
    m["ccc"] = (g(inv)/cogs_v*365 + g(recv)/g(rev)*365 - g(pay)/cogs_v*365) \
               if (ok(cogs_v) and ok(g(rev)) and g(rev) != 0) else np.nan
    m["fcf_sales"] = fcf/g(rev) if (ok(fcf) and ok(g(rev)) and g(rev) != 0) else np.nan
    m["accruals"] = (g(ni)-g(ocf))/g(assets) if (ok(g(ni)) and ok(g(ocf)) and ok(g(assets)) and g(assets)) else np.nan
    # --- Health / governance / momentum ---
    m["de"] = (g(debt)/g(eq)) if g(eq) else (info.get("debtToEquity", 0) or 0)/100
    m["int_cov"] = g(ebit)/g(intr) if (ok(g(intr)) and g(intr) > 0) else 999
    m["pledge"] = pledge_override
    try:
        px = t.history(period="1y")["Close"]
        m["momentum"] = px.iloc[-1]/px.iloc[-126]-1 if len(px) >= 126 else px.iloc[-1]/px.iloc[0]-1
    except Exception:
        m["momentum"] = np.nan
    m["sector"] = info.get("sector") or "Other"
    m["name"] = info.get("shortName") or sym
    return m

def main():
    uni = fetch_universe()
    rows = []
    for _, r in uni.iterrows():
        try:
            m = fetch(r.ticker, 0.0)
            m.update(ticker=r.ticker, in_n100=bool(r.in_n100), in_s30=bool(r.in_s30))
            rows.append(m); time.sleep(SLEEP)
        except Exception as ex:
            print(f"SKIP {r.ticker}: {ex}")
    df = pd.DataFrame(rows)
    df["bfsi"] = df.sector == BFSI_SECTOR
    live = df[~df.bfsi].copy()
    for col, src in [("rel_pe","pe"), ("rel_ev","ev")]:
        live[col] = live[src] / live.groupby("sector")[src].transform("median")
    for k in WEIGHTS:
        pct = live[k].rank(pct=True)
        live[k+"_s"] = (pct*100 if k in HIGHER_BETTER else (1-pct)*100).fillna(50).clip(0,100)
    live["c_plus"] = sum(live[k+"_s"]*w for k, w in WEIGHTS.items())
    live["val_pillar"] = (live.rel_pe_s*.07 + live.rel_ev_s*.05 + live.fcf_yield_s*.08)/.20
    live["recommendation"] = live.apply(
        lambda x: "SELL" if x.c_plus < 65 else
        ("BUY" if x.c_plus >= 80 and x.val_pillar >= 75 else "HOLD/WAIT"), axis=1)
    def reason(x):
        med = x.pe/x.rel_pe if (ok(x.rel_pe) and x.rel_pe != 0) else np.nan
        return (f"ROCE {fmt_pct(x.roce)}, P/E {fmt_x(x.pe)} vs sector {fmt_x(med)}, "
                f"FCF yld {fmt_pct(x.fcf_yield)}, D/E {x.de:.2f}")
    live["reason"] = live.apply(reason, axis=1)
    df = df.merge(live[["ticker","c_plus","recommendation","reason"]], on="ticker", how="left")
    df.loc[df.bfsi, ["recommendation","reason"]] = ["NOT SCORED (BFSI)", "BFSI engine (v3) pending"]
    df["badge"] = np.where(df.in_s30, "[S30·N100]", "[N100]")
    df = df.sort_values(["bfsi","c_plus"], ascending=[True, False])
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/c_plus_matrix.csv", index=False)
    json.dump({"updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
               "universe": "SENSEX + NIFTY 100 (non-BFSI engine)",
               "rows": df.replace({np.nan: None}).to_dict(orient="records")},
              open("output/c_plus_scores.json","w"))
    print(df[["ticker","badge","c_plus","recommendation"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
