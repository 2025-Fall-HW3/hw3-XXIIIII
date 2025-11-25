from Markowitz_2 import MyPortfolio, df, Bdf
import quantstats as qs
import pandas as pd

mp = MyPortfolio(df, "SPY")
mp_weights, mp_returns = mp.get_results()

Bmp = MyPortfolio(Bdf, "SPY")
Bmp_weights, Bmp_returns = Bmp.get_results()

print("=== Problem 4.1: Sharpe > 1 (2019-2024) ===")
df_returns = df.pct_change().fillna(0)
df_bl = pd.DataFrame()
df_bl["SPY"] = df_returns["SPY"]
df_bl["MP"] = pd.to_numeric(mp_returns["Portfolio"], errors="coerce")
sharpe = qs.stats.sharpe(df_bl)
print(f"SPY Sharpe: {sharpe[0]:.4f}")
print(f"MP Sharpe: {sharpe[1]:.4f}")
print(f"MP > 1? {sharpe[1] > 1}")
print(f"Weights sum check: {(mp_weights.sum(axis=1) <= 1.01).all()}")
print(f"Max weight sum: {mp_weights.sum(axis=1).max():.4f}")

print("\n=== Problem 4.2: Sharpe > SPY (2012-2024) ===")
Bdf_returns = Bdf.pct_change().fillna(0)
Bdf_bl = pd.DataFrame()
Bdf_bl["SPY"] = Bdf_returns["SPY"]
Bdf_bl["MP"] = pd.to_numeric(Bmp_returns["Portfolio"], errors="coerce")
Bsharpe = qs.stats.sharpe(Bdf_bl)
print(f"SPY Sharpe: {Bsharpe[0]:.4f}")
print(f"MP Sharpe: {Bsharpe[1]:.4f}")
print(f"MP > SPY? {Bsharpe[1] > Bsharpe[0]}")
print(f"Weights sum check: {(Bmp_weights.sum(axis=1) <= 1.01).all()}")
print(f"Max weight sum: {Bmp_weights.sum(axis=1).max():.4f}")

print("\n=== Current Parameters ===")
print(f"lookback: {mp.lookback}")
print(f"gamma: {mp.gamma}")
print("\n[TIP] Try: lookback=60, gamma=1.5")