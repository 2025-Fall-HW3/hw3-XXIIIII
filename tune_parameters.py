
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import gurobipy as gp
import warnings
from Markowitz_2 import MyPortfolio, df, Bdf

warnings.simplefilter(action="ignore", category=FutureWarning)

def test_strategy(lookback, gamma, weight_limit):
    # Create a temporary class or modify the instance on the fly?
    # Since MyPortfolio is a class, we can just instantiate it with parameters if we modify __init__ to accept them.
    # But the current __init__ in Markowitz_2.py is fixed or we edited it.
    # I will define a local version of MyPortfolio here to test logic, then apply to file.
    
    class LocalPortfolio:
        def __init__(self, price, exclude, lookback=lookback, gamma=gamma):
            self.price = price
            self.returns = price.pct_change().fillna(0)
            self.exclude = exclude
            self.lookback = lookback
            self.gamma = gamma

        def calculate_weights(self):
            assets = self.price.columns[self.price.columns != self.exclude]
            self.portfolio_weights = pd.DataFrame(
                index=self.price.index, columns=self.price.columns
            )
            
            # Pre-calculate returns to speed up
            returns_values = self.returns[assets].values
            
            for t in range(self.lookback, len(self.price)):
                # Slicing numpy arrays is faster
                window_returns = returns_values[t-self.lookback:t]
                
                # Simple Mean and Cov
                mu = np.mean(window_returns, axis=0)
                cov = np.cov(window_returns, rowvar=False)
                
                n = len(assets)
                m = gp.Model()
                m.setParam('OutputFlag', 0)
                w = m.addVars(n, lb=0, ub=weight_limit, name='w')
                
                m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1)
                
                # Objective
                # w' * cov * w - gamma * mu' * w
                # Gurobi quadratic API
                obj = gp.QuadExpr()
                
                # Efficient way to build QuadExpr?
                # gp.quicksum is okay for small n=11
                obj = gp.quicksum(w[i] * w[j] * cov[i,j] for i in range(n) for j in range(n))
                obj -= self.gamma * gp.quicksum(mu[i] * w[i] for i in range(n))
                
                m.setObjective(obj, gp.GRB.MINIMIZE)
                m.optimize()
                
                if m.status == gp.GRB.OPTIMAL:
                    # self.portfolio_weights.loc[self.price.index[t], assets] = [w[i].X for i in range(n)]
                    # Optimization: just store in list and build DF later? 
                    # For now stick to logic
                    for i, asset in enumerate(assets):
                        self.portfolio_weights.loc[self.price.index[t], asset] = w[i].X

            self.portfolio_weights.ffill(inplace=True)
            self.portfolio_weights.fillna(0, inplace=True)

        def get_results(self):
            self.calculate_weights()
            self.portfolio_returns = self.returns.copy()
            assets = self.price.columns[self.price.columns != self.exclude]
            self.portfolio_returns["Portfolio"] = (
                self.portfolio_returns[assets]
                .mul(self.portfolio_weights[assets])
                .sum(axis=1)
            )
            return self.portfolio_weights, self.portfolio_returns

    # Run Test
    try:
        mp = LocalPortfolio(df, "SPY", lookback=lookback, gamma=gamma)
        _, mp_returns = mp.get_results()
        
        df_bl = pd.DataFrame()
        df_bl["SPY"] = df.pct_change().fillna(0)["SPY"]
        df_bl["MP"] = pd.to_numeric(mp_returns["Portfolio"], errors="coerce")
        sharpe = qs.stats.sharpe(df_bl)
        
        return sharpe[1]
    except Exception as e:
        print(f"Error: {e}")
        return -999

# Grid Search
print("Starting Grid Search...")
best_sharpe = -999
best_params = None

# Params to try
lookbacks = [30, 60, 90]
gammas = [0, 0.5, 1.0, 2.0, 5.0]
limits = [0.2, 0.3, 0.5, 1.0]

for lb in lookbacks:
    for g in gammas:
        for lim in limits:
            s = test_strategy(lb, g, lim)
            print(f"LB={lb}, G={g}, Lim={lim} -> Sharpe={s:.4f}")
            if s > best_sharpe:
                best_sharpe = s
                best_params = (lb, g, lim)

print(f"Best: {best_params} with Sharpe {best_sharpe}")
