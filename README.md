# Factor-Based Portfolio to Outperform S&P 500 (Lean)

## Project Topic
Long-only, monthly factor strategy using Fama–French 5 (FF5) to target outperformance vs SPY.

## Team Members
- Rayhan Basheer Patel	  	| UDI: 122087934
- Govind Singahl		    	  | UDI: 117780413
- Chaithanya Sai Musalreddy | UDI: 122257672
  
## Datasets (with sources)
We use **two datasets** (within course limit):
1) **Fama–French 5 Factors + RF (Monthly, 1963–Aug 2025)**  
   Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  
   **Freshness (required, preprocessing on FF5):** The latest month isn’t posted, we **replicate that month** using:  
   https://dilequante.com/replicate-fama-french-5-factor-model-from-publicly-available-data-sources/  
   Replicated rows are **tagged `synthetic=true`**.
2) **S&P 500 equities & SPY** (prices & dividends) via `yfinance`  
   https://pypi.org/project/yfinance/

*Reference only (not used in modeling):* JKP Global Factor Data — https://jkpfactors.com/

## Why these datasets
- **Fit for goal:** FF5 is the standard factor set; SPY/S&P 500 aligns with our benchmark.
- **Scale & history:** 60+ years of monthly factors and deep US equity coverage = robust train/test.
- **Reproducible & current:** Public sources, easy retrieval (`yfinance`), and latest-month FF5 replication keeps results up to date.

## How we combine & model (one paragraph)
We compute **monthly total returns** (adjusted prices/dividends) and convert to **excess returns** by subtracting RF. At each month-end, we **join on dates** (point-in-time), fit **36-month rolling OLS** of each stock’s excess return on FF5 to estimate exposures, and form a **forecast** using trailing factor means. We then **rank by predicted excess return**, buy the **Top-10** stocks **equal-weight**, and rebalance monthly with **10 bps** cost. Evaluation: annualized return/volatility, **Sharpe** (excess RF), **max drawdown**, annual returns, and turnover vs **SPY**.
