# Factor-Based Portfolio to Outperform S&P 500 (Lean)

## Project Description
A reproducible factor-based investment strategy that uses rolling-window regression to estimate Fama–French 5-factor loadings for S&P 500 constituents and forecasts returns based on trailing factor exposures. A walk-forward backtest evaluates performance using Sharpe ratio (on excess returns) and cumulative returns, benchmarked against SPY.

## Team Members

- Rayhan Basheer Patel	  	| UDI: 122087934
- Govind Singahl		    	  | UDI: 117780413
- Chaithanya Sai Musalreddy | UDI: 122257672

## 2 Datasets with Sources
- **Fama–French 5 Factors + RF (Daily)** — Ken French Data Library  
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  
  We convert daily factors to monthly returns by compounding within each month.
  
- **S&P 500 equities & SPY** (prices, dividends) via **yfinance**  
  https://pypi.org/project/yfinance/

## 3 Why These Datasets (Brief Rationale)
- **Relevance:** FF5 is a standard factor set for expected-return modeling; SPY/S&P 500 align with our benchmark objective.  
- **Scale & history:** 60+ years of monthly factors and broad U.S. equity coverage enable robust train/test windows.  
- **Accessibility & reproducibility:** Public, well-documented sources; `yfinance` provides adjusted prices/dividends on a monthly grid.

## Setup

### Install dependencies
Create and activate a fresh Python environment, then install:

```bash
pip install -r requirements.txt
```

### GitHub Pages (final tutorial)
This repository includes a GitHub Pages-ready tutorial landing page in `docs/index.md`.

To publish:

1. Push this repository to GitHub.
2. In GitHub: Settings > Pages > Build and deployment.
3. Set **Source** to Deploy from a branch.
4. Set **Branch** to `main` and **Folder** to `/docs`.

### Run order (Jupyter notebooks)
Run the notebooks in this exact order (Kernel: Restart & Run All for each):

1. `data-cleanup.ipynb`
2. `eda.ipynb`
3. `S&P500_FF5_Rolling_Strategy.ipynb`

## Data inputs and outputs

### Generated files
After running `data-cleanup.ipynb`, the following files should exist in `data/raw/`:

- `ff5_data.csv`
- `market_data.csv`
- `sp500.csv`
- `spy_monthly_returns.csv`

## Method summary

- **Model:** Rolling-window OLS regression using the Fama-French 5 factors plus risk-free rate (monthly).
- **Returns:** Uses **excess returns** (asset return minus `RF`).
- **Forecasting (no look-ahead):** Expected factor returns are formed using only information available up to time `t-1` (rolling historical estimates), and then applied to the estimated factor loadings to produce a one-period-ahead return signal.
- **Portfolio:** Long-only, equal-weighted selection of the top-ranked stocks by predicted return (rebalanced monthly).
- **Benchmark:** SPY monthly returns.

## Limitations

- **Survivorship bias:** The constituent list may reflect present-day S&P 500 membership, which can bias historical backtests upward.
- **Data availability:** IPOs, delistings, and missing price history can create gaps and reduce the effective sample.
- **Transaction cost simplification:** Turnover-based costs are approximations and do not fully capture market impact or slippage.
