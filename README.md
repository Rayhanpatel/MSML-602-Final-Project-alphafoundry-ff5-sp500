# Factor-Based Portfolio to Outperform S&P 500 (Lean)

## Project Description
A reproducible factor-based investment strategy that uses rolling-window regression to estimate Fama–French 5-factor loadings for S&P 500 constituents and forecasts returns based on trailing factor exposures. A walk-forward backtest evaluates performance using Sharpe ratio (on excess returns) and cumulative returns, benchmarked against SPY. We also compare two ML models (Logistic Regression vs XGBoost learning-to-rank) using the same feature set and walk-forward rules.

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

### Run (submission notebook)
For submission/grading, run the single merged notebook:

1. Open `one.ipynb`
2. Run **Kernel: Restart & Run All**

`one.ipynb` contains:
- Data curation
- EDA (distributions, outliers, correlations)
- Baseline FF5 rolling strategy backtest (vs SPY)
- Two-model ML comparison (Logistic Regression vs XGBoost learning-to-rank)

### GitHub Pages (final tutorial)
This repository includes a GitHub Pages-ready tutorial landing page in `docs/index.md`.

Final tutorial URL:

- https://rayhanpatel.github.io/MSML-602-Final-Project-alphafoundry-ff5-sp500/

To publish:

1. Push this repository to GitHub.
2. In GitHub: Settings > Pages > Build and deployment.
3. Set **Source** to Deploy from a branch.
4. Set **Branch** to `main` and **Folder** to `/docs`.

### Run order (Jupyter notebooks)
For submission, run `one.ipynb` only.

## Data inputs and outputs

### Generated files
The project uses cached CSVs in `data/raw/`. The following files should exist in `data/raw/`:

- `ff5_data.csv`
- `market_data.csv`
- `sp500.csv`
- `spy_monthly_returns.csv`

`one.ipynb` defaults to **cached/offline mode** (`USE_CACHED_DATA = True`). If a required CSV is missing, temporarily set `USE_CACHED_DATA = False` to allow downloading and regenerating the files.

### Expected outputs
Running `one.ipynb` generates figures used by the GitHub Pages tutorial in `docs/assets/`, including:

- `strategy_cumulative_returns.png`
- `strategy_drawdowns.png`
- `strategy_rolling_12m_returns.png`
- `strategy_cumulative_returns_xgb_compare.png`

Note: although `market_data.csv` begins in 2005 in a typical run, the backtest in `one.ipynb` will start later due to rolling-window “burn-in” requirements (e.g., 36-month betas and a lagged factor forecast).

## Method summary

- **Model:** Rolling-window OLS regression using the Fama-French 5 factors plus risk-free rate (monthly).
- **Returns:** Uses **excess returns** (asset return minus `RF`).
- **Forecasting (no look-ahead):** Expected factor returns are formed using only information available up to time `t-1` (rolling historical estimates), and then applied to the estimated factor loadings to produce a one-period-ahead return signal.
- **Portfolio:** Long-only, equal-weighted selection of the top-ranked stocks by predicted return (rebalanced monthly).
- **Benchmark:** SPY monthly returns.

## Limitations

- **Survivorship bias:** The constituent list is sourced from the present-day S&P 500 (Wikipedia). This can materially bias historical backtests upward because removed/delisted names are missing.
- **Data availability:** IPOs, delistings, and missing price history can create gaps and reduce the effective sample.
- **Transaction costs:** The default notebook results are **gross returns** (transaction costs disabled by default). The strategy notebook also includes a simple turnover-based **cost sensitivity** check to show how results change under non-zero bps assumptions.

## Submission checklist
- Run `one.ipynb` with **Restart & Run All** and confirm it completes without errors.
- Ensure `data/raw/` includes the 4 CSVs listed above.
- If submitting a ZIP, exclude `.git/`, `.venv/`, and `.DS_Store`.
