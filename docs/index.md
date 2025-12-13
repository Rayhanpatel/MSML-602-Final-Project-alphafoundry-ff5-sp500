---
layout: default
title: Factor-Based Portfolio to Outperform the S&P 500 (SPY Benchmark)
---

# Factor-Based Portfolio to Outperform the S&P 500 (SPY Benchmark)

## Overview
This tutorial builds a factor-based investing strategy using the Fama-French 5-factor model. We estimate rolling factor loadings (betas) for S&P 500 constituents, form a monthly return signal using only information available up to time `t-1`, and evaluate the strategy in a walk-forward backtest benchmarked against SPY.

## Repository entry points

### Notebooks
- `data-cleanup.ipynb`: downloads and prepares monthly returns data and benchmark data.
- `eda.ipynb`: exploratory analysis and statistical tests.
- `S&P500_FF5_Rolling_Strategy.ipynb`: rolling regression, return forecasting, portfolio construction, and backtest vs SPY.

### Data outputs
After running `data-cleanup.ipynb`, these files should exist in `data/raw/`:
- `ff5_data.csv`
- `market_data.csv`
- `sp500.csv`
- `spy_monthly_returns.csv`

## How to run

### Install
```bash
pip install -r requirements.txt
```

### Execute
Run the notebooks in this exact order (Kernel: Restart & Run All):
1. `data-cleanup.ipynb`
2. `eda.ipynb`
3. `S&P500_FF5_Rolling_Strategy.ipynb`

## Data sources
- Fama-French 5 Factors + RF: Ken French Data Library
  - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- S&P 500 equities and SPY prices: `yfinance`
  - https://pypi.org/project/yfinance/

## Method summary (high level)
- Rolling-window OLS (monthly) to estimate FF5 betas.
- Excess returns are used (return minus `RF`).
- Factor expectations for prediction are computed using only historical information up to time `t-1`.
- Monthly rebalance into an equal-weighted long-only portfolio of the top-ranked stocks by predicted return.
- Monthly timestamps are standardized to month-start to align data sources.
- Benchmark performance is computed using SPY monthly returns.

## Key results (from the latest notebook run)

Backtest window and headline metrics are printed by `S&P500_FF5_Rolling_Strategy.ipynb` after a full run.

- **Backtest window**
  - 2008-02 to 2025-10 (213 months)
- **Sharpe (excess returns, annualized)**
  - Strategy (gross): ~0.77
  - Strategy (net, 10 bps turnover cost): ~0.76
  - SPY: ~0.69
- **CAGR (total returns)**
  - Strategy (gross): ~16.3%
  - Strategy (net): ~16.1%
  - SPY: ~11.5%

### Plots (saved to `docs/assets/` by the strategy notebook)

If you re-run the strategy notebook, it will export these images automatically.

#### Cumulative returns
![Cumulative Returns: Strategy vs SPY](assets/strategy_cumulative_returns.png)

#### Drawdowns
![Drawdowns: Strategy vs SPY](assets/strategy_drawdowns.png)

#### Rolling 12-month returns
![Rolling 12-Month Returns](assets/strategy_rolling_12m_returns.png)

## Limitations
- Survivorship bias from constituent list construction.
- Missing data for IPOs/delistings and data gaps.
- Transaction costs and slippage are simplified (modeled as bps cost times portfolio turnover).

## References
- Ken French Data Library (Fama-French factors): https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- yfinance (Yahoo Finance data access): https://pypi.org/project/yfinance/
