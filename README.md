# Factor-Based Portfolio to Outperform S&P 500 (Lean)

## Project Description
A reproducible factor-based investment strategy that uses rolling-window regression to estimate Fama–French 5-factor loadings for S&P 500 constituents and forecasts returns based on trailing factor exposures. A walk-forward backtest spanning 2015–2025 evaluates performance using the Sharpe ratio and cumulative returns, benchmarked against SPY across multiple time horizons.

## Team Members
| Name                     | UDI        |
|--------------------------|------------|
| Rayhan Basheer Patel     | 122087934  |
| Govind Singahl           | 117780413  |
| Chaithanya Sai Musalreddy| 122257672  |

## 2 Datasets with Sources
- **Fama–French 5 Factors + RF (Monthly, 1963–Aug 2025)** — Ken French Data Library  
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  
  *Freshness:* The latest month isn’t posted, we replicate that month using the public proxy method and tag those rows as `synthetic=true`.  
  https://dilequante.com/replicate-fama-french-5-factor-model-from-publicly-available-data-sources/
  
- **S&P 500 equities & SPY** (prices, dividends) via **yfinance**  
  https://pypi.org/project/yfinance/

## 3 Why These Datasets (Brief Rationale)
- **Relevance:** FF5 is a standard factor set for expected-return modeling; SPY/S&P 500 align with our benchmark objective.  
- **Scale & history:** 60+ years of monthly factors and broad U.S. equity coverage enable robust train/test windows.  
- **Freshness:** The latest missing factor month is replicated via the cited proxy method and clearly flagged as synthetic.  
- **Accessibility & reproducibility:** Public, well-documented sources; `yfinance` provides adjusted prices/dividends on a monthly grid.
