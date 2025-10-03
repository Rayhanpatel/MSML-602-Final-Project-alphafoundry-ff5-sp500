# Factor-Based Portfolio to Outperform S&P 500 (Lean)

## Project Topic
Long-only, monthly factor strategy using Fama–French 5 (FF5) to target outperformance vs SPY.

## Team Members
- Rayhan Basheer Patel		| UDI: 122087934
- Govind Singahl		    	| UDI: 117780413
<!-- - Chaithanya Sai Musalreddy	| UDI: 122257672 -->

## Datasets (with sources)
- **Fama–French 5 Factors + RF (Monthly, 1963–Aug 2025)**  
  Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **S&P 500 equities & SPY** (prices & dividends)  
  Via `yfinance`: https://pypi.org/project/yfinance/
- **Latest-month FF5 replication (required for this project)**  
  Tutorial: https://dilequante.com/replicate-fama-french-5-factor-model-from-publicly-available-data-sources/

## Why these datasets
- **Relevance:** FF5 is the standard academic factor set for expected-return modeling; SPY/S&P 500 universe matches the benchmarked goal.
- **Scale & history:** 60+ years of monthly factors and deep US equity coverage enable robust train/test windows.
- **Accessibility & reproducibility:** Public, well-documented sources; `yfinance` provides straightforward adjusted prices/dividends; latest-month FF5 replication ensures timely evaluation.

