# ML Stock Trader

An ML-driven stock trading system that learns buy/sell/hold signals directly from price data using a custom-built ensemble learning pipeline. Rather than relying on hand-crafted rules, the system trains a model on technical indicators and historical returns, then evaluates its performance against both a manual rule-based strategy and a buy-and-hold benchmark.

---

## How It Works

### Feature Engineering — Technical Indicators

Three technical indicators are computed from raw price data and used as input features to the model:

**1. SMA Ratio (Momentum)**
The ratio of a short-window (5-day) simple moving average to a long-window (20-day) SMA. Values above 1.0 indicate short-term upward momentum; values below 1.0 indicate the reverse.

**2. Bollinger Bands %B (Mean Reversion)**
Measures where the current price sits relative to the upper and lower Bollinger Bands (SMA ± 2 standard deviations). A value near 0 suggests the price is near the lower band (potentially oversold); near 1 suggests it's near the upper band (potentially overbought).

**3. MACD Histogram (Trend)**
The difference between the MACD line (EMA12 − EMA26) and its 9-day signal line EMA. Positive values indicate bullish momentum; negative values indicate bearish momentum.

---

## Machine Learning Model

The core learner is a **Bagged Random Tree ensemble** — built from scratch without scikit-learn.

- **RandomTree**: A decision tree that randomly selects a feature at each split and uses the median value as the split threshold. Leaf size is constrained (minimum 12 samples) to prevent overfitting.
- **BootstrapAggregator**: Trains an ensemble of 12 RandomTree learners, each on a different bootstrap sample of the training data. Final predictions are made by majority vote (mode) across the ensemble.

This bagging approach reduces variance compared to a single tree and improves generalization to out-of-sample data.

---

## Training Labels

Labels are generated using a **3-day lookahead return**, adjusted for market impact:

| Condition | Label |
|---|---|
| Future 3-day return > +2% | `BUY` (+1) |
| Future 3-day return < −2% | `SELL` (−1) |
| Otherwise | `HOLD` (0) |

This frames the problem as a **3-class classification task**. The model learns to associate the current indicator state with the likely near-term price direction.

---

## Trading Constraints

The system operates under realistic trading constraints:
- Position is constrained to `{−1000, 0, +1000}` shares at any time (long, flat, or short)
- Valid trade sizes are `±1000` or `±2000` shares
- Each trade incurs a **commission** and **market impact** cost

---

## Experiments

To evaluate the ML strategy, two experiments were run using **JPM (JPMorgan Chase)** as an example stock. JPM was chosen for its liquidity and well-defined price trends, but the system is stock-agnostic and can be applied to any equity with available daily price data.

### Experiment 1 — ML Strategy vs. Manual Strategy vs. Benchmark

Compares three approaches across both in-sample (2008–2009) and out-of-sample (2010–2011) periods:

- **ML Strategy (StrategyLearner)**: Trained on in-sample data, signals generated from the learned model
- **Manual Strategy**: A hard-coded voting rule — buy if ≥2 of 3 indicators are bullish, sell if ≥2 are bearish, hold otherwise
- **Benchmark**: Buy 1000 shares on day 1 and hold through the entire period

All portfolio values are normalized to 1.0 at the start date for fair comparison.

**In-Sample (2008–2009)**

![In-Sample](images/experiment1_in_sample.png)

**Out-of-Sample (2010–2011)**

![Out-of-Sample](images/experiment1_out_of_sample.png)

### Experiment 2 — Impact of Market Impact on Strategy Behavior

Examines how varying the **market impact parameter** affects the ML strategy's trading behavior and returns. Higher impact costs discourage frequent trading, causing the strategy to become more selective. This experiment highlights the sensitivity of learned policies to transaction cost assumptions.

![Cumulative Returns](images/experiment2_cumulative_returns.png)

![Portfolio Values](images/experiment2_portfolio_values.png)

![Std Deviation](images/experiment2_std_deviation.png)

---

## Project Structure

```
ML_Stock_Trader/
├── RandomTree.py           # Custom random decision tree implementation
├── BootstrapAggregator.py  # Bagging ensemble over RandomTree learners
├── StrategyLearner.py      # Core ML trading strategy (train + test)
├── MLTrader.py             # Entry point / wrapper for StrategyLearner
├── ManualStrategy.py       # Rule-based trading strategy for comparison
├── indicators.py           # SMA ratio, Bollinger Bands %B, MACD histogram
├── marketsimcode.py        # Portfolio simulation and P&L computation
├── experiment1.py          # ML vs. Manual vs. Benchmark comparison
├── experiment2.py          # Market impact sensitivity analysis
├── testproject.py          # Runs the full pipeline end-to-end
└── images/                 # Output charts from experiments
```

---

## Running the Project

This project is part of the Georgia Tech ML4T framework and requires the shared `util.py` and `data/` directory from the course environment. Run from the **parent directory**:

```bash
# Run the full pipeline (ManualStrategy + StrategyLearner + both experiments)
python ML_Stock_Trader/testproject.py

# Or run experiments individually
python ML_Stock_Trader/experiment1.py
python ML_Stock_Trader/experiment2.py
```

Output charts are saved to `ML_Stock_Trader/images/`.

---

## Dependencies

- Python 3.x
- NumPy
- pandas
- matplotlib

---

*Code available upon request for recruiters and potential employers.*
