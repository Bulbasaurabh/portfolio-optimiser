# Asset Portfolio Optimizer

A Python-based desktop application for optimizing investment portfolios using Modern Portfolio Theory (MPT). Built with PyQt5, this tool calculates optimal asset allocations to maximize returns while managing risk.

## Features

- **Three Optimization Strategies:**
  - **Minimum Variance Portfolio**: Minimizes portfolio risk
  - **Efficient Portfolio with Constraint**: Optimizes for a target expected return
  - **Tangency Portfolio**: Maximizes the Sharpe ratio (risk-adjusted return)

- **Flexible Constraints**: Toggle between allowing or restricting negative weights (short positions)
- **Visual Analytics**: Interactive bar charts displaying portfolio weights
- **Real-time Data**: Fetches historical stock data via Yahoo Finance API
- **Portfolio Statistics**: Displays expected return, standard deviation, and Sharpe ratio

## Requirements

```
PyQt5
yfinance
numpy
matplotlib
```

### Custom Dependency
- `MyMatrixLibrary.MatrixModule` - Custom matrix multiplication library (you'll need to provide this)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
```

2. Install dependencies:
```bash
pip install PyQt5 yfinance numpy matplotlib
```

## Usage

1. Run the application:
```bash
python PortfolioOptimiser.py
```

2. **Input Stock Tickers**: Enter comma-separated ticker symbols (e.g., `AAPL, MSFT, GOOGL`)

3. **Set Desired Return** (Optional): For constrained optimization, enter your target annual return as a decimal (e.g., `0.15` for 15%)

4. **Choose Optimization Method**: Click one of the three optimization buttons

5. **Review Results**: View optimal weights, expected return, standard deviation, and Sharpe ratio

## How It Works

The optimizer uses historical price data from the past year to calculate:
- Daily log returns for each asset
- Covariance matrix of returns
- Annualized mean returns and standard deviations

Based on Modern Portfolio Theory, it then computes optimal portfolio weights using:
- Lagrange multipliers for constrained optimization
- Matrix operations for efficient calculations
- Risk-free rate (default: 2%) for Sharpe ratio and tangency portfolio

### Optimization Methods

**Minimum Variance**: Finds the portfolio with the lowest possible risk regardless of return.

**Efficient Portfolio with Constraint**: Minimizes variance while achieving a specified expected return.

**Tangency Portfolio**: Identifies the portfolio with the highest Sharpe ratio, representing the optimal risk-adjusted return when a risk-free asset is available.

## Configuration

Key parameters can be modified in the code:
- `rf`: Risk-free rate (default: 0.02 or 2%)
- Historical data range (default: past 365 days)

## Limitations

- Relies on historical data which may not predict future performance
- Assumes returns follow a normal distribution
- Does not account for transaction costs or taxes
- Requires at least 2 assets for meaningful optimization

## Disclaimer

This tool is for educational and research purposes only. It should not be considered financial advice. Always consult with a qualified financial advisor before making investment decisions.