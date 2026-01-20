"""
Market Data Module for GAAIF Challenge
======================================

This module handles market data sourcing and processing for:
- LBMA Gold Spot Prices (USD/oz)
- EUR/USD Exchange Rates (ECB Reference)
- Implied Volatilities
- Correlation Estimation

Data Sources:
- Gold: LBMA, Yahoo Finance, FRED
- EUR/USD: ECB, Yahoo Finance, FRED
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class MarketDataFetcher:
    """Fetches and processes market data for gold and EUR/USD."""

    def __init__(self):
        self.gold_data: Optional[pd.DataFrame] = None
        self.eurusd_data: Optional[pd.DataFrame] = None

    def fetch_yahoo_data(self, start_date: str = "2020-01-01",
                         end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch gold and EUR/USD data from Yahoo Finance.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (defaults to today)

        Returns:
            Tuple of (gold_df, eurusd_df)
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch Gold (GC=F is gold futures, GLD is gold ETF)
        print("Fetching gold data from Yahoo Finance...")
        gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
        gold_df = gold[['Close']].rename(columns={'Close': 'gold_price'})

        # Fetch EUR/USD
        print("Fetching EUR/USD data from Yahoo Finance...")
        eurusd = yf.download("EURUSD=X", start=start_date, end=end_date, progress=False)
        eurusd_df = eurusd[['Close']].rename(columns={'Close': 'eurusd_rate'})

        self.gold_data = gold_df
        self.eurusd_data = eurusd_df

        return gold_df, eurusd_df

    def generate_synthetic_data(self, n_days: int = 1260,
                               gold_spot: float = 2750.0,
                               eurusd_spot: float = 1.08,
                               sigma_gold: float = 0.18,
                               sigma_eurusd: float = 0.08,
                               rho: float = -0.25,
                               seed: int = 123) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic historical data for testing.

        Args:
            n_days: Number of trading days
            gold_spot: Current gold price
            eurusd_spot: Current EUR/USD rate
            sigma_gold: Gold volatility
            sigma_eurusd: EUR/USD volatility
            rho: Correlation
            seed: Random seed

        Returns:
            Tuple of (gold_df, eurusd_df)
        """
        np.random.seed(seed)

        dt = 1/252  # Daily time step
        sqrt_dt = np.sqrt(dt)

        # Generate correlated random walks (backward from current spot)
        Z1 = np.random.randn(n_days)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_days)

        # Simulate backward to get historical path
        gold_returns = sigma_gold * sqrt_dt * Z1
        eurusd_returns = sigma_eurusd * sqrt_dt * Z2

        # Build price series backward
        gold_log_prices = np.zeros(n_days + 1)
        eurusd_log_prices = np.zeros(n_days + 1)

        gold_log_prices[-1] = np.log(gold_spot)
        eurusd_log_prices[-1] = np.log(eurusd_spot)

        for i in range(n_days - 1, -1, -1):
            gold_log_prices[i] = gold_log_prices[i + 1] - gold_returns[i]
            eurusd_log_prices[i] = eurusd_log_prices[i + 1] - eurusd_returns[i]

        gold_prices = np.exp(gold_log_prices)
        eurusd_rates = np.exp(eurusd_log_prices)

        # Create date index
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=n_days + 1, freq='B')

        gold_df = pd.DataFrame({'gold_price': gold_prices}, index=dates)
        eurusd_df = pd.DataFrame({'eurusd_rate': eurusd_rates}, index=dates)

        self.gold_data = gold_df
        self.eurusd_data = eurusd_df

        return gold_df, eurusd_df


class VolatilityEstimator:
    """Estimates volatilities and correlations from historical data."""

    @staticmethod
    def calculate_historical_volatility(prices: pd.Series,
                                        window: int = 252,
                                        annualize: bool = True) -> pd.Series:
        """
        Calculate rolling historical volatility.

        Args:
            prices: Price series
            window: Rolling window size
            annualize: Whether to annualize (multiply by sqrt(252))

        Returns:
            Series of rolling volatilities
        """
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.rolling(window=window).std()

        if annualize:
            vol = vol * np.sqrt(252)

        return vol

    @staticmethod
    def calculate_ewma_volatility(prices: pd.Series,
                                  lambda_: float = 0.94,
                                  annualize: bool = True) -> pd.Series:
        """
        Calculate EWMA (RiskMetrics-style) volatility.

        Args:
            prices: Price series
            lambda_: Decay factor (0.94 for RiskMetrics)
            annualize: Whether to annualize

        Returns:
            Series of EWMA volatilities
        """
        log_returns = np.log(prices / prices.shift(1))
        variance = log_returns.ewm(alpha=1-lambda_, adjust=False).var()
        vol = np.sqrt(variance)

        if annualize:
            vol = vol * np.sqrt(252)

        return vol

    @staticmethod
    def calculate_correlation(series1: pd.Series, series2: pd.Series,
                             window: int = 252) -> pd.Series:
        """
        Calculate rolling correlation between two return series.

        Args:
            series1: First price series
            series2: Second price series
            window: Rolling window size

        Returns:
            Series of rolling correlations
        """
        ret1 = np.log(series1 / series1.shift(1))
        ret2 = np.log(series2 / series2.shift(1))

        correlation = ret1.rolling(window=window).corr(ret2)

        return correlation

    @staticmethod
    def estimate_parameters(gold_prices: pd.Series,
                           eurusd_rates: pd.Series,
                           lookback: int = 252) -> dict:
        """
        Estimate all model parameters from historical data.

        Args:
            gold_prices: Gold price series
            eurusd_rates: EUR/USD rate series
            lookback: Number of days for estimation

        Returns:
            Dictionary with estimated parameters
        """
        # Use most recent lookback period
        gold = gold_prices.iloc[-lookback:]
        eurusd = eurusd_rates.iloc[-lookback:]

        # Log returns
        gold_ret = np.log(gold / gold.shift(1)).dropna()
        eurusd_ret = np.log(eurusd / eurusd.shift(1)).dropna()

        # Volatilities (annualized)
        sigma_gold = gold_ret.std() * np.sqrt(252)
        sigma_eurusd = eurusd_ret.std() * np.sqrt(252)

        # Correlation
        correlation = gold_ret.corr(eurusd_ret)

        # Current spots
        gold_spot = gold_prices.iloc[-1]
        eurusd_spot = eurusd_rates.iloc[-1]

        return {
            'gold_spot': gold_spot,
            'eurusd_spot': eurusd_spot,
            'sigma_gold': sigma_gold,
            'sigma_eurusd': sigma_eurusd,
            'correlation': correlation,
            'lookback_days': lookback
        }


def create_sample_market_data() -> dict:
    """
    Create sample market data based on realistic 2025-2026 market conditions.

    Returns:
        Dictionary with market parameters
    """
    # Based on market conditions as of late 2025/early 2026
    return {
        # Spot prices
        'gold_spot': 2750.0,  # Gold near all-time highs
        'eurusd_spot': 1.08,  # EUR/USD in middle of typical range

        # Interest rates (approximate late 2025 levels)
        'r_eur': 0.025,  # ECB deposit rate ~2.5%
        'r_usd': 0.045,  # Fed funds ~4.5%

        # Volatilities (from recent market data)
        'sigma_gold': 0.18,  # Gold 1Y implied vol ~18%
        'sigma_eurusd': 0.08,  # EUR/USD 1Y implied vol ~8%

        # Correlation (historical estimate)
        'rho': -0.25,  # Typically negative (gold up when USD weakens)

        # Other
        'gold_yield': 0.005,  # Convenience yield / lease rate ~0.5%
    }


def get_market_data_summary(gold_df: pd.DataFrame,
                           eurusd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for market data.

    Args:
        gold_df: Gold price DataFrame
        eurusd_df: EUR/USD DataFrame

    Returns:
        Summary statistics DataFrame
    """
    # Align data
    data = pd.concat([gold_df['gold_price'], eurusd_df['eurusd_rate']], axis=1)
    data.columns = ['Gold (USD/oz)', 'EUR/USD']
    data = data.dropna()

    # Calculate returns
    returns = np.log(data / data.shift(1)).dropna()

    # Summary statistics
    summary = pd.DataFrame({
        'Current': data.iloc[-1],
        'Mean': data.mean(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Daily Vol': returns.std(),
        'Annual Vol': returns.std() * np.sqrt(252)
    })

    return summary.T


if __name__ == "__main__":
    print("=" * 60)
    print("Market Data Module Test")
    print("=" * 60)

    # Generate synthetic data
    fetcher = MarketDataFetcher()
    gold_df, eurusd_df = fetcher.generate_synthetic_data()

    print(f"\nGenerated {len(gold_df)} days of synthetic data")

    # Estimate parameters
    estimator = VolatilityEstimator()
    params = estimator.estimate_parameters(
        gold_df['gold_price'],
        eurusd_df['eurusd_rate']
    )

    print("\nEstimated Parameters:")
    print(f"  Gold Spot:        ${params['gold_spot']:,.2f}")
    print(f"  EUR/USD Spot:     {params['eurusd_spot']:.4f}")
    print(f"  Gold Volatility:  {params['sigma_gold']*100:.1f}%")
    print(f"  EUR/USD Vol:      {params['sigma_eurusd']*100:.1f}%")
    print(f"  Correlation:      {params['correlation']:.3f}")

    # Summary
    print("\nData Summary:")
    summary = get_market_data_summary(gold_df, eurusd_df)
    print(summary.round(4))
