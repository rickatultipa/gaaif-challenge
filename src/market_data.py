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

Live Data Provider:
- MarketDataProvider: Single source of truth for all market data
- SensitivityRangeGenerator: Dynamic ranges for sensitivity analyses
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

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


@dataclass
class MarketDataProvenance:
    """Tracks the source and freshness of each market data parameter."""
    source: str  # 'live', 'cached', 'fallback'
    timestamp: str  # ISO format
    ticker: str  # yfinance ticker or 'N/A'
    raw_value: float  # original fetched value


class MarketDataProvider:
    """
    Single source of truth for all market data parameters.

    Fetches live data via yfinance when available, with graceful
    fallback to reference values for offline/judging scenarios.

    Usage:
        provider = MarketDataProvider(use_live=True)
        market = provider.fetch_market_data()
        print(provider.get_provenance_report())
    """

    # Fallback defaults: snapshot for Feb 26, 2026 (offline judging reference)
    FALLBACK_DEFAULTS = {
        'gold_spot': 5200.0,
        'eurusd_spot': 1.18,
        'r_usd': 0.041,       # 13-week T-bill proxy
        'r_eur': 0.020,       # ECB deposit rate (configurable)
        'sigma_gold': 0.37,   # EWMA vol ~ GVZ level
        'sigma_eurusd': 0.10, # EWMA vol
        'rho': -0.30,         # 126-day rolling correlation
        'gold_yield': 0.003,  # Convenience yield from futures
    }

    def __init__(self, use_live: bool = True, eur_rate: float = 0.020,
                 vol_lambda: float = 0.94, corr_window: int = 126):
        """
        Args:
            use_live: Attempt to fetch live data from yfinance
            eur_rate: EUR risk-free rate (configurable; ECB not on yfinance)
            vol_lambda: EWMA decay factor (0.94 = RiskMetrics)
            corr_window: Rolling window for correlation estimation (trading days)
        """
        self.use_live = use_live and HAS_YFINANCE
        self.eur_rate = eur_rate
        self.vol_lambda = vol_lambda
        self.corr_window = corr_window
        self._provenance: Dict[str, MarketDataProvenance] = {}
        self._live_data_fetched = False

    def fetch_market_data(self):
        """
        Main entry point: returns a MarketData object (from pricing_model).

        Attempts live fetch, then falls back to defaults.

        Returns:
            MarketData instance with best-available parameters
        """
        from pricing_model import MarketData

        params = dict(self.FALLBACK_DEFAULTS)
        params['r_eur'] = self.eur_rate
        now = datetime.now().isoformat(timespec='seconds')

        # Initialize provenance with fallback for everything
        for key, val in params.items():
            self._provenance[key] = MarketDataProvenance(
                source='fallback', timestamp=now, ticker='N/A', raw_value=val
            )

        if self.use_live:
            try:
                self._fetch_live_data(params, now)
                self._live_data_fetched = True
            except Exception as e:
                logger.warning(f"Live data fetch failed: {e}. Using fallback defaults.")
                print(f"  [WARN] Live data fetch failed: {e}")
                print(f"  [INFO] Using fallback defaults (Feb 26, 2026 snapshot)")

        return MarketData(
            gold_spot=params['gold_spot'],
            eurusd_spot=params['eurusd_spot'],
            r_eur=params['r_eur'],
            r_usd=params['r_usd'],
            sigma_gold=params['sigma_gold'],
            sigma_eurusd=params['sigma_eurusd'],
            rho=params['rho'],
            gold_yield=params['gold_yield'],
        )

    def _fetch_live_data(self, params: dict, now: str):
        """Attempt to fetch all live data components."""
        self._fetch_spots(params, now)
        self._estimate_volatilities(params, now)
        self._estimate_correlation(params, now)
        self._fetch_interest_rates(params, now)
        self._estimate_convenience_yield(params, now)

    def _fetch_spots(self, params: dict, now: str):
        """Fetch gold and EUR/USD spot prices via yfinance."""
        # Gold futures as spot proxy
        try:
            gold = yf.download("GC=F", period="5d", progress=False)
            if gold is not None and len(gold) > 0:
                # Handle yfinance MultiIndex columns (recent versions)
                close = gold['Close']
                if hasattr(close, 'columns'):
                    close = close.iloc[:, 0]
                val = float(close.dropna().iloc[-1])
                params['gold_spot'] = val
                self._provenance['gold_spot'] = MarketDataProvenance(
                    source='live', timestamp=now, ticker='GC=F', raw_value=val
                )
        except Exception as e:
            logger.warning(f"Gold spot fetch failed: {e}")

        # EUR/USD
        try:
            fx = yf.download("EURUSD=X", period="5d", progress=False)
            if fx is not None and len(fx) > 0:
                close = fx['Close']
                if hasattr(close, 'columns'):
                    close = close.iloc[:, 0]
                val = float(close.dropna().iloc[-1])
                params['eurusd_spot'] = val
                self._provenance['eurusd_spot'] = MarketDataProvenance(
                    source='live', timestamp=now, ticker='EURUSD=X', raw_value=val
                )
        except Exception as e:
            logger.warning(f"EUR/USD spot fetch failed: {e}")

    def _estimate_volatilities(self, params: dict, now: str):
        """Estimate EWMA volatilities from historical data."""
        estimator = VolatilityEstimator()

        # Gold vol
        try:
            gold_hist = yf.download("GC=F", period="2y", progress=False)
            if gold_hist is not None and len(gold_hist) > 20:
                close = gold_hist['Close']
                if hasattr(close, 'columns'):
                    close = close.iloc[:, 0]
                close = close.dropna()
                ewma_vol = estimator.calculate_ewma_volatility(
                    close, lambda_=self.vol_lambda, annualize=True
                )
                val = float(ewma_vol.dropna().iloc[-1])
                if 0.05 < val < 1.5:  # sanity check
                    params['sigma_gold'] = val
                    self._provenance['sigma_gold'] = MarketDataProvenance(
                        source='live', timestamp=now, ticker='GC=F',
                        raw_value=val
                    )
        except Exception as e:
            logger.warning(f"Gold vol estimation failed: {e}")

        # EUR/USD vol
        try:
            fx_hist = yf.download("EURUSD=X", period="2y", progress=False)
            if fx_hist is not None and len(fx_hist) > 20:
                close = fx_hist['Close']
                if hasattr(close, 'columns'):
                    close = close.iloc[:, 0]
                close = close.dropna()
                ewma_vol = estimator.calculate_ewma_volatility(
                    close, lambda_=self.vol_lambda, annualize=True
                )
                val = float(ewma_vol.dropna().iloc[-1])
                if 0.02 < val < 0.5:
                    params['sigma_eurusd'] = val
                    self._provenance['sigma_eurusd'] = MarketDataProvenance(
                        source='live', timestamp=now, ticker='EURUSD=X',
                        raw_value=val
                    )
        except Exception as e:
            logger.warning(f"EUR/USD vol estimation failed: {e}")

    def _estimate_correlation(self, params: dict, now: str):
        """Estimate gold-EURUSD correlation from historical data."""
        estimator = VolatilityEstimator()
        try:
            gold_hist = yf.download("GC=F", period="2y", progress=False)
            fx_hist = yf.download("EURUSD=X", period="2y", progress=False)

            if (gold_hist is not None and fx_hist is not None and
                    len(gold_hist) > self.corr_window and len(fx_hist) > self.corr_window):
                gold_close = gold_hist['Close']
                fx_close = fx_hist['Close']
                if hasattr(gold_close, 'columns'):
                    gold_close = gold_close.iloc[:, 0]
                if hasattr(fx_close, 'columns'):
                    fx_close = fx_close.iloc[:, 0]

                corr_series = estimator.calculate_correlation(
                    gold_close.dropna(), fx_close.dropna(),
                    window=self.corr_window
                )
                val = float(corr_series.dropna().iloc[-1])
                if -1 <= val <= 1:
                    params['rho'] = val
                    self._provenance['rho'] = MarketDataProvenance(
                        source='live', timestamp=now,
                        ticker='GC=F vs EURUSD=X',
                        raw_value=val
                    )
        except Exception as e:
            logger.warning(f"Correlation estimation failed: {e}")

    def _fetch_interest_rates(self, params: dict, now: str):
        """Fetch USD rate from ^IRX (13-week T-bill); EUR is configurable."""
        # USD rate from T-bill
        try:
            irx = yf.download("^IRX", period="5d", progress=False)
            if irx is not None and len(irx) > 0:
                close = irx['Close']
                if hasattr(close, 'columns'):
                    close = close.iloc[:, 0]
                # ^IRX is quoted as percentage (e.g. 4.1 means 4.1%)
                val = float(close.dropna().iloc[-1]) / 100.0
                if 0 < val < 0.15:
                    params['r_usd'] = val
                    self._provenance['r_usd'] = MarketDataProvenance(
                        source='live', timestamp=now, ticker='^IRX',
                        raw_value=val
                    )
        except Exception as e:
            logger.warning(f"USD rate fetch failed: {e}")

        # EUR rate: configurable default (ECB not on yfinance)
        params['r_eur'] = self.eur_rate
        self._provenance['r_eur'] = MarketDataProvenance(
            source='configured', timestamp=now, ticker='N/A (ECB)',
            raw_value=self.eur_rate
        )

    def _estimate_convenience_yield(self, params: dict, now: str):
        """Estimate convenience yield from gold futures term structure: q = r - ln(F/S)/T."""
        try:
            # Near-month futures (GC=F) vs next contract
            spot_data = yf.download("GC=F", period="5d", progress=False)
            # Use a further-out contract for term structure
            far_data = yf.download("GCJ26.CMX", period="5d", progress=False)

            if (spot_data is not None and far_data is not None and
                    len(spot_data) > 0 and len(far_data) > 0):
                s_close = spot_data['Close']
                f_close = far_data['Close']
                if hasattr(s_close, 'columns'):
                    s_close = s_close.iloc[:, 0]
                if hasattr(f_close, 'columns'):
                    f_close = f_close.iloc[:, 0]

                S = float(s_close.dropna().iloc[-1])
                F = float(f_close.dropna().iloc[-1])
                T = 0.25  # approximate time to far contract (quarter)
                if S > 0 and F > 0 and T > 0:
                    q = params['r_usd'] - np.log(F / S) / T
                    q = max(0.0, min(q, 0.05))  # clamp to [0, 5%]
                    params['gold_yield'] = q
                    self._provenance['gold_yield'] = MarketDataProvenance(
                        source='live', timestamp=now,
                        ticker='GC=F vs GCJ26.CMX',
                        raw_value=q
                    )
        except Exception as e:
            logger.warning(f"Convenience yield estimation failed: {e}")

    def get_provenance_report(self) -> str:
        """Generate a human-readable provenance report."""
        lines = [
            "=" * 70,
            "DATA PROVENANCE REPORT",
            "=" * 70,
        ]

        param_labels = {
            'gold_spot': 'Gold Spot (USD/oz)',
            'eurusd_spot': 'EUR/USD Spot',
            'r_usd': 'USD Risk-Free Rate',
            'r_eur': 'EUR Risk-Free Rate',
            'sigma_gold': 'Gold Volatility (EWMA)',
            'sigma_eurusd': 'EUR/USD Volatility (EWMA)',
            'rho': 'Gold-EURUSD Correlation',
            'gold_yield': 'Gold Convenience Yield',
        }

        for key, label in param_labels.items():
            prov = self._provenance.get(key)
            if prov:
                if key in ('r_usd', 'r_eur', 'sigma_gold', 'sigma_eurusd', 'gold_yield'):
                    val_str = f"{prov.raw_value*100:.2f}%"
                elif key == 'rho':
                    val_str = f"{prov.raw_value:.3f}"
                elif key == 'gold_spot':
                    val_str = f"${prov.raw_value:,.2f}"
                else:
                    val_str = f"{prov.raw_value:.4f}"

                source_tag = f"[{prov.source.upper()}]"
                ticker_str = f"({prov.ticker})" if prov.ticker != 'N/A' else ""
                lines.append(f"  {label:<30s} {val_str:>12s}  {source_tag:<12s} {ticker_str}")

        lines.append(f"\n  Timestamp: {self._provenance.get('gold_spot', MarketDataProvenance('','','',0)).timestamp}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def get_provenance_dataframe(self) -> pd.DataFrame:
        """Return provenance data as a DataFrame for Excel export."""
        rows = []
        param_labels = {
            'gold_spot': 'Gold Spot (USD/oz)',
            'eurusd_spot': 'EUR/USD Spot',
            'r_usd': 'USD Risk-Free Rate',
            'r_eur': 'EUR Risk-Free Rate',
            'sigma_gold': 'Gold Volatility (EWMA)',
            'sigma_eurusd': 'EUR/USD Volatility (EWMA)',
            'rho': 'Gold-EURUSD Correlation',
            'gold_yield': 'Gold Convenience Yield',
        }
        for key, label in param_labels.items():
            prov = self._provenance.get(key)
            if prov:
                rows.append({
                    'Parameter': label,
                    'Value': prov.raw_value,
                    'Source': prov.source,
                    'Ticker': prov.ticker,
                    'Timestamp': prov.timestamp,
                })
        return pd.DataFrame(rows)


class SensitivityRangeGenerator:
    """
    Generates dynamic sensitivity ranges based on current market conditions.

    All ranges center on current market values and scale with observed
    volatility, ensuring analyses remain meaningful regardless of market level.
    """

    def __init__(self, market, contract):
        """
        Args:
            market: MarketData instance (from pricing_model)
            contract: ContractTerms instance (from pricing_model)
        """
        self.market = market
        self.contract = contract

    def gold_spot_range(self, n: int = 9) -> np.ndarray:
        """Gold spot: spot +/- 2 sigma over tenor horizon."""
        S = self.market.gold_spot
        sigma = self.market.sigma_gold
        T = self.contract.tenor
        spread = 2.0 * sigma * S * np.sqrt(T)
        lo = max(S - spread, S * 0.3)  # floor at 30% of spot
        hi = S + spread
        return np.linspace(lo, hi, n)

    def eurusd_spot_range(self, n: int = 10) -> np.ndarray:
        """EUR/USD: bounded by barriers [1.05, 1.25] with small margin."""
        lo = self.contract.barrier_lower + 0.01
        hi = self.contract.barrier_upper - 0.01
        return np.linspace(lo, hi, n)

    def gold_vol_range(self, n: int = 11) -> np.ndarray:
        """Gold vol: center on current vol, 0.4x to 2.0x."""
        center = self.market.sigma_gold
        lo = max(center * 0.4, 0.05)
        hi = center * 2.0
        return np.linspace(lo, hi, n)

    def eurusd_vol_range(self, n: int = 11) -> np.ndarray:
        """EUR/USD vol: center on current vol, 0.4x to 2.0x."""
        center = self.market.sigma_eurusd
        lo = max(center * 0.4, 0.02)
        hi = center * 2.0
        return np.linspace(lo, hi, n)

    def correlation_range(self, n: int = 11) -> np.ndarray:
        """Correlation: centered on current rho, +/- 0.4 (clamped to [-0.9, 0.9])."""
        center = self.market.rho
        lo = max(center - 0.4, -0.9)
        hi = min(center + 0.4, 0.9)
        return np.linspace(lo, hi, n)

    def payoff_diagram_range(self) -> Tuple[float, float]:
        """Gold price range for payoff diagrams: centered on strike, spanning spot and beyond."""
        K = self.contract.strike
        S = self.market.gold_spot
        sigma = self.market.sigma_gold
        T = self.contract.tenor
        spread = 2.0 * sigma * S * np.sqrt(T)
        lo = min(K, S) - spread * 0.5
        hi = max(K, S) + spread * 0.5
        lo = max(lo, K * 0.5)  # floor
        return (lo, hi)

    def scenario_strikes(self) -> List[float]:
        """Dynamic strike prices for scenario analysis, centered around contract strike."""
        K = self.contract.strike
        S = self.market.gold_spot
        # Include current strike, current spot level, and symmetric spread
        spread = abs(S - K) * 0.5
        strikes = sorted(set([
            round(K - spread * 2, -2),
            round(K - spread, -2),
            K,
            round(K + spread, -2),
            round(S, -2),
            round(S + spread, -2),
        ]))
        return [s for s in strikes if s > 0]

    def scenario_vol_regimes(self) -> List[Tuple[float, float, str]]:
        """Dynamic volatility regimes based on current market vol."""
        gv = self.market.sigma_gold
        fv = self.market.sigma_eurusd
        return [
            (gv * 0.5, fv * 0.6, 'Low Vol'),
            (gv * 0.7, fv * 0.8, 'Moderate Vol'),
            (gv, fv, 'Current Vol'),
            (gv * 1.3, fv * 1.3, 'High Vol'),
            (gv * 1.8, fv * 1.5, 'Crisis Vol'),
        ]

    def stress_test_scenarios(self) -> List[Tuple[str, dict]]:
        """Dynamic stress test scenarios using relative shocks."""
        m = self.market
        c = self.contract
        # Near-barrier EUR/USD scenarios
        near_lower = c.barrier_lower + 0.02
        near_upper = c.barrier_upper - 0.02
        midpoint = (c.barrier_lower + c.barrier_upper) / 2.0

        return [
            ('Base Case', {}),
            ('Gold Vol +50%', {'sigma_gold': m.sigma_gold * 1.5}),
            ('Gold Vol +100%', {'sigma_gold': m.sigma_gold * 2.0}),
            ('FX Vol +50%', {'sigma_eurusd': m.sigma_eurusd * 1.5}),
            ('FX Vol +100%', {'sigma_eurusd': m.sigma_eurusd * 2.0}),
            ('Correlation = 0', {'rho': 0.0}),
            ('Correlation = -0.5', {'rho': -0.5}),
            ('Correlation = +0.3', {'rho': 0.3}),
            ('EUR Rate +100bp', {'r_eur': m.r_eur + 0.01}),
            ('USD Rate +100bp', {'r_usd': m.r_usd + 0.01}),
            ('Gold Spot +10%', {'gold_spot': m.gold_spot * 1.1}),
            ('Gold Spot -10%', {'gold_spot': m.gold_spot * 0.9}),
            ('EUR/USD Near Lower', {'eurusd_spot': near_lower}),
            ('EUR/USD Near Upper', {'eurusd_spot': near_upper}),
            ('EUR/USD Midpoint', {'eurusd_spot': midpoint}),
        ]


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

    # Test MarketDataProvider
    print("\n" + "=" * 60)
    print("MarketDataProvider Test")
    print("=" * 60)

    provider = MarketDataProvider(use_live=True)
    market = provider.fetch_market_data()
    print(provider.get_provenance_report())
    print(f"\nMarket data loaded:")
    print(f"  Gold Spot: ${market.gold_spot:,.2f}")
    print(f"  EUR/USD:   {market.eurusd_spot:.4f}")
    print(f"  Gold Vol:  {market.sigma_gold*100:.1f}%")
    print(f"  Rho:       {market.rho:.3f}")
