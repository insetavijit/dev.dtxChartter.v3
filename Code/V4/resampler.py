"""Clean OHLCV Data Resampler
Efficient resampling for financial time series data with mplfinance compatibility.
"""

import pandas as pd
from typing import Optional, Dict, Any


class OHLCVResampler:
    """Enterprise-grade OHLCV data resampler."""

    # OHLC aggregation rules (case-insensitive)
    OHLC_RULES = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vol': 'sum'
    }

    def __init__(self, default_agg: str = 'last'):
        """Initialize resampler.

        Args:
            default_agg: Default aggregation for non-OHLCV columns ('last', 'first', 'mean', etc.)
        """
        self.default_agg = default_agg

    def resample(
        self,
        df: pd.DataFrame,
        period: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        custom_agg: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Resample all columns in DataFrame while preserving OHLCV logic.

        Args:
            df: DataFrame with datetime index or column
            period: Resampling period ('1H', '4H', '1D', '1W', '1M', etc.)
            start_date: Optional start date filter
            end_date: Optional end date filter
            custom_agg: Custom aggregation rules for specific columns

        Returns:
            Resampled DataFrame with all original columns

        Examples:
            >>> resampler = OHLCVResampler()
            >>> result = resampler.resample(df, '1M', '2024-01-01', '2025-01-01')
        """
        if df.empty:
            return df.copy()

        # Prepare dataframe with datetime index
        df = self._prepare_dataframe(df)

        # Apply date filtering
        df = self._filter_dates(df, start_date, end_date)

        if df.empty:
            return df

        # Build aggregation rules for all columns
        agg_rules = self._build_aggregation_rules(df.columns, custom_agg)

        # Resample
        resampled = df.resample(period).agg(agg_rules)

        # Remove empty periods
        resampled = resampled.dropna(how='all')

        return resampled

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe with datetime index."""
        df = df.copy()

        # Auto-detect and set datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            datetime_col = self._find_datetime_column(df)

            if datetime_col:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                df = df.set_index(datetime_col)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise ValueError(
                        "No datetime column or index found. "
                        "Ensure data has datetime column or DatetimeIndex."
                    ) from e

        return df.sort_index()

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column by common names."""
        datetime_names = ['datetime', 'date', 'time', 'timestamp']

        for col in df.columns:
            if col.lower() in datetime_names:
                return col

        return None

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter dataframe by date range."""
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]

        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def _build_aggregation_rules(
        self,
        columns: pd.Index,
        custom_agg: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Build aggregation rules for all columns."""
        agg_rules = {}

        for col in columns:
            col_lower = col.lower()

            # Check if custom aggregation specified
            if custom_agg and col in custom_agg:
                agg_rules[col] = custom_agg[col]
            # Check if it's an OHLCV column
            elif col_lower in self.OHLC_RULES:
                agg_rules[col] = self.OHLC_RULES[col_lower]
            # Use default aggregation
            else:
                agg_rules[col] = self.default_agg

        return agg_rules


def resample_ohlcv(
    df: pd.DataFrame,
    period: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    default_agg: str = 'last',
    custom_agg: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Quick function to resample OHLCV data with all columns preserved.

    Args:
        df: DataFrame with OHLCV data
        period: Resampling period ('1H', '4H', '1D', '1W', '1M')
        start_date: Optional start date
        end_date: Optional end date
        default_agg: Default aggregation for non-OHLCV columns
        custom_agg: Custom aggregation rules

    Returns:
        Resampled DataFrame with all columns
    """
    resampler = OHLCVResampler(default_agg)
    return resampler.resample(df, period, start_date, end_date, custom_agg)
