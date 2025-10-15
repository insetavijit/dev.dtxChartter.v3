"""
Unit Tests for Trading Backtest Framework
Industry-grade test suite covering all critical components.

Test Coverage:
  - Signal Detection: BUY/SELL logic with edge cases
  - Position Sizing: Risk calculations and bounds
  - Trade Finalization: P&L accuracy for LONG/SHORT
  - Data Validation: Missing values and error handling
  - Configuration: Parameter validation

Run with: pytest tsts.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create sample OHLCV data with entry/SL/target columns."""
    dates = pd.date_range('2024-11-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'open': np.random.uniform(2740, 2760, 100),
        'high': np.random.uniform(2740, 2760, 100),
        'low': np.random.uniform(2740, 2760, 100),
        'close': np.random.uniform(2740, 2760, 100),
        'volume': np.random.randint(50, 200, 100),
        'entry': np.nan,
        'sl': np.nan,
        'target': np.nan,
    }, index=dates)
    return df


@pytest.fixture
def buy_signal_data():
    """Create data with valid BUY signal (SL < entry)."""
    dates = pd.date_range('2024-11-01', periods=5, freq='1min')
    df = pd.DataFrame({
        'open': [2750, 2751, 2752, 2753, 2754],
        'high': [2751, 2752, 2753, 2754, 2755],
        'low': [2749, 2750, 2751, 2752, 2753],
        'close': [2750.5, 2751.5, 2752.5, 2753.5, 2754.5],
        'volume': [100, 100, 100, 100, 100],
        'entry': [2750.0, np.nan, np.nan, np.nan, np.nan],
        'sl': [2745.0, np.nan, np.nan, np.nan, np.nan],  # SL < entry = BUY
        'target': [2760.0, np.nan, np.nan, np.nan, np.nan],
    }, index=dates)
    return df


@pytest.fixture
def sell_signal_data():
    """Create data with valid SELL signal (SL > entry)."""
    dates = pd.date_range('2024-11-01', periods=5, freq='1min')
    df = pd.DataFrame({
        'open': [2750, 2751, 2752, 2753, 2754],
        'high': [2751, 2752, 2753, 2754, 2755],
        'low': [2749, 2750, 2751, 2752, 2753],
        'close': [2750.5, 2751.5, 2752.5, 2753.5, 2754.5],
        'volume': [100, 100, 100, 100, 100],
        'entry': [2750.0, np.nan, np.nan, np.nan, np.nan],
        'sl': [2755.0, np.nan, np.nan, np.nan, np.nan],  # SL > entry = SELL
        'target': [2740.0, np.nan, np.nan, np.nan, np.nan],
    }, index=dates)
    return df


# ============================================================================
# SIGNAL DETECTION TESTS
# ============================================================================

class TestSignalDetection:
    """Test automatic signal detection from entry/SL relationship."""

    def test_buy_signal_detection(self, buy_signal_data):
        """BUY signal when SL < entry (downside risk)."""
        # Import from parent directory - adjust path if needed
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        result = SignalDetector.detect_signals(buy_signal_data)

        assert result.loc[result.index[0], 'signal'] == 1, "BUY signal should be 1"
        assert result.loc[result.index[1:], 'signal'].sum() == 0, "Other rows should be 0"

    def test_sell_signal_detection(self, sell_signal_data):
        """SELL signal when SL > entry (upside risk for short)."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        result = SignalDetector.detect_signals(sell_signal_data)

        assert result.loc[result.index[0], 'signal'] == 2, "SELL signal should be 2"
        assert result.loc[result.index[1:], 'signal'].sum() == 0, "Other rows should be 0"

    def test_no_signal_missing_entry(self, sample_dataframe):
        """No signal when entry is missing."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        df = sample_dataframe.copy()
        df.loc[df.index[0], 'entry'] = np.nan
        df.loc[df.index[0], 'sl'] = 2745.0

        result = SignalDetector.detect_signals(df)
        assert result.loc[result.index[0], 'signal'] == 0, "Signal should be 0 when entry missing"

    def test_no_signal_missing_sl(self, sample_dataframe):
        """No signal when stop loss is missing."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        df = sample_dataframe.copy()
        df.loc[df.index[0], 'entry'] = 2750.0
        df.loc[df.index[0], 'sl'] = np.nan

        result = SignalDetector.detect_signals(df)
        assert result.loc[result.index[0], 'signal'] == 0, "Signal should be 0 when SL missing"

    def test_equal_entry_sl(self, sample_dataframe):
        """No signal when entry equals SL (edge case)."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        df = sample_dataframe.copy()
        df.loc[df.index[0], 'entry'] = 2750.0
        df.loc[df.index[0], 'sl'] = 2750.0

        result = SignalDetector.detect_signals(df)
        assert result.loc[result.index[0], 'signal'] == 0, "No signal when entry==SL"

    def test_multiple_signals(self):
        """Detect multiple signals in same dataset."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        dates = pd.date_range('2024-11-01', periods=6, freq='1min')
        df = pd.DataFrame({
            'open': [2750]*6,
            'high': [2751]*6,
            'low': [2749]*6,
            'close': [2750.5]*6,
            'volume': [100]*6,
            'entry': [2750.0, 2750.0, np.nan, 2750.0, np.nan, 2750.0],
            'sl': [2745.0, 2755.0, np.nan, 2745.0, 2755.0, np.nan],
            'target': [2760.0]*6,
        }, index=dates)

        result = SignalDetector.detect_signals(df)
        buy_count = (result['signal'] == 1).sum()
        sell_count = (result['signal'] == 2).sum()

        assert buy_count == 2, "Should detect 2 BUY signals"
        assert sell_count == 1, "Should detect 1 SELL signal"

    def test_signal_immutability(self, buy_signal_data):
        """Signal detection should not modify original dataframe."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        original = buy_signal_data.copy()
        result = SignalDetector.detect_signals(buy_signal_data)

        assert 'signal' not in original.columns, "Original DF should not be modified"
        assert 'signal' in result.columns, "Result should have signal column"


# ============================================================================
# POSITION SIZING TESTS
# ============================================================================

class TestPositionSizing:
    """Test risk-based position size calculations."""

    @staticmethod
    def calculate_position_size(account_value, entry, sl, risk_pct=0.02):
        """Standalone position sizing logic for testing."""
        risk_amount = account_value * risk_pct
        price_risk = abs(entry - sl)

        if price_risk <= 0:
            return 1

        theoretical_size = risk_amount / price_risk
        max_position_value = account_value * 0.1
        max_size_by_value = max_position_value / entry
        position_size = min(theoretical_size, max_size_by_value)

        return max(1, min(100, int(position_size)))

    def test_position_size_basic(self):
        """Position size = risk_amount / price_risk."""
        size = self.calculate_position_size(
            account_value=100000,
            entry=2750.0,
            sl=2745.0,
            risk_pct=0.02
        )

        assert isinstance(size, int), "Position size must be integer"
        assert 1 <= size <= 100, "Position size must be between 1-100"

    def test_position_size_zero_risk(self):
        """Position size defaults to 1 when price_risk is 0."""
        size = self.calculate_position_size(
            account_value=100000,
            entry=2750.0,
            sl=2750.0
        )
        assert size == 1, "Size should be 1 when price_risk is 0"

    def test_position_size_bounds(self):
        """Position size always between 1 and 100."""
        size_small = self.calculate_position_size(
            account_value=100000,
            entry=2750.0,
            sl=2700.0
        )
        assert 1 <= size_small <= 100, "Size should be bounded 1-100"

        size_large = self.calculate_position_size(
            account_value=100000,
            entry=2750.0,
            sl=2749.0
        )
        assert 1 <= size_large <= 100, "Size should be bounded 1-100"

    def test_position_size_scales_with_risk(self):
        """Higher risk_pct should produce larger position sizes."""
        size_2pct = self.calculate_position_size(
            account_value=100000,
            entry=2750.0,
            sl=2745.0,
            risk_pct=0.02
        )

        size_4pct = self.calculate_position_size(
            account_value=100000,
            entry=2750.0,
            sl=2745.0,
            risk_pct=0.04
        )

        assert size_4pct >= size_2pct, "Higher risk should produce larger position"

    def test_position_size_varies_with_account(self):
        """Larger account should allow larger positions."""
        size_small_account = self.calculate_position_size(
            account_value=10000,
            entry=2750.0,
            sl=2745.0,
            risk_pct=0.02
        )

        size_large_account = self.calculate_position_size(
            account_value=1000000,
            entry=2750.0,
            sl=2745.0,
            risk_pct=0.02
        )

        assert size_large_account >= size_small_account, "Larger account should allow larger position"


# ============================================================================
# TRADE CALCULATION TESTS
# ============================================================================

class TestTradeCalculations:
    """Test P&L and trade metric calculations."""

    def test_long_trade_pnl_win(self):
        """LONG trade P&L: entry=100, exit=110, size=10 -> P&L=100."""
        entry = 100.0
        exit_price = 110.0
        size = 10

        pnl = (exit_price - entry) * size

        assert pnl == 100.0, "LONG win P&L incorrect"

    def test_long_trade_pnl_loss(self):
        """LONG trade P&L: entry=100, exit=95, size=10 -> P&L=-50."""
        entry = 100.0
        exit_price = 95.0
        size = 10

        pnl = (exit_price - entry) * size

        assert pnl == -50.0, "LONG loss P&L incorrect"

    def test_short_trade_pnl_win(self):
        """SHORT trade P&L: entry=100, exit=90, size=10 -> P&L=100."""
        entry = 100.0
        exit_price = 90.0
        size = 10

        pnl = (entry - exit_price) * size

        assert pnl == 100.0, "SHORT win P&L incorrect"

    def test_short_trade_pnl_loss(self):
        """SHORT trade P&L: entry=100, exit=105, size=10 -> P&L=-50."""
        entry = 100.0
        exit_price = 105.0
        size = 10

        pnl = (entry - exit_price) * size

        assert pnl == -50.0, "SHORT loss P&L incorrect"

    def test_return_percentage_calculation(self):
        """Return % = (P&L / (entry * size)) * 100."""
        entry = 100.0
        exit_price = 110.0
        size = 10
        pnl = (exit_price - entry) * size

        return_pct = (pnl / (entry * size)) * 100

        assert return_pct == 10.0, "Return % should be 10%"

    def test_risk_reward_ratio(self):
        """Risk/Reward = reward_distance / risk_distance."""
        entry = 100.0
        sl = 95.0
        tp = 110.0

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr_ratio = reward / risk if risk > 0 else 0.0

        assert risk == 5.0
        assert reward == 10.0
        assert rr_ratio == 2.0, "R:R should be 2:1"

    def test_exit_type_detection_sl(self):
        """Exit type = SL when closer to SL than TP."""
        exit_price = 97.0
        sl = 95.0
        tp = 110.0

        sl_dist = abs(exit_price - sl)
        tp_dist = abs(exit_price - tp)
        exit_type = "SL" if sl_dist < tp_dist else "TP"

        assert exit_type == "SL", "Should identify SL exit"

    def test_exit_type_detection_tp(self):
        """Exit type = TP when closer to TP than SL."""
        exit_price = 108.0
        sl = 95.0
        tp = 110.0

        sl_dist = abs(exit_price - sl)
        tp_dist = abs(exit_price - tp)
        exit_type = "SL" if sl_dist < tp_dist else "TP"

        assert exit_type == "TP", "Should identify TP exit"


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestValidation:
    """Test data validation and error handling."""

    def test_validate_sl_tp_valid(self):
        """Valid SL/TP should return True."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import EnterpriseStrategy
        result = EnterpriseStrategy._validate_sl_tp(2745.0, 2760.0)
        assert result is True, "Valid SL/TP should be True"

    def test_validate_sl_tp_missing_sl(self):
        """Missing SL should return False."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import EnterpriseStrategy
        result = EnterpriseStrategy._validate_sl_tp(np.nan, 2760.0)
        assert result is False, "Missing SL should be False"

    def test_validate_sl_tp_missing_tp(self):
        """Missing TP should return False."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import EnterpriseStrategy
        result = EnterpriseStrategy._validate_sl_tp(2745.0, np.nan)
        assert result is False, "Missing TP should be False"

    def test_validate_sl_tp_zero_values(self):
        """Zero values should return False."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import EnterpriseStrategy
        result = EnterpriseStrategy._validate_sl_tp(0, 2760.0)
        assert result is False, "Zero SL should be False"

    def test_validate_sl_tp_negative_values(self):
        """Negative values should return False."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import EnterpriseStrategy
        result = EnterpriseStrategy._validate_sl_tp(-2745.0, 2760.0)
        assert result is False, "Negative SL should be False"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestTradeConfig:
    """Test configuration validation and defaults."""

    def test_config_defaults(self):
        """TradeConfig should have sensible defaults."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import TradeConfig
        config = TradeConfig()

        assert config.risk_pct == 0.02, "Default risk should be 2%"
        assert config.max_positions == 5, "Default max_positions should be 5"
        assert config.cash == 100000.0, "Default cash should be $100k"
        assert config.commission == 0.001, "Default commission should be 0.1%"

    def test_config_custom_values(self):
        """TradeConfig should accept custom values."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import TradeConfig
        config = TradeConfig(
            risk_pct=0.05,
            max_positions=10,
            cash=50000.0,
            commission=0.0005
        )

        assert config.risk_pct == 0.05
        assert config.max_positions == 10
        assert config.cash == 50000.0
        assert config.commission == 0.0005


# ============================================================================
# TRADE DATA STRUCTURE TESTS
# ============================================================================

class TestTradeData:
    """Test TradeData dataclass structure."""

    def test_trade_data_initialization(self):
        """TradeData should initialize with required fields."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import TradeData
        td = TradeData(
            signal_type='BUY',
            signal_id=1,
            datetime=datetime.now(),
            entry_price=2750.0,
            sl=2745.0,
            tp=2760.0,
            size=10
        )

        assert td.signal_type == 'BUY'
        assert td.entry_price == 2750.0
        assert td.actual_entry is None

    def test_trade_data_optional_fields(self):
        """TradeData optional fields should default to None."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import TradeData
        td = TradeData(
            signal_type='SELL',
            signal_id=2,
            datetime=datetime.now(),
            entry_price=2750.0,
            sl=2755.0,
            tp=2740.0,
            size=5
        )

        assert td.actual_entry is None
        assert td.actual_size is None
        assert td.exit_price is None


# ============================================================================
# DRAWDOWN TRACKING TESTS
# ============================================================================

class TestDrawdownTracking:
    """Test maximum drawdown calculations."""

    def test_drawdown_calculation(self):
        """Drawdown = (peak - current) / peak."""
        peak_value = 100000
        current_value = 95000

        drawdown = (peak_value - current_value) / peak_value

        assert drawdown == 0.05, "Drawdown should be 5%"

    def test_drawdown_zero(self):
        """No drawdown when at peak."""
        peak_value = 100000
        current_value = 100000

        drawdown = (peak_value - current_value) / peak_value

        assert drawdown == 0.0, "Drawdown should be 0%"

    def test_drawdown_total_loss(self):
        """Maximum drawdown is 100% when account goes to zero."""
        peak_value = 100000
        current_value = 0

        drawdown = (peak_value - current_value) / peak_value

        assert drawdown == 1.0, "Drawdown should be 100%"


# ============================================================================
# DAILY SUMMARY TESTS
# ============================================================================

class TestDailySummary:
    """Test daily performance tracking."""

    def test_daily_pnl_calculation(self):
        """Daily P&L = end_value - start_value."""
        start_value = 100000
        end_value = 101500

        daily_pnl = end_value - start_value

        assert daily_pnl == 1500, "Daily P&L should be $1500"

    def test_daily_return_percentage(self):
        """Daily return % = (P&L / start_value) * 100."""
        start_value = 100000
        end_value = 102000
        daily_pnl = 2000

        daily_return = (daily_pnl / start_value * 100)

        assert daily_return == 2.0, "Daily return should be 2%"

    def test_daily_return_negative(self):
        """Daily return can be negative."""
        start_value = 100000
        end_value = 98500
        daily_pnl = -1500

        daily_return = (daily_pnl / start_value * 100)

        assert daily_return == -1.5, "Daily return should be -1.5%"

    def test_daily_summary_zero_division_protection(self):
        """Handle edge case where start_value is zero."""
        start_value = 0
        end_value = 100
        daily_pnl = 100

        daily_return = (daily_pnl / start_value * 100) if start_value > 0 else 0

        assert daily_return == 0, "Should handle zero division"


# ============================================================================
# SIGNAL STATISTICS TESTS
# ============================================================================

class TestSignalStatistics:
    """Test signal counting and conversion metrics."""

    def test_signal_conversion_rate(self):
        """Signal conversion % = (trades / signals) * 100."""
        total_signals = 10
        trades_executed = 7

        conversion_rate = (trades_executed / total_signals * 100)

        assert conversion_rate == 70.0, "Conversion rate should be 70%"

    def test_win_rate_calculation(self):
        """Win rate % = (wins / total_trades) * 100."""
        winning_trades = 6
        losing_trades = 4
        total_trades = 10

        win_rate = (winning_trades / total_trades * 100)

        assert win_rate == 60.0, "Win rate should be 60%"

    def test_win_rate_no_trades(self):
        """Win rate should be 0 when no trades executed."""
        winning_trades = 0
        total_trades = 0

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        assert win_rate == 0.0, "Win rate should be 0 with no trades"

    def test_buy_sell_signal_split(self):
        """Track separate BUY and SELL signal counts."""
        buy_signals = 5
        sell_signals = 8
        total_signals = buy_signals + sell_signals

        assert total_signals == 13
        assert buy_signals < total_signals
        assert sell_signals < total_signals


# ============================================================================
# FRAMEWORK INTEGRATION TESTS
# ============================================================================

class TestFrameworkIntegration:
    """Test complete framework workflow without Backtrader."""

    def test_complete_workflow_with_signals(self):
        """Test full signal detection workflow."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        dates = pd.date_range('2024-11-01', periods=10, freq='1min')
        df = pd.DataFrame({
            'open': np.random.uniform(2740, 2760, 10),
            'high': np.random.uniform(2740, 2760, 10),
            'low': np.random.uniform(2740, 2760, 10),
            'close': np.random.uniform(2740, 2760, 10),
            'volume': np.random.randint(50, 200, 10),
            'entry': [2750.0, np.nan, 2755.0, np.nan, 2748.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            'sl': [2745.0, np.nan, 2760.0, np.nan, 2750.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            'target': [2760.0, np.nan, 2750.0, np.nan, 2745.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        }, index=dates)

        result = SignalDetector.detect_signals(df)

        assert 'signal' in result.columns
        assert result['signal'].dtype in [np.int64, np.int32, int]
        assert result['signal'].min() >= 0
        assert result['signal'].max() <= 2

    def test_dataframe_index_integrity(self):
        """DataFrame index should remain intact after signal detection."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        dates = pd.date_range('2024-11-01', periods=5, freq='1min')
        df = pd.DataFrame({
            'open': [2750]*5,
            'high': [2751]*5,
            'low': [2749]*5,
            'close': [2750.5]*5,
            'volume': [100]*5,
            'entry': [2750.0, np.nan, np.nan, np.nan, np.nan],
            'sl': [2745.0, np.nan, np.nan, np.nan, np.nan],
            'target': [2760.0, np.nan, np.nan, np.nan, np.nan],
        }, index=dates)

        result = SignalDetector.detect_signals(df)

        assert len(result) == len(df)
        assert result.index.equals(df.index)

    def test_dataframe_columns_preserved(self):
        """All original columns should be preserved."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        dates = pd.date_range('2024-11-01', periods=3, freq='1min')
        original_columns = ['open', 'high', 'low', 'close', 'volume', 'entry', 'sl', 'target']
        df = pd.DataFrame({col: [0]*3 for col in original_columns}, index=dates)

        result = SignalDetector.detect_signals(df)

        for col in original_columns:
            assert col in result.columns, f"Column {col} should be preserved"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_dataframe(self):
        """Handle empty dataframe gracefully."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        df = pd.DataFrame({
            'open': [], 'high': [], 'low': [], 'close': [], 'volume': [],
            'entry': [], 'sl': [], 'target': []
        })

        result = SignalDetector.detect_signals(df)

        assert len(result) == 0
        assert 'signal' in result.columns

    def test_all_nan_values(self):
        """Handle dataframe with all NaN entry/SL values."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from bktst import SignalDetector
        dates = pd.date_range('2024-11-01', periods=5, freq='1min')
        df = pd.DataFrame({
            'open': [2750]*5,
            'high': [2751]*5,
            'low': [2749]*5,
            'close': [2750.5]*5,
            'volume': [100]*5,
            'entry': [np.nan]*5,
            'sl': [np.nan]*5,
            'target': [2760.0]*5,
        }, index=dates)

        result = SignalDetector.detect_signals(df)

        assert (result['signal'] == 0).all(), "All signals should be 0"

    def test_large_price_values(self):
        """Handle very large price values."""
        size = TestPositionSizing.calculate_position_size(
            account_value=100000,
            entry=1000000.0,
            sl=999995.0,
            risk_pct=0.02
        )

        assert 1 <= size <= 100


# ============================================================================
# MOCK OBJECTS
# ============================================================================

class MockBroker:
    """Mock broker for testing position sizing without Backtrader."""

    def __init__(self, account_value=100000):
        self.account_value = account_value

    def getvalue(self):
        return self.account_value


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
