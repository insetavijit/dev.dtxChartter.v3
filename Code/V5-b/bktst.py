"""
Trading Backtest Framework - Enterprise Grade
A modular, well-documented backtesting system for signal-based trading strategies.

Features:
  - BUY/SELL signal processing with bracket orders
  - Risk-based position sizing
  - Trade journal with P&L tracking
  - Daily performance summaries
  - Professional logging and reporting

Author: Trading Research Team
Version: 1.0.0
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """
    Configure professional logger with timestamp and level formatting.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

logger = setup_logger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeConfig:
    """Configuration parameters for trading strategy."""
    risk_pct: float = 0.02  # Risk 2% per trade
    max_positions: int = 5  # Max concurrent positions
    max_daily_loss: float = 0.05  # Stop trading if -5% daily
    cash: float = 100000.0
    commission: float = 0.001

@dataclass
class TradeData:
    """Container for pending trade information."""
    signal_type: str  # 'BUY' or 'SELL'
    signal_id: int
    datetime: datetime
    entry_price: float
    sl: float
    tp: float
    size: int
    orders: List = None
    actual_entry: Optional[float] = None
    actual_size: Optional[float] = None
    exit_price: Optional[float] = None

# ============================================================================
# SIGNAL DETECTION
# ============================================================================

class SignalDetector:
    """
    Automatic signal detection based on entry and stop loss levels.

    Logic:
      - BUY signal (1): Stop loss is BELOW entry price (risk is downward)
      - SELL signal (2): Stop loss is ABOVE entry price (risk is upward/short)
      - NO signal (0): No entry or SL provided
    """

    @staticmethod
    def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect BUY/SELL signals from entry and SL columns.

        Args:
            df: DataFrame with columns [entry, sl, target]

        Returns:
            DataFrame with new 'signal' column (0, 1, or 2)

        Example:
            >>> df = pd.read_csv('data.csv')
            >>> df = SignalDetector.detect_signals(df)
            >>> df[['entry', 'sl', 'target', 'signal']].head()
        """
        df = df.copy()
        df['signal'] = 0  # Default: no signal

        # Both entry and SL must be present
        valid_mask = df['entry'].notna() & df['sl'].notna()

        # BUY signal: SL < entry (stop loss below entry, risk is downside)
        buy_mask = valid_mask & (df['sl'] < df['entry'])
        df.loc[buy_mask, 'signal'] = 1

        # SELL signal: SL > entry (stop loss above entry, risk is upside/short)
        sell_mask = valid_mask & (df['sl'] > df['entry'])
        df.loc[sell_mask, 'signal'] = 2

        return df

# ============================================================================
# DATA FEED
# ============================================================================

class TradeDataFeed(bt.feeds.PandasData):
    """
    Custom Backtrader data feed for trading signals.

    Expected DataFrame columns:
      - open, high, low, close, volume (OHLCV)
      - signal (0=None, 1=BUY, 2=SELL)
      - sl (stop loss level)
      - target (take profit level)
    """
    lines = ('signal', 'sl', 'target')
    params = (
        ('signal', 'signal'),
        ('sl', 'sl'),
        ('target', 'target'),
    )

# ============================================================================
# STRATEGY LOGIC
# ============================================================================

class EnterpriseStrategy(bt.Strategy):
    """
    Professional trading strategy with signal processing and risk management.

    Processes BUY (1) and SELL (2) signals with bracket orders.
    Implements position sizing, daily loss limits, and complete trade tracking.
    """

    params = (
        ('risk_pct', 0.02),
        ('max_positions', 5),
        ('max_daily_loss', 0.05),
        ('cash', 100000.0),
        ('commission', 0.001),
    )

    def __init__(self):
        """Initialize strategy with data references and tracking structures."""
        logger.info("Initializing Enterprise Strategy")

        # Data line references
        self.signal = self.data.signal
        self.sl = self.data.sl
        self.target = self.data.target

        # Results tables
        self.orders_table = []
        self.trades_table = []
        self.daily_summary_table = []

        # Counters
        self.signal_count = {'buy': 0, 'sell': 0}
        self.trade_count = 0
        self.order_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Risk tracking
        self.max_drawdown = 0.0
        self.peak_value = self.broker.getvalue()
        self.daily_start_value = self.broker.getvalue()

        # Position tracking
        self.current_positions = 0
        self.last_date = None
        self.pending_trades: Dict[str, TradeData] = {}

        logger.info(f"Strategy initialized | Capital: ${self.broker.getvalue():,.2f}")

    def next(self):
        """
        Called on each bar. Processes signals and updates tracking.
        """
        current_date = self.data.datetime.date()
        current_value = self.broker.getvalue()

        # Daily boundary check
        if self.last_date != current_date:
            if self.last_date is not None:
                self._record_daily_summary()
            self._reset_daily_tracking()
            self.last_date = current_date

        # Update drawdown
        self._update_drawdown(current_value)

        # Process signals
        if self.signal[0] == 1:
            self.signal_count['buy'] += 1
            self._process_buy_signal()
        elif self.signal[0] == 2:
            self.signal_count['sell'] += 1
            self._process_sell_signal()

    # ========================================================================
    # SIGNAL PROCESSING
    # ========================================================================

    def _process_buy_signal(self):
        """
        Process BUY signal (signal=1).
        Opens LONG position with bracket order.
        """
        signal_time = self.data.datetime.datetime()
        can_trade, reason = self._validate_trade()

        if not can_trade:
            logger.debug(f"BUY Signal rejected: {reason}")
            return

        entry = self.data.close[0]

        if not self._validate_sl_tp(self.sl[0], self.target[0]):
            logger.warning(f"Invalid SL/TP for BUY: SL={self.sl[0]}, TP={self.target[0]}")
            return

        position_size = self._calculate_position_size(entry, self.sl[0])

        try:
            orders = self.buy_bracket(
                size=position_size,
                price=entry,
                stopprice=self.sl[0],
                limitprice=self.target[0]
            )

            trade_id = f"trade_buy_{self.signal_count['buy']}"
            self.pending_trades[trade_id] = TradeData(
                signal_type='BUY',
                signal_id=self.signal_count['buy'],
                datetime=signal_time,
                entry_price=entry,
                sl=self.sl[0],
                tp=self.target[0],
                size=position_size,
                orders=orders
            )

            self.current_positions += 1
            logger.info(f"BUY Order | Entry: ${entry:.2f} | SL: ${self.sl[0]:.2f} | TP: ${self.target[0]:.2f} | Size: {position_size}")

        except Exception as e:
            logger.error(f"BUY Order failed: {str(e)}")

    def _process_sell_signal(self):
        """
        Process SELL signal (signal=2).
        Opens SHORT position with bracket order.
        """
        signal_time = self.data.datetime.datetime()
        can_trade, reason = self._validate_trade()

        if not can_trade:
            logger.debug(f"SELL Signal rejected: {reason}")
            return

        entry = self.data.close[0]

        if not self._validate_sl_tp(self.sl[0], self.target[0]):
            logger.warning(f"Invalid SL/TP for SELL: SL={self.sl[0]}, TP={self.target[0]}")
            return

        position_size = self._calculate_position_size(entry, self.sl[0])

        try:
            orders = self.sell_bracket(
                size=position_size,
                price=entry,
                stopprice=self.sl[0],
                limitprice=self.target[0]
            )

            trade_id = f"trade_sell_{self.signal_count['sell']}"
            self.pending_trades[trade_id] = TradeData(
                signal_type='SELL',
                signal_id=self.signal_count['sell'],
                datetime=signal_time,
                entry_price=entry,
                sl=self.sl[0],
                tp=self.target[0],
                size=position_size,
                orders=orders
            )

            self.current_positions += 1
            logger.info(f"SELL Order | Entry: ${entry:.2f} | SL: ${self.sl[0]:.2f} | TP: ${self.target[0]:.2f} | Size: {position_size}")

        except Exception as e:
            logger.error(f"SELL Order failed: {str(e)}")

    # ========================================================================
    # VALIDATION & SIZING
    # ========================================================================

    def _validate_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed per risk management rules.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if self.current_positions >= self.p.max_positions:
            return False, f"Max positions reached ({self.current_positions}/{self.p.max_positions})"

        daily_loss_pct = (self.broker.getvalue() - self.daily_start_value) / self.daily_start_value
        if daily_loss_pct <= -self.p.max_daily_loss:
            return False, f"Daily loss limit ({daily_loss_pct*100:.1f}%)"

        if self.position:
            return False, "Position already open"

        return True, "Approved"

    @staticmethod
    def _validate_sl_tp(sl: float, tp: float) -> bool:
        """Validate stop loss and take profit values are present and positive."""
        return pd.notna(sl) and sl > 0 and pd.notna(tp) and tp > 0

    def _calculate_position_size(self, entry: float, sl: float) -> int:
        """
        Calculate position size based on risk percentage.

        Formula: size = (account_risk) / (price_risk)
        Where account_risk = account_value * risk_pct

        Args:
            entry: Entry price
            sl: Stop loss price

        Returns:
            Position size (bounded 1-100)
        """
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_pct
        price_risk = abs(entry - sl)

        if price_risk <= 0:
            return 1

        theoretical_size = risk_amount / price_risk
        max_position_value = account_value * 0.1
        max_size_by_value = max_position_value / entry
        position_size = min(theoretical_size, max_size_by_value)

        return max(1, min(100, int(position_size)))

    # ========================================================================
    # TRACKING & REPORTING
    # ========================================================================

    def _update_drawdown(self, current_value: float):
        """Update maximum drawdown from peak."""
        if current_value > self.peak_value:
            self.peak_value = current_value

        current_dd = (self.peak_value - current_value) / self.peak_value
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

    def _reset_daily_tracking(self):
        """Reset daily tracking variables at day boundary."""
        self.daily_start_value = self.broker.getvalue()

    def _record_daily_summary(self):
        """Record daily performance metrics."""
        current_value = self.broker.getvalue()
        daily_pnl = current_value - self.daily_start_value
        daily_return = (daily_pnl / self.daily_start_value * 100) if self.daily_start_value > 0 else 0

        self.daily_summary_table.append({
            'date': self.last_date,
            'start_value': self.daily_start_value,
            'end_value': current_value,
            'daily_pnl': daily_pnl,
            'daily_return_pct': daily_return,
            'trades_executed': self.trade_count,
            'positions_open': self.current_positions,
            'drawdown_pct': self.max_drawdown * 100
        })

    # ========================================================================
    # ORDER & TRADE CALLBACKS
    # ========================================================================

    def notify_order(self, order: bt.Order):
        """
        Called when order status changes.
        Tracks execution and matches exit orders to trades.
        """
        self.order_count += 1

        order_record = {
            'order_id': self.order_count,
            'datetime': self.data.datetime.datetime(),
            'type': 'BUY' if order.isbuy() else 'SELL',
            'size': order.size,
            'price': order.price if hasattr(order, 'price') else None,
            'status': self._get_order_status(order.status),
            'executed_price': order.executed.price if order.executed.price else None,
            'executed_size': order.executed.size if order.executed.size else None,
            'commission': order.executed.comm if order.executed.comm else None
        }
        self.orders_table.append(order_record)

        if order.status == order.Completed:
            logger.info(f"Order executed | {order_record['type']} {order_record['executed_size']} @ ${order_record['executed_price']:.2f}")
            self._handle_order_completion(order)

        elif order.status in [order.Margin, order.Rejected]:
            self.current_positions = max(0, self.current_positions - 1)
            logger.warning(f"Order failed | Status: {order_record['status']}")

    def _handle_order_completion(self, order: bt.Order):
        """Match completed orders to trades and close when appropriate."""
        for trade_id, trade_data in list(self.pending_trades.items()):
            if order not in trade_data.orders:
                continue

            if order.isbuy():
                # BUY order - entry or exit
                if trade_data.signal_type == 'BUY':
                    trade_data.actual_entry = order.executed.price
                    trade_data.actual_size = order.executed.size
                elif trade_data.signal_type == 'SELL':
                    # BUY order closes SHORT position
                    trade_data.exit_price = order.executed.price
                    trade_data.exit_size = order.executed.size
                    self._finalize_trade(trade_id, trade_data)
                    self.current_positions = max(0, self.current_positions - 1)

            elif order.issell():
                # SELL order - entry or exit
                if trade_data.signal_type == 'SELL':
                    trade_data.actual_entry = order.executed.price
                    trade_data.actual_size = order.executed.size
                elif trade_data.signal_type == 'BUY':
                    # SELL order closes LONG position
                    trade_data.exit_price = order.executed.price
                    trade_data.exit_size = order.executed.size
                    self._finalize_trade(trade_id, trade_data)
                    self.current_positions = max(0, self.current_positions - 1)
            break


    def _finalize_trade(self, trade_id: str, trade_data: TradeData):
        """
        Complete trade and record to trades journal.

        Calculates P&L, return %, exit type, and other metrics.
        Handles both LONG and SHORT positions correctly.
        """
        if not trade_data.actual_entry or not trade_data.exit_price:
            return

        self.trade_count += 1

        entry = trade_data.actual_entry
        exit_price = trade_data.exit_price
        size = abs(trade_data.actual_size)

        # P&L calculation (SHORT positions have reversed profit logic)
        if trade_data.signal_type == 'SELL':
            pnl = (entry - exit_price) * size
        else:
            pnl = (exit_price - entry) * size

        # Determine exit reason
        sl_dist = abs(exit_price - trade_data.sl)
        tp_dist = abs(exit_price - trade_data.tp)
        exit_type = "SL" if sl_dist < tp_dist else "TP"

        # Risk/reward metrics
        risk = abs(entry - trade_data.sl)
        reward = abs(trade_data.tp - entry)
        rr_ratio = reward / risk if risk > 0 else 0.0

        # Return percentage
        ret_pct = (pnl / (entry * size)) * 100 if entry > 0 else 0.0

        # Outcome
        outcome = "WIN" if pnl > 0 else "LOSS"
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Compute account balance based on previous trades + current P&L
        previous_balance = self.trades_table[-1]['account_balance'] if self.trades_table else self.p.cash
        account_balance = previous_balance + pnl

        # Record trade
        self.trades_table.append({
            'trade_id': self.trade_count,
            'signal_type': trade_data.signal_type,
            'entry_datetime': trade_data.datetime,
            'exit_datetime': self.data.datetime.datetime(),
            'entry_price': entry,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'size': size,
            'pnl': pnl,
            'return_pct': ret_pct,
            'outcome': outcome,
            'sl_level': trade_data.sl,
            'tp_level': trade_data.tp,
            'risk_amount': risk * size,
            'reward_amount': reward * size,
            'rr_ratio': rr_ratio,
            'account_balance': account_balance,
            'commission': 0.0
        })

        logger.info(f"Trade #{self.trade_count} closed | {trade_data.signal_type} | {outcome} | P&L: ${pnl:.2f}")

        if trade_id in self.pending_trades:
            del self.pending_trades[trade_id]

    @staticmethod
    def _get_order_status(status: int) -> str:
        """Convert order status integer to readable string."""
        status_map = {
            0: 'CREATED', 1: 'SUBMITTED', 2: 'ACCEPTED', 3: 'PARTIAL',
            4: 'COMPLETED', 5: 'EXPIRED', 6: 'CANCELLED', 7: 'MARGIN', 8: 'REJECTED'
        }
        return status_map.get(status, f'UNKNOWN_{status}')

    def stop(self):
        """
        Called when backtest ends.
        Finalizes remaining open positions and generates summary.
        """
        logger.info("Finalizing backtest execution")

        # Close remaining open trades at final price
        for trade_id, trade_data in list(self.pending_trades.items()):
            if trade_data.actual_entry:
                trade_data.exit_price = self.data.close[0]
                trade_data.exit_size = trade_data.actual_size
                self._finalize_trade(trade_id, trade_data)

        # Record final daily summary
        if self.last_date is not None:
            self._record_daily_summary()

        # Generate account summary
        final_value = self.broker.getvalue()
        starting_capital = self.p.cash
        total_return = (final_value - starting_capital) / starting_capital * 100
        total_signals = self.signal_count['buy'] + self.signal_count['sell']

        self.account_summary = {
            'starting_capital': starting_capital,
            'ending_capital': final_value,
            'total_pnl': final_value - starting_capital,
            'total_return_pct': total_return,
            'max_drawdown_pct': self.max_drawdown * 100,
            'buy_signals': self.signal_count['buy'],
            'sell_signals': self.signal_count['sell'],
            'total_signals': total_signals,
            'trades_executed': self.trade_count,
            'total_orders': self.order_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0.0,
            'signal_conversion_pct': (self.trade_count / total_signals * 100) if total_signals > 0 else 0.0
        }

        logger.info(f"Backtest complete | Return: {total_return:.2f}% | Trades: {self.trade_count} | Win Rate: {self.account_summary['win_rate_pct']:.1f}%")

# ============================================================================
# FRAMEWORK
# ============================================================================

class EnterpriseTradingFramework:
    """
    Main framework for running backtests and generating reports.

    Manages Cerebro engine setup, strategy execution, and result generation.
    Provides clean interface for backtest execution and result retrieval.
    """

    def __init__(self):
        """Initialize framework."""
        self.strategy = None
        logger.info("Enterprise Trading Framework initialized")

    def run_backtest(
        self,
        data_df: pd.DataFrame,
        config: Optional[TradeConfig] = None
    ) -> Dict:
        """
        Execute backtest with given data and configuration.

        Args:
            data_df: DataFrame with columns [open, high, low, close, volume, signal, sl, target]
                     Index must be datetime
            config: TradeConfig object with strategy parameters

        Returns:
            Dictionary containing:
              - account_summary: Overall performance metrics
              - trades_table: Individual trade details
              - orders_table: Order execution history
              - daily_summary: Daily performance breakdown
              - analyzers: Backtrader analyzer results
        """
        if config is None:
            config = TradeConfig()

        logger.info(f"Starting backtest | Capital: ${config.cash:,.0f} | Risk/Trade: {config.risk_pct*100}%")

        # Setup Cerebro
        cerebro = bt.Cerebro()
        data = TradeDataFeed(dataname=data_df)
        cerebro.adddata(data)
        cerebro.addstrategy(
            EnterpriseStrategy,
            risk_pct=config.risk_pct,
            max_positions=config.max_positions,
            max_daily_loss=config.max_daily_loss,
            cash=config.cash,
            commission=config.commission
        )

        # Broker setup
        cerebro.broker.set_cash(config.cash)
        cerebro.broker.setcommission(commission=config.commission)
        cerebro.broker.set_coc(True)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        # Execute
        results = cerebro.run()
        self.strategy = results[0]

        logger.info("Backtest completed successfully")
        return self._generate_results()

    def _generate_results(self) -> Dict:
        """Convert strategy results to clean DataFrames."""
        orders_df = pd.DataFrame(self.strategy.orders_table) if self.strategy.orders_table else pd.DataFrame()
        trades_df = pd.DataFrame(self.strategy.trades_table) if self.strategy.trades_table else pd.DataFrame()
        daily_df = pd.DataFrame(self.strategy.daily_summary_table) if self.strategy.daily_summary_table else pd.DataFrame()
        account_df = pd.DataFrame([self.strategy.account_summary]) if hasattr(self.strategy, 'account_summary') else pd.DataFrame()

        analyzers = {
            'sharpe': self.strategy.analyzers.sharpe.get_analysis(),
            'drawdown': self.strategy.analyzers.drawdown.get_analysis(),
            'returns': self.strategy.analyzers.returns.get_analysis()
        }

        logger.info("Results tables generated successfully")

        return {
            'account_summary': account_df,
            'trades_table': trades_df,
            'orders_table': orders_df,
            'daily_summary': daily_df,
            'analyzers': analyzers,
            'strategy': self.strategy
        }
