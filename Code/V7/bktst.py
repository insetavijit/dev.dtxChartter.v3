"""
Trading Backtest Framework - Enterprise Grade
Uses Backtrader's native analyzers and trade tracking.

Features:
  - BUY/SELL signal processing with bracket orders
  - Native Backtrader TradeAnalyzer for trading journal
  - Risk-based position sizing
  - Professional logging and reporting
  - Multiple concurrent positions support

Author: Trading Research Team
Version: 2.0.0
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
    """Configure professional logger with timestamp and level formatting."""
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
    risk_pct: float = 0.02
    max_positions: int = 5
    max_daily_loss: float = 0.05
    cash: float = 100000.0
    commission: float = 0.001

# ============================================================================
# SIGNAL DETECTION
# ============================================================================

class SignalDetector:
    """Automatic signal detection based on entry and stop loss levels."""

    @staticmethod
    def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect BUY/SELL signals from entry and SL columns.

        BUY signal (1): Stop loss is BELOW entry price
        SELL signal (2): Stop loss is ABOVE entry price
        NO signal (0): No entry or SL provided
        """
        df = df.copy()
        df['signal'] = 0

        valid_mask = df['entry'].notna() & df['sl'].notna()
        buy_mask = valid_mask & (df['sl'] < df['entry'])
        df.loc[buy_mask, 'signal'] = 1

        sell_mask = valid_mask & (df['sl'] > df['entry'])
        df.loc[sell_mask, 'signal'] = 2

        return df

# ============================================================================
# DATA FEED
# ============================================================================

class TradeDataFeed(bt.feeds.PandasData):
    """Custom Backtrader data feed for trading signals."""
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
    Professional trading strategy using Backtrader's native trade tracking.
    Processes BUY (1) and SELL (2) signals independently.
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
        logger.info("Initializing Enterprise Strategy v2.0")

        # Data line references
        self.signal = self.data.signal
        self.sl = self.data.sl
        self.target = self.data.target

        # Execution tracking (for logging only)
        self.orders_table = []
        self.signal_log = []
        self.trade_history = []  # Complete trade history table

        # Counters
        self.signal_count = {'buy': 0, 'sell': 0}
        self.order_count = 0
        self.current_positions = 0

        # Risk tracking
        self.max_drawdown = 0.0
        self.peak_value = self.broker.getvalue()
        self.daily_start_value = self.broker.getvalue()

        # Position tracking
        self.last_date = None
        self.trade_counter = 0
        self.active_trades = {}  # Track open trades by ID

        logger.info(f"Strategy initialized | Capital: ${self.broker.getvalue():,.2f}")

    def next(self):
        """Called on each bar. Processes signals independently."""
        current_date = self.data.datetime.date()
        current_value = self.broker.getvalue()

        # Daily boundary check
        if self.last_date != current_date:
            self._reset_daily_tracking()
            self.last_date = current_date

        # Update drawdown
        self._update_drawdown(current_value)

        # Process signals (independent of open positions)
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
        """Process BUY signal (signal=1)."""
        signal_time = self.data.datetime.datetime()
        can_trade, reason = self._validate_trade()

        if not can_trade:
            logger.debug(f"BUY Signal #{self.signal_count['buy']} rejected: {reason}")
            self.signal_log.append({
                'signal_id': self.signal_count['buy'],
                'type': 'BUY',
                'datetime': signal_time,
                'status': 'REJECTED',
                'reason': reason
            })
            return

        entry = self.data.close[0]

        if not self._validate_sl_tp(self.sl[0], self.target[0]):
            logger.warning(f"Invalid SL/TP for BUY: SL={self.sl[0]}, TP={self.target[0]}")
            return

        position_size = self._calculate_position_size(entry, self.sl[0])

        try:
            self.buy_bracket(
                size=position_size,
                price=entry,
                stopprice=self.sl[0],
                limitprice=self.target[0]
            )

            self.current_positions += 1
            logger.info(f"BUY Signal #{self.signal_count['buy']} | Entry: ${entry:.2f} | SL: ${self.sl[0]:.2f} | TP: ${self.target[0]:.2f} | Size: {position_size}")

            self.signal_log.append({
                'signal_id': self.signal_count['buy'],
                'type': 'BUY',
                'datetime': signal_time,
                'status': 'EXECUTED',
                'entry_price': entry,
                'sl': self.sl[0],
                'tp': self.target[0],
                'size': position_size
            })

        except Exception as e:
            logger.error(f"BUY Order failed: {str(e)}")

    def _process_sell_signal(self):
        """Process SELL signal (signal=2)."""
        signal_time = self.data.datetime.datetime()
        can_trade, reason = self._validate_trade()

        if not can_trade:
            logger.debug(f"SELL Signal #{self.signal_count['sell']} rejected: {reason}")
            self.signal_log.append({
                'signal_id': self.signal_count['sell'],
                'type': 'SELL',
                'datetime': signal_time,
                'status': 'REJECTED',
                'reason': reason
            })
            return

        entry = self.data.close[0]

        if not self._validate_sl_tp(self.sl[0], self.target[0]):
            logger.warning(f"Invalid SL/TP for SELL: SL={self.sl[0]}, TP={self.target[0]}")
            return

        position_size = self._calculate_position_size(entry, self.sl[0])

        try:
            self.sell_bracket(
                size=position_size,
                price=entry,
                stopprice=self.sl[0],
                limitprice=self.target[0]
            )

            self.current_positions += 1
            logger.info(f"SELL Signal #{self.signal_count['sell']} | Entry: ${entry:.2f} | SL: ${self.sl[0]:.2f} | TP: ${self.target[0]:.2f} | Size: {position_size}")

            self.signal_log.append({
                'signal_id': self.signal_count['sell'],
                'type': 'SELL',
                'datetime': signal_time,
                'status': 'EXECUTED',
                'entry_price': entry,
                'sl': self.sl[0],
                'tp': self.target[0],
                'size': position_size
            })

        except Exception as e:
            logger.error(f"SELL Order failed: {str(e)}")

    # ========================================================================
    # VALIDATION & SIZING
    # ========================================================================

    def _validate_trade(self) -> Tuple[bool, str]:
        """Risk management validation (daily loss only)."""
        current_daily_loss = (self.broker.getvalue() - self.daily_start_value) / self.daily_start_value
        if current_daily_loss <= -self.p.max_daily_loss:
            return False, f"Daily loss limit ({current_daily_loss*100:.1f}%)"

        return True, "Approved"

    @staticmethod
    def _validate_sl_tp(sl: float, tp: float) -> bool:
        """Validate stop loss and take profit values are present and positive."""
        return pd.notna(sl) and sl > 0 and pd.notna(tp) and tp > 0

    def _calculate_position_size(self, entry: float, sl: float) -> int:
        """Calculate position size based on risk percentage."""
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

    # ========================================================================
    # ORDER CALLBACKS
    # ========================================================================
    def notify_trade(self, trade):
        """Called when a trade is opened or closed. Tracks trade history."""
        if trade.isopen:
            # Trade opened: Store in active_trades with trade.ref as key
            self.trade_counter += 1
            self.active_trades[trade.ref] = {  # Use trade.ref as the key
                'type': 'BUY' if trade.size > 0 else 'SELL',
                'open_datetime': self.data.datetime.datetime(),
                'open_price': trade.price,
                'size': abs(trade.size),
                'account_balance_at_entry': self.broker.getvalue()
            }
            logger.info(f"Trade #{self.trade_counter} opened | {self.active_trades[trade.ref]['type']} | "
                        f"Price: ${trade.price:.2f} | Size: {abs(trade.size)}")

        elif trade.isclosed:
            # Trade closed: Record in trade_history and remove from active_trades
            trade_id = trade.ref  # Use trade.ref to find the trade
            trade_data = self.active_trades.get(trade_id)

            if trade_data is None:
                logger.warning(f"Closed trade (ref: {trade_id}) not found in active_trades: {trade}")
                return

            close_price = trade.price
            pnl = trade.pnlcomm  # P&L including commissions
            duration_bars = trade.barlen

            # Record trade in trade_history
            self.trade_history.append({
                'trade_id': self.trade_counter,  # Keep trade_counter for user-friendly ID
                'type': trade_data['type'],
                'open_datetime': trade_data['open_datetime'],
                'close_datetime': self.data.datetime.datetime(),
                'open_price': trade_data['open_price'],
                'close_price': close_price,
                'size': trade_data['size'],
                'pnl': pnl,
                'return_pct': (pnl / (trade_data['open_price'] * trade_data['size'])) * 100 if trade_data['open_price'] > 0 else 0,
                'account_balance_at_entry': trade_data['account_balance_at_entry'],
                'account_balance_at_exit': self.broker.getvalue(),
                'outcome': 'WIN' if pnl > 0 else 'LOSS',
                'duration_bars': duration_bars
            })

            self.active_trades.pop(trade_id)  # Remove from active_trades
            self.current_positions = max(0, self.current_positions - 1)
            logger.info(f"Trade #{self.trade_counter} closed | P&L: ${pnl:.2f} | "
                        f"Return: {self.trade_history[-1]['return_pct']:.2f}% | "
                        f"Duration: {duration_bars} bars")

    def notify_order(self, order: bt.Order):
        """Called when order status changes. Logs all order activity."""
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

        elif order.status in [order.Margin, order.Rejected]:
            self.current_positions = max(0, self.current_positions - 1)
            logger.warning(f"Order failed | Status: {order_record['status']}")

    @staticmethod
    def _get_order_status(status: int) -> str:
        """Convert order status integer to readable string."""
        status_map = {
            0: 'CREATED', 1: 'SUBMITTED', 2: 'ACCEPTED', 3: 'PARTIAL',
            4: 'COMPLETED', 5: 'EXPIRED', 6: 'CANCELLED', 7: 'MARGIN', 8: 'REJECTED'
        }
        return status_map.get(status, f'UNKNOWN_{status}')

    def stop(self):
        """Called when backtest ends. Close all open trades and generate history."""
        logger.info("Finalizing backtest execution")

        # Close all remaining open trades at final price
        for trade_id, trade_data in list(self.active_trades.items()):
            close_price = self.data.close[0]

            # Calculate P&L
            if trade_data['type'] == 'SELL':
                pnl = (trade_data['open_price'] - close_price) * trade_data['size']
            else:
                pnl = (close_price - trade_data['open_price']) * trade_data['size']

            # Record trade
            self.trade_history.append({
                'trade_id': trade_id,
                'type': trade_data['type'],
                'open_datetime': trade_data['open_datetime'],
                'close_datetime': self.data.datetime.datetime(),
                'open_price': trade_data['open_price'],
                'close_price': close_price,
                'size': trade_data['size'],
                'pnl': pnl,
                'return_pct': (pnl / (trade_data['open_price'] * trade_data['size'])) * 100 if trade_data['open_price'] > 0 else 0,
                'account_balance_at_entry': trade_data['account_balance_at_entry'],
                'account_balance_at_exit': self.broker.getvalue(),
                'outcome': 'WIN' if pnl > 0 else 'LOSS',
                'duration_bars': 0  # Would need bar counter for this
            })

            logger.info(f"Trade #{trade_id} closed at end of backtest | P&L: ${pnl:.2f}")

        # Log final summary
        final_value = self.broker.getvalue()
        starting_capital = self.p.cash
        total_return = (final_value - starting_capital) / starting_capital * 100
        total_signals = self.signal_count['buy'] + self.signal_count['sell']

        logger.info(f"Backtest complete | Return: {total_return:.2f}% | Max Drawdown: {self.max_drawdown*100:.2f}% | Signals: {total_signals} | Trades: {len(self.trade_history)}")

# ============================================================================
# FRAMEWORK
# ============================================================================

class EnterpriseTradingFramework:
    """Main framework for running backtests using native Backtrader analysis."""

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
            Dictionary containing native Backtrader analysis results
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

        # Add native Backtrader analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='timedrawdown')

        # Execute
        results = cerebro.run()
        self.strategy = results[0]

        logger.info("Backtest completed successfully")
        return self._generate_results()

    def _generate_results(self) -> Dict:
        """Extract native Backtrader analysis and execution logs."""

        # Native Backtrader analysis
        trade_analysis = self.strategy.analyzers.trades.get_analysis()
        sharpe_analysis = self.strategy.analyzers.sharpe.get_analysis()
        drawdown_analysis = self.strategy.analyzers.drawdown.get_analysis()
        returns_analysis = self.strategy.analyzers.returns.get_analysis()
        timedrawdown_analysis = self.strategy.analyzers.timedrawdown.get_analysis()

        # Convert execution logs to DataFrames
        orders_df = pd.DataFrame(self.strategy.orders_table) if self.strategy.orders_table else pd.DataFrame()
        signal_df = pd.DataFrame(self.strategy.signal_log) if self.strategy.signal_log else pd.DataFrame()
        trade_history_df = pd.DataFrame(self.strategy.trade_history) if self.strategy.trade_history else pd.DataFrame()

        # Debug: Log trade_history contents
        logger.info(f"Trade history before generating results: {self.strategy.trade_history}")
        logger.info(f"Trade history DataFrame shape: {trade_history_df.shape}")

        # Account summary
        account_summary = {
            'starting_capital': self.strategy.p.cash,
            'ending_capital': self.strategy.broker.getvalue(),
            'total_pnl': self.strategy.broker.getvalue() - self.strategy.p.cash,
            'total_return_pct': (self.strategy.broker.getvalue() - self.strategy.p.cash) / self.strategy.p.cash * 100,
            'max_drawdown_pct': self.strategy.max_drawdown * 100,
            'buy_signals': self.strategy.signal_count['buy'],
            'sell_signals': self.strategy.signal_count['sell'],
            'total_orders': self.strategy.order_count,
            'total_trades': len(self.strategy.trade_history)
        }

        logger.info("Results extracted successfully")

        return {
            'account_summary': pd.DataFrame([account_summary]),
            'trade_history': trade_history_df,          # Ensure this key is included
            'trading_journal': trade_analysis,
            'orders_table': orders_df,
            'signal_log': signal_df,
            'sharpe_ratio': sharpe_analysis,
            'drawdown_analysis': drawdown_analysis,
            'returns_analysis': returns_analysis,
            'timedrawdown_analysis': timedrawdown_analysis,
            'strategy': self.strategy
        }
