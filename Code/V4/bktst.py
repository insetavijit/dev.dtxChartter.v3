import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PandasDataWithExtras(bt.feeds.PandasData):
    """Enhanced data feed with technical indicators and signals"""
    lines = ('rsi14', 'ema15', 'rsi_cross', 'sl', 'tp')
    params = (
        ('rsi14', 'RSI_14'),
        ('ema15', 'EMA_15'),
        ('rsi_cross', 'RSI_14_crossed_up_EMA_15'),
        ('sl', 'SL'),
        ('tp', 'TP'),
    )

class EnterpriseStrategy(bt.Strategy):
    """Professional trading strategy with structured table outputs"""

    params = (
        ('trade_size', 1),
        ('max_positions', 5),
        ('risk_per_trade', 0.02),
        ('max_daily_loss', 0.05),
        ('enable_logging', True),
    )

    def __init__(self):
        logger.info("Initializing Enterprise Strategy")

        # Data references
        self.signal = self.data.rsi_cross
        self.sl = self.data.sl
        self.tp = self.data.tp

        # Structured data tables
        self.orders_table = []
        self.trades_table = []
        self.account_table = []
        self.daily_summary_table = []

        # Tracking variables
        self.signal_count = 0
        self.trade_count = 0
        self.order_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_value = self.broker.getvalue()
        self.daily_start_value = self.broker.getvalue()
        self.current_positions = 0
        self.last_date = None

        # Order tracking
        self.pending_trades = {}

        logger.info(f"Strategy initialized - Starting Capital: ${self.broker.getvalue():,.2f}")

    def next(self):
        current_date = self.data.datetime.date()
        current_value = self.broker.getvalue()

        # Daily account summary
        if self.last_date != current_date:
            if self.last_date is not None:
                self._record_daily_summary()
            self._reset_daily_tracking()
            self.last_date = current_date

        # Update drawdown tracking
        self._update_drawdown_tracking(current_value)

        # Process signals
        if self.signal[0] == 1:
            self.signal_count += 1
            logger.debug(f"Signal #{self.signal_count} detected at {self.data.datetime.datetime()}")
            self._process_signal()

    def _process_signal(self):
        """Process trading signal with risk management"""
        signal_time = self.data.datetime.datetime()

        # Risk management check
        can_trade, reason = self._can_trade()

        if can_trade:
            entry = self.data.open[1] if len(self.data.open) > 1 else None
            if entry is None:
                logger.warning(f"Signal #{self.signal_count}: No next bar available")
                return

            # Calculate position size
            position_size = self._calculate_position_size(entry, self.sl[0])

            # Place bracket order
            try:
                orders = self.buy_bracket(
                    size=position_size,
                    price=entry,
                    stopprice=self.sl[0],
                    limitprice=self.tp[0]
                )

                # Store trade details
                trade_id = f"trade_{self.signal_count}"
                self.pending_trades[trade_id] = {
                    'signal_id': self.signal_count,
                    'datetime': signal_time,
                    'entry_price': entry,
                    'sl': self.sl[0],
                    'tp': self.tp[0],
                    'size': position_size,
                    'orders': orders
                }

                self.current_positions += 1
                logger.info(f"Order placed - Signal #{self.signal_count}, Size: {position_size}, Entry: {entry:.3f}")

            except Exception as e:
                logger.error(f"Order placement failed - Signal #{self.signal_count}: {str(e)}")
        else:
            logger.debug(f"Signal #{self.signal_count} rejected: {reason}")

    def _can_trade(self):
        """Risk management validation"""
        if self.current_positions >= self.p.max_positions:
            return False, f"Max positions ({self.current_positions}/{self.p.max_positions})"

        current_daily_loss = (self.broker.getvalue() - self.daily_start_value) / self.daily_start_value
        if current_daily_loss <= -self.p.max_daily_loss:
            return False, f"Daily loss limit ({current_daily_loss*100:.1f}%)"

        if self.position:
            return False, "Position already open"

        return True, "Approved"

    def _calculate_position_size(self, entry_price, sl_price):
        """Calculate risk-based position size"""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_per_trade
        price_risk = abs(entry_price - sl_price)

        if price_risk > 0:
            theoretical_size = risk_amount / price_risk
            max_position_value = account_value * 0.1
            max_size_by_value = max_position_value / entry_price
            position_size = min(theoretical_size, max_size_by_value)
            return max(1, min(100, int(position_size)))
        return 1

    def _update_drawdown_tracking(self, current_value):
        """Update maximum drawdown tracking"""
        if current_value > self.peak_value:
            self.peak_value = current_value
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def _reset_daily_tracking(self):
        """Reset daily tracking variables"""
        self.daily_start_value = self.broker.getvalue()

    def _record_daily_summary(self):
        """Record daily performance summary"""
        current_value = self.broker.getvalue()
        daily_pnl = current_value - self.daily_start_value

        self.daily_summary_table.append({
            'date': self.last_date,
            'start_value': self.daily_start_value,
            'end_value': current_value,
            'daily_pnl': daily_pnl,
            'daily_return_pct': daily_pnl / self.daily_start_value * 100,
            'total_trades': self.trade_count,
            'positions_open': self.current_positions,
            'drawdown_pct': self.max_drawdown * 100
        })

    def notify_order(self, order):
        """Order lifecycle tracking"""
        self.order_count += 1

        # Record order details
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
            logger.info(f"Order executed - {order_record['type']} {order_record['executed_size']} @ {order_record['executed_price']:.3f}")

            if order.isbuy():
                # Store entry execution
                for trade_id, trade_data in self.pending_trades.items():
                    if order in trade_data.get('orders', []):
                        trade_data['actual_entry'] = order.executed.price
                        trade_data['actual_size'] = order.executed.size
                        logger.debug(f"Entry recorded for {trade_id}")
                        break

            elif order.issell():
                # Process trade completion
                self._process_trade_completion(order)
                self.current_positions = max(0, self.current_positions - 1)

        elif order.status in [order.Margin, order.Rejected]:
            self.current_positions = max(0, self.current_positions - 1)
            logger.warning(f"Order failed - Status: {self._get_order_status(order.status)}")

    def _process_trade_completion(self, order):
        """Process completed trade and record to trades table"""
        for trade_id, trade_data in list(self.pending_trades.items()):
            if order in trade_data.get('orders', []) and 'actual_entry' in trade_data:
                self.trade_count += 1

                # Calculate trade metrics
                entry_price = trade_data['actual_entry']
                exit_price = order.executed.price
                size = trade_data['actual_size']
                pnl = (exit_price - entry_price) * size

                # Determine exit type
                sl_distance = abs(exit_price - trade_data['sl'])
                tp_distance = abs(exit_price - trade_data['tp'])
                exit_type = "SL" if sl_distance < tp_distance else "TP"

                # Calculate metrics
                risk = abs(entry_price - trade_data['sl'])
                reward = abs(trade_data['tp'] - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0
                trade_return_pct = (pnl / (entry_price * abs(size))) * 100

                # Determine outcome
                outcome = "WIN" if pnl > 0 else "LOSS"
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                # Record trade
                trade_record = {
                    'trade_id': self.trade_count,
                    'signal_id': trade_data['signal_id'],
                    'datetime_entry': trade_data['datetime'],
                    'datetime_exit': self.data.datetime.datetime(),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'size': abs(size),
                    'pnl': pnl,
                    'return_pct': trade_return_pct,
                    'outcome': outcome,
                    'sl_level': trade_data['sl'],
                    'tp_level': trade_data['tp'],
                    'risk_amount': risk * abs(size),
                    'reward_amount': reward * abs(size),
                    'rr_ratio': rr_ratio,
                    'account_balance': self.broker.getvalue(),
                    'commission': order.executed.comm if order.executed.comm else 0
                }

                self.trades_table.append(trade_record)
                logger.info(f"Trade #{self.trade_count} closed - {outcome}, P&L: ${pnl:.2f}")

                # Clean up
                del self.pending_trades[trade_id]
                break

    def _get_order_status(self, status):
        """Convert order status to readable string"""
        status_map = {
            0: 'CREATED', 1: 'SUBMITTED', 2: 'ACCEPTED', 3: 'PARTIAL',
            4: 'COMPLETED', 5: 'EXPIRED', 6: 'CANCELLED', 7: 'MARGIN', 8: 'REJECTED'
        }
        return status_map.get(status, f'UNKNOWN_{status}')

    def stop(self):
        """Finalize strategy and record account summary"""
        logger.info("Strategy execution completed")

        # Record final daily summary if needed
        if self.last_date is not None:
            self._record_daily_summary()

        # Create account summary
        final_value = self.broker.getvalue()
        total_return = (final_value - 100000) / 100000 * 100

        self.account_summary = {
            'starting_capital': 100000,
            'ending_capital': final_value,
            'total_pnl': final_value - 100000,
            'total_return_pct': total_return,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_signals': self.signal_count,
            'total_trades': self.trade_count,
            'total_orders': self.order_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0,
            'signal_conversion_pct': (self.trade_count / self.signal_count * 100) if self.signal_count > 0 else 0
        }

        logger.info(f"Final Results - Total Return: {total_return:.2f}%, Trades: {self.trade_count}, Win Rate: {self.account_summary['win_rate_pct']:.1f}%")

class EnterpriseTradingFramework:
    """Clean framework returning structured tables only"""

    def __init__(self):
        self.results = None
        self.strategy = None
        logger.info("Enterprise Trading Framework initialized")

    def run_backtest(self, data_df, **kwargs):
        """Execute backtest and return structured tables"""

        # Default parameters
        params = {
            'cash': 100000,
            'commission': 0.001,
            'trade_size': 1,
            'max_positions': 5,
            'risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'enable_logging': True
        }
        params.update(kwargs)

        logger.info(f"Starting backtest with ${params['cash']:,.0f} capital")

        # Initialize Cerebro
        cerebro = bt.Cerebro()

        # Add data and strategy
        data = PandasDataWithExtras(dataname=data_df)
        cerebro.adddata(data)
        cerebro.addstrategy(
            EnterpriseStrategy,
            trade_size=params['trade_size'],
            max_positions=params['max_positions'],
            risk_per_trade=params['risk_per_trade'],
            max_daily_loss=params['max_daily_loss'],
            enable_logging=params['enable_logging']
        )

        # Set broker parameters
        cerebro.broker.set_cash(params['cash'])
        cerebro.broker.setcommission(commission=params['commission'])
        cerebro.broker.set_coc(True)
        cerebro.addsizer(bt.sizers.FixedSize, stake=1)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        results = cerebro.run()
        self.strategy = results[0]
        self.results = results

        logger.info("Backtest completed successfully")

        return self._generate_structured_results()

    def _generate_structured_results(self):
        """Generate clean structured table results"""

        # Convert to DataFrames
        orders_df = pd.DataFrame(self.strategy.orders_table) if self.strategy.orders_table else pd.DataFrame()
        trades_df = pd.DataFrame(self.strategy.trades_table) if self.strategy.trades_table else pd.DataFrame()
        daily_summary_df = pd.DataFrame(self.strategy.daily_summary_table) if self.strategy.daily_summary_table else pd.DataFrame()

        # Account summary as DataFrame
        account_summary_df = pd.DataFrame([self.strategy.account_summary]) if hasattr(self.strategy, 'account_summary') else pd.DataFrame()

        # Analyzer results
        analyzers = {
            'sharpe': self.strategy.analyzers.sharpe.get_analysis(),
            'drawdown': self.strategy.analyzers.drawdown.get_analysis(),
            'returns': self.strategy.analyzers.returns.get_analysis(),
            'trades_analyzer': self.strategy.analyzers.trades.get_analysis()
        }

        logger.info("Results tables generated successfully")

        return {
            'orders_table': orders_df,
            'trades_table': trades_df,
            'account_summary': account_summary_df,
            'daily_summary': daily_summary_df,
            'analyzers': analyzers,
            'strategy': self.strategy
        }
