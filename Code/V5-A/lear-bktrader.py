from datetime import datetime
import backtrader as bt
import yfinance as yf
import pandas as pd

# ==============================
# 1️⃣ Download historical data
# ==============================
symbol = "MSFT"
start_date = "2011-01-01"
end_date = "2012-12-31"

# Force a single DataFrame
df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

# Ensure columns are named correctly for Backtrader
# Backtrader expects: 'open', 'high', 'low', 'close', 'volume', optional 'openinterest'
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.columns = ['open', 'high', 'low', 'close', 'volume']

# ==============================
# 2️⃣ Define SMA Crossover Strategy
# ==============================
class SmaCross(bt.Strategy):
    params = dict(
        pfast=10,  # fast SMA
        pslow=30   # slow SMA
    )

    def __init__(self):
        self.sma_fast = bt.ind.SMA(period=self.p.pfast)
        self.sma_slow = bt.ind.SMA(period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position:  # Not in position
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.close()

# ==============================
# 3️⃣ Backtrader setup
# ==============================
cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

# Feed data into Backtrader
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

# Set starting cash
cerebro.broker.set_cash(10000.0)
cerebro.broker.setcommission(commission=0.001)

# ==============================
# 4️⃣ Run backtest
# ==============================
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

# ==============================
# 5️⃣ Plot results
# ==============================
cerebro.plot()
