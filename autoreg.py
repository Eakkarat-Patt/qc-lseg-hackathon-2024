import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import pyplot as plt
import yfinance as yf

ticker = 'PTT.BK'
start = '2014-01-02'
end = '2023-09-28'
stock_data = yf.download(ticker, start, end)
df = pd.DataFrame()
df['stock'] = stock_data.loc[:, 'Adj Close']

df.loc[:, 'ret'] = df['stock'].pct_change(fill_method=None)
df = df.resample('M').agg(lambda x: (x + 1).prod() - 1)
# df.loc[:, 'lagged_ret'] = df.loc[:, 'ret'].shift(1)
df = df.dropna()

with pm.Model() as AR:
    pass

t_data = list(range(df.shape[0]))

AR.add_coord('obs_id', t_data, mutable=True)

