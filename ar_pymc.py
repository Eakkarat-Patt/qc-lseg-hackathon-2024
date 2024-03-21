import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymc import HalfCauchy, Model, Normal, sample, HalfNormal
import arviz as az
import yfinance as yf

df = pd.read_csv('allfactors.csv')
df['date'] = pd.DatetimeIndex(df['date'])
df = df.set_index('date')


ticker = 'GULF.BK'
start = '2014-01-02'
end = '2023-09-28'
stock_data = yf.download(ticker, start, end)

df['stock'] = stock_data.loc[:, 'Adj Close']

df.loc[:, 'ret'] = df['stock'].pct_change(fill_method=None)
df = df.resample('ME').agg(lambda x: (x + 1).prod() - 1)
# df = df.resample('M').last()
df.loc[:, 'mkt_lag'] = df.loc[:, 'mkt'].shift(1)
df.loc[:, 'ret_lag'] = df.loc[:, 'ret'].shift(1)

df = df.dropna()
pred_dict = {}
if __name__ == '__main__':
    with Model() as model:
        beta0 = Normal('beta0', 0, 10)
        beta1 = Normal('beta1', 0, 10)
        stdev = HalfNormal('stdev', 10)
        X_im1 = pm.MutableData( 'lagged_X', df['ret_lag'].values)
        R_im1 = pm.MutableData('lagged_mkt', df['mkt_lag'].values)
        X_i = df['ret'].values
        mu = X_im1 * beta0 + R_im1 * beta1
        obs = Normal('obs', mu=mu, sigma=stdev, observed=X_i)
        ar_trace = sample(20000)
    az.plot_trace(ar_trace, figsize=(10, 7))
    plt.tight_layout()
    plt.show()
    # with model:
    #     X_im1.set_value(np.array([0.1]))
    #     sample_params = pm.sample_posterior_predictive(ar_trace, predictions=True)