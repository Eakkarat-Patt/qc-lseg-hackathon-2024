import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymc import HalfCauchy, Model, Normal, sample, HalfNormal
import arviz as az

import yfinance as yf
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import DiscreteAllocation
start = '2014-03-31'
end = '2023-09-30'
df = pd.read_csv('allfactors.csv')
df['date'] = pd.DatetimeIndex(df['date'])
df = df.set_index('date')
stock_data = pd.read_csv('stock_data_adj.csv')
stock_data['Date'] = pd.DatetimeIndex(stock_data['Date'])
stock_data = stock_data.set_index('Date')
stock_data = stock_data.dropna(axis=1)
stock_data = stock_data.resample('ME').last()

for ticker in stock_data.columns:
    stock_data.loc[:, f'{ticker}_ret'] = stock_data.loc[:, ticker].pct_change(fill_method=None)
    stock_data.loc[:, f'{ticker}_ret_lag'] = stock_data.loc[:, f'{ticker}_ret'].shift(1)
stock_data = stock_data.filter(like='ret')
stock_data = stock_data.dropna()
df = df.resample('ME').agg(lambda x: (x + 1).prod() - 1)
df.loc[:, 't_lag'] = df.loc[:, 'Total'].shift(1)

df = df.dropna()
df = df.loc[start:end, :]
stock_data = stock_data.loc[start:end, :]
model_dict = {}

def predict(model, X):
        beta0 = model.posterior.beta0
        beta1 = model.posterior.beta1
        pred = X[0] * beta0 + X[1] * beta1
        pred = np.array(pred).reshape(20000, )
        return pd.Series(pred)
if __name__ == '__main__':
    symbols = ['MINT.BK', 'BANPU.BK', 'TOP.BK', 'DELTA.BK', 'IRPC.BK', 'PTTGC.BK', 'WHA.BK', 'SCC.BK',
                   'CPN.BK', 'CENTEL.BK', 'PTT.BK', 'BCP.BK']
    for symbol in symbols:
        with Model() as model:
            beta0 = Normal('beta0', 0, 10)
            beta1 = Normal('beta1', 0, 0.001)
            stdev = HalfNormal('stdev', 10)
            X_im1 = pm.MutableData( 'lagged_X', stock_data[f'{symbol}_ret_lag'].values)
            R_im1 = pm.MutableData('lagged_R_im1', df['t_lag'].values)
            X_i = stock_data[f'{symbol}_ret'].values
            mu = X_im1 * beta0 + R_im1 * beta1
            obs = Normal('obs', mu=mu, sigma=stdev, observed=X_i)
            ar_trace = sample(20000, chains=1)
            model_dict[symbol] = ar_trace
        # az.plot_trace(ar_trace, figsize=(10, 7))
        # plt.tight_layout()
        # plt.show()
    num_periods = 12

    rets_dict = {}
    for symbol in symbols:
        rets_dict[symbol] = predict(model_dict[symbol], [stock_data.loc[:, f'{symbol}_ret'][-1], df.loc[:, 'mkt'][-1]])

    views_dict = {}
    views_uncertainty = {}
    for symbol in rets_dict.keys():
        views_dict[symbol] = rets_dict[symbol].mean()
        views_uncertainty[symbol] = rets_dict[symbol].var()

