import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymc import Model, Normal, sample, HalfNormal
import arviz as az

import yfinance as yf
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import DiscreteAllocation
start = '2014-12-31'
end = '2022-12-31'
ext_factors = pd.read_parquet('factors.parquet')
stock_prices = pd.read_parquet('stock_data.parquet')
stock_prices = stock_prices.loc[start:end, :]

stock_rets = stock_prices.pct_change()
ext_factors = ext_factors.resample('M').agg(lambda x: (x + 1).prod() - 1)
stock_rets = stock_rets.resample('M').agg(lambda x: (x + 1).prod() - 1)

ext_factors[[f'{factor}_lag' for factor in ext_factors.columns]] = ext_factors.shift(1)
stock_rets[[f'{symbol}_lag' for symbol in stock_rets.columns]] = stock_rets.shift(1)

ext_factors = ext_factors.dropna()
stock_rets = stock_rets.dropna()

model_dict = {}

def predict(model, X):
        beta0 = model.posterior.beta0
        beta1 = model.posterior.beta1
        pred = X[0] * beta0 + X[1] * beta1
        pred = np.array(pred).reshape(20000, )
        return pd.Series(pred)
if __name__ == '__main__':
    # symbols = ['MINT.BK', 'BANPU.BK', 'TOP.BK', 'DELTA.BK', 'IRPC.BK', 'PTTGC.BK', 'WHA.BK', 'SCC.BK',
    #                'CPN.BK', 'CENTEL.BK', 'PTT.BK', 'BCP.BK']
    symbols = ['MINT.BK']
    for symbol in symbols:
        with Model() as model:
            beta0 = Normal('beta0', 0, 10)
            beta1 = Normal('beta1', 0, 10)
            beta2 = Normal('beta2', 0, 10)
            beta3 = Normal('beta3', 0, 10)
            beta4 = Normal('beta4', 0, 10)
            stdev = HalfNormal('stdev', 10)
            X_l1 = pm.MutableData( 'lagged_X', stock_rets.loc[:, f'{symbol}_lag'].values)
            R_l1 = pm.MutableData('lagged_R_l1', ext_factors.loc[:, 'mkt_lag'].values)
            R_l2 = pm.MutableData('lagged_R_l2', ext_factors.loc[:, 'E_lag'].values)
            R_l3 = pm.MutableData('lagged_R_l3', ext_factors.loc[:, 'S_lag'].values)
            R_l4 = pm.MutableData('lagged_R_l4', ext_factors.loc[:, 'G_lag'].values)
            X = stock_rets.loc[:, symbol].values
            mu = X_l1 * beta0 + R_l1 * beta1 + R_l2 * beta2 + R_l3 * beta3 + R_l4 * beta4
            obs = Normal('obs', mu=mu, sigma=stdev, observed=X)
            ar_trace = sample(20000, chains=1)
            model_dict[symbol] = ar_trace
        az.plot_trace(ar_trace, figsize=(10, 7))
        plt.tight_layout()
        plt.show()
    # num_periods = 12
    #
    # rets_dict = {}
    # for symbol in symbols:
    #     rets_dict[symbol] = predict(model_dict[symbol], [stock_data.loc[:, f'{symbol}_ret'][-1], df.loc[:, 'mkt'][-1]])
    #
    # views_dict = {}
    # views_uncertainty = {}
    # for symbol in rets_dict.keys():
    #     views_dict[symbol] = rets_dict[symbol].mean()
    #     views_uncertainty[symbol] = rets_dict[symbol].var()

