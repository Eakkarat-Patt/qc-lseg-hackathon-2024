import pandas as pd
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
import yfinance as yf
import numpy as np
df = pd.read_csv('allfactors.csv')



# Define the ticker symbol
ticker_symbol = 'AAPL'  # Example: Apple Inc.

# Get data on this ticker
ticker_data = yf.Ticker(ticker_symbol)



data = ticker_data.history(period='1d', start='2013-12-16', end='2023-09-28')
data.index = data.index
data['pf'] = np.log(data.Close / data.Close.shift(1))
data = data.loc['2014-01-02': '2023-09-28', :]

data.index = pd.DatetimeIndex(data.index.date)
df = df.set_index('date')
df.index = pd.DatetimeIndex(df.index)
df['pf'] = data['pf']
df = df.dropna()
df = df.round(5)
model = BayesianNetwork([('mkt', 'pf')])

model.fit(df)


infer = VariableElimination(model)
q = infer.query(variables=['pf'], evidence={'mkt': 0.00548})
print(q['pf'])
