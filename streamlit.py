import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from pymc import Model, Normal, sample, HalfNormal
import arviz as az
import datetime
from dateutil.relativedelta import relativedelta
import time
import riskfolio as rp
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel
from pypfopt import DiscreteAllocation

# plt.style.use('dark_background')

class SteamLit:
    def __init__(self):
        self.end_test_at = None
        self.with_esg = None
        self.X_test = None
        self.X_train = None
        self.start_test_at = None
        self.end_train_at = None
        self.start_train_at = None
        self.stock_rets = None
        self.ext_factors = None
        self.stock_prices = None
        self.stock_name = None
        self.universe = None
        self.stock = None
        self.start = None
        self.end = None


        if 'run' not in st.session_state:
            st.session_state['run'] = False

    def click_button(self):
        st.session_state['run'] = True

    def display_input(self):

        self.universe = st.multiselect('Stock Name',
                                       tuple(self.stock_name),
                                       ['BANPU.BK', 'BCP.BK', 'CENTEL.BK', 'CPN.BK', 'DELTA.BK', 'IRPC.BK', 'MINT.BK',
                                        'PTT.BK', 'PTTGC.BK', 'SCC.BK', 'TOP.BK', 'WHA.BK'])
        # self.universe = st.multiselect('Stock Name',
        #                                tuple(self.stock_name),
        #                                ['BANPU.BK', 'BCP.BK'])
        print(self.universe)

        self.start = st.date_input('Start Train',
                                   value=datetime.date(2015, 12, 31),
                                   min_value=self.stock_prices.index[0].date(),
                                   max_value=(self.stock_prices.index[-1] - relativedelta(months=2)).date())


        self.end = st.date_input('Start Test',
                                 value=datetime.date(2022, 1, 31),
                                 min_value=self.stock_prices.index[0].date(),
                                 max_value=(self.stock_prices.index[-1] - relativedelta(months=1)).date())

        # self.end = st.date_input('Start Test',
        #                          value=datetime.date(2017, 1, 31),
        #                          min_value=self.stock_prices.index[0].date(),
        #                          max_value=(self.stock_prices.index[-1] - relativedelta(months=1)).date())
        self.start_train_at = pd.to_datetime(self.start)
        self.end_train_at = pd.to_datetime(self.end - datetime.timedelta(days=1))
        self.start_test_at = pd.to_datetime(self.end)
        self.end_test_at = pd.to_datetime(self.end + relativedelta(months=1))
        self.with_esg = st.toggle('Include ESG Factors', True)
        print(self.start_train_at, self.end_train_at, self.start_test_at)
        if not self.universe:
            st.warning('The stock list is empty!')
        else:

            st.button("Run", on_click=self.click_button)

    def get_data(self):
        self.stock_prices = pd.read_parquet('stock_prices.parquet')
        self.stock_name = self.stock_prices.columns.to_list()
        self.ext_factors = pd.read_parquet('factors.parquet')
        max_start = max(self.stock_prices.index[0], self.ext_factors.index[0])
        min_end = min(self.stock_prices.index[-1], self.ext_factors.index[-1])
        self.stock_prices = self.stock_prices.loc[max_start:min_end]
        self.ext_factors = self.ext_factors.loc[max_start:min_end]

    def pre_preprocess(self):
        # self.stock_prices = self.stock_prices.loc[self.start:self.end, :]
        # self.ext_factors = self.ext_factors.loc[self.start:self.end, :]
        # # self.stock_prices = self.stock_prices.loc[self.start:'2016-12-31', :]
        # # self.ext_factors = self.ext_factors.loc[self.start:'2016-12-31', :]

        self.stock_rets = self.stock_prices.pct_change()
        self.ext_factors = self.ext_factors.resample('M').agg(lambda x: (x + 1).prod() - 1)
        self.stock_rets = self.stock_rets.resample('M').agg(lambda x: (x + 1).prod() - 1)

        self.ext_factors[[f'{factor}_lag' for factor in self.ext_factors.columns]] = self.ext_factors.shift(1)
        self.stock_rets[[f'{symbol}_lag' for symbol in self.stock_rets.columns]] = self.stock_rets.shift(1)

        self.ext_factors = self.ext_factors.dropna()
        self.stock_rets = self.stock_rets.dropna()
        self.X_train = pd.concat(
            [self.stock_rets.loc[self.start_train_at:self.end_train_at, :],
             self.ext_factors.loc[self.start_train_at:self.end_train_at, :]], axis=1)
        self.X_test = pd.concat(
            [self.stock_rets.loc[self.start_test_at:, :], self.ext_factors.loc[self.start_test_at:, :]], axis=1)
        print(self.X_test)
        print(self.X_train)

    def train(self):
        model_dict = {}
        for symbol in self.universe:
            with Model() as model:
                if self.with_esg:
                    print('yes')
                    beta0 = Normal('beta0', 0, 10)
                    beta1 = Normal('beta1', 0, 10)
                    beta2 = Normal('beta2', 0, 10)
                    beta3 = Normal('beta3', 0, 10)
                    beta4 = Normal('beta4', 0, 10)
                    stdev = HalfNormal('stdev', 10)
                    X_l1 = pm.MutableData('lagged_X', self.X_train.loc[:, f'{symbol}_lag'].values)
                    R_l1 = pm.MutableData('lagged_R_l1', self.X_train.loc[:, 'mkt_lag'].values)
                    R_l2 = pm.MutableData('lagged_R_l2', self.X_train.loc[:, 'E_lag'].values)
                    R_l3 = pm.MutableData('lagged_R_l3', self.X_train.loc[:, 'S_lag'].values)
                    R_l4 = pm.MutableData('lagged_R_l4', self.X_train.loc[:, 'G_lag'].values)
                    X = self.X_train.loc[:, symbol].values
                    mu = X_l1 * beta0 + R_l1 * beta1 + R_l2 * beta2 + R_l3 * beta3 + R_l4 * beta4
                    obs = Normal('obs', mu=mu, sigma=stdev, observed=X)
                    ar_trace = sample(20000, chains=1)
                    model_dict[symbol] = ar_trace
                else:
                    beta0 = Normal('beta0', 0, 10)
                    beta1 = Normal('beta1', 0, 10)
                    stdev = HalfNormal('stdev', 10)
                    X_l1 = pm.MutableData('lagged_X', self.X_train.loc[:, f'{symbol}_lag'].values)
                    R_l1 = pm.MutableData('lagged_R_l1', self.X_train.loc[:, 'mkt_lag'].values)
                    X = self._train.loc[:, symbol].values
                    mu = X_l1 * beta0 + R_l1 * beta1
                    obs = Normal('obs', mu=mu, sigma=stdev, observed=X)
                    ar_trace = sample(20000, chains=1)
                    model_dict[symbol] = ar_trace
        return model_dict

    def predict(self, X, symbol, model):
        draw = model.posterior['draw'].shape[0]
        beta0_sample = model.posterior['beta0']
        beta1_sample = model.posterior['beta1']
        if self.with_esg:
            beta2_sample = model.posterior['beta2']
            beta3_sample = model.posterior['beta3']
            beta4_sample = model.posterior['beta4']
            next_period_return = (X[f'{symbol}_lag'] * beta0_sample +
                                  X['mkt_lag'] * beta1_sample +
                                  X['E_lag'] * beta2_sample +
                                  X['S_lag'] * beta3_sample +
                                  X['G_lag'] * beta4_sample)
        else:
            next_period_return = (X[f'{symbol}_lag'] * beta0_sample +
                                  X['mkt_lag'] * beta1_sample)

        next_period_return = np.array(next_period_return).reshape(draw, )
        return next_period_return

    def fit(self,model_dict):

        st.write('Starting test: ',self.start_test_at.date())
        st.write('Ending test: ', self.end_test_at.date())
        portfolio = self.stock_prices.loc[self.start_test_at:self.end_test_at,
                    self.universe]
        S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()

        views = {}
        view_uncertainty = {}
        for symbol in self.universe:
            pred = self.predict(self.X_test.loc[(self.start_test_at + relativedelta(months=1)), :], symbol,
                                model_dict[symbol])
            views[symbol] = pred.mean()
            view_uncertainty[symbol] = pred.var()

        omega = np.diag(list(view_uncertainty.values()))

        bl = BlackLittermanModel(S, pi="equal",
                                 absolute_views=views, omega=omega)

        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        ef = EfficientFrontier(ret_bl, S_bl)
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe(risk_free_rate=0.0227 / 12)
        weights = ef.clean_weights()

        return weights, ef.portfolio_performance(risk_free_rate=0.0227 / 12, verbose=True)

    def plot(self):
        pass
        # az.plot_trace(self.ar_trace, figsize=(10, 7))
        # plt.tight_layout()
        # st.pyplot(plt)
        # st.table(az.summary(self.ar_trace, kind="stats"))

    def stream_data(self,weights,perf):

        weights_df = pd.DataFrame(list(weights.items()), columns=['Stock', 'Weight'])

        st.write('Expected annual return (%):', round(perf[0]*100,2))
        st.write('Annual volatility (%):', round(perf[1] * 100, 2))
        st.write('Sharpe Ratio:', round(perf[2], 2))

        weights_df_plot = weights_df
        weights_df_plot.index = weights_df_plot['Stock']
        weights_df_plot = weights_df_plot.drop(columns=['Stock'])

        fig, ax = plt.subplots(figsize=(10, 6))
        rp.plot_pie(w=weights_df_plot, title='Asset Allocation', others=0.05, nrow=25,
                    cmap="tab20", ax=ax)
        st.pyplot(fig)





    def run(self):

        self.get_data()
        self.display_input()

        if st.session_state.run:
            self.pre_preprocess()
            model_dict = self.train()
            weights, perf = self.fit(model_dict)
            self.stream_data(weights,perf)


if __name__ == "__main__":
    app = SteamLit()
    app.run()
