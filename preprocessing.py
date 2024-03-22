import pandas as pd
import yfinance as yf

df = pd.read_csv('esg.csv')


df = df.drop(['Unnamed: 0'], axis=1)
df = df.set_index('Instrument', inplace=False)
# df = df.dropna(how='all')



char_to_del = ['f', 'ol', 'ss', 'va', 'm', 'z']
for char in char_to_del:
    df = df.loc[~df.index.str.contains(char, case=True), :]


df = df.reset_index()
df = df.set_index('Date')
df2 = pd.DataFrame(index=['2015-12-31', '2016-12-31', '2017-12-31',
                          '2018-12-31', '2019-12-31', '2020-12-31',
                          '2021-12-31', '2022-12-31'],
                   columns=df.loc['2022-12-31', 'Instrument'])


for symbol in df2.columns:
    df2[symbol] = df[df.Instrument == symbol]['Social Pillar Score']

gov_df = pd.DataFrame()
for year in ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']:
    low = df2.loc[f'{year}-12-31'].dropna().sort_values().iloc[:round(df2.loc[f'{year}-12-31'].dropna().shape[0]/2)].index.to_list()
    high = df2.loc[f'{year}-12-31'].dropna().sort_values().iloc[round(df2.loc[f'{year}-12-31'].dropna().shape[0]/2):].index.to_list()
    pf_low = yf.download(low, start=f"{year}-01-01", end=f"{year}-12-31")['Adj Close']
    pf_high = yf.download(high, start=f"{year}-01-01", end=f"{year}-12-31")['Adj Close']
    pf_low_ret = pf_low.pct_change()
    pf_high_ret = pf_high.pct_change()
    pf_low_ret['pf'] = pf_low_ret.sum(axis=1)
    pf_high_ret['pf'] = pf_high_ret.sum(axis=1)
    gov_df = pd.concat([gov_df, pf_high_ret['pf']-pf_low_ret['pf']])




