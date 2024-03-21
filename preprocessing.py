import pandas as pd

df = pd.read_csv('esg.csv')


df = df.drop(['Unnamed: 0'], axis=1)
df = df.set_index('Instrument', inplace=False)
# df = df.dropna(how='all')



char_to_del = ['f', 'ol', 'ss', 'va', 'm', 'z']
for char in char_to_del:
    df = df.loc[~df.index.str.contains(char, case=True), :]


df = df.reset_index()
df = df.set_index('Date')
# df_eod = pd.read_parquet('thai_stocks_eod_price')
# df_esg = pd.read_parquet('thai_esg_scores')
columns = []
for column in df.columns[1:]:
    columns.append(('2019-12-31', column))
columns_index = pd.MultiIndex.from_tuples(columns,
                                          names=['year', 'factors'])
df2 = pd.DataFrame(index=df.loc['2019-12-31', 'Instrument'], columns=columns_index)


for year in ['2019-12-31']:
    for factor in df.columns[1:]:
        df2[(year, factor)] = df.loc[year, factor].values
























