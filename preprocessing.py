import pandas as pd
from urllib.request import urlopen
from pgmpy.models import BayesianNetwork

df = pd.read_csv('esg.csv')


df = df.drop(['Unnamed: 0'], axis=1)
df = df.set_index('Instrument')
df = df.dropna(how='all')

df.filter('')


char_to_del = ['f', 'ol', 'ss', 'va', 'm']
for char in char_to_del:
    df = df.loc[~df.index.str.contains(char, case=True), :]


































