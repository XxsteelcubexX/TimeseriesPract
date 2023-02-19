import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
from sklearn.metrics import r2_score

path = '/home/piyush/Projects/Timeseries/Datasets/'
fileName = 'sp500sub.csv'


data = pd.read_csv(path+fileName, index_col= 0 , parse_dates=True)

df = data[data['Name'] == 'GOOG' ][['Close']].copy()
df.head()
df.tail()
df.plot(figsize=(25,10))
df.index

df['LogClose'] = np.log(df['Close'])
df['LogClose'].plot(figsize = (25,10))
Ntest = 30
train = df.iloc[:-Ntest]
test = df.iloc[-Ntest:]

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

hw = ExponentialSmoothing(train['LogClose'],initialization_method="legacy-heuristic" ,
        trend = 'add')
res = hw.fit()
df.loc[train_idx, 'HWtrain'] = res.fittedvalues
df.loc[test_idx, 'HWtest'] = res.forecast(Ntest).to_numpy()
df[['LogClose','HWtrain','HWtest']].plot(figsize =(14,7))
cols = ['LogClose','HWtrain','HWtest']
df.iloc[-100:][cols].plot(figsize = (14,7))
# Naive Forecasts

df['NaiveTrain'] = train[['LogClose']].shift(1)

df.loc[test_idx,'NaiveTest'] = train.iloc[-1,1]
df.tail(40)

new_cols = ['LogClose','HWtrain','HWtest','NaiveTrain','NaiveTest']
df.iloc[-100:][new_cols].plot(figsize = (14,7))
