import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt

path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df = pd.read_csv(path+fileName,index_col = 'Month', parse_dates = True)

df.index.freq = 'MS'
df.index

alpha = 0.2
df['EWMA'] = df['Passengers'].ewm(alpha = alpha, adjust = False).mean()

ses = SimpleExpSmoothing(df['Passengers'], initialization_method= 'legacy-heuristic')
res = ses.fit(smoothing_level=alpha, optimized= False)

df['SES']= res.predict(start = df.index[0], end = df.index[-1])
print(df.head(),df.tail(),sep = '\n')

N_test = 12
train = df.iloc[:-N_test]
test = df.iloc[-N_test:]

ses = SimpleExpSmoothing(train['Passengers'], initialization_method = 'legacy-heuristic')
res = ses.fit()

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

df.loc[train_idx,'SESfitted'] = res.fittedvalues
df.loc[test_idx, 'SESfitted'] = res.predict(N_test)

holt = Holt(df['Passengers'], initialization_method='legacy-heuristic')
res_holt = holt.fit()

df['Holt'] = res_holt.fittedvalues

df[['Passengers','Holt']].plot(figsize = (20,10))

holt = Holt(train['Passengers'], initialization_method= 'legacy-heuristic')
res_holt = holt.fit()

df.loc[train_idx,'Holt'] = res_holt.fittedvalues
df.loc[test_idx, 'Holt'] = res_holt.forecast(N_test)

df[['Passengers','Holt']].plot(figsize=(20,10))
holt.params
