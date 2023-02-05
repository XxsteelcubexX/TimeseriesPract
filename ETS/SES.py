import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df = pd.read_csv(path+fileName, index_col= 'Month', parse_dates= True)
df.head()

alpha = 0.2
df['EWMA'] = df['Passengers'].ewm(alpha = alpha, adjust = False).mean()
df.plot()

ses = SimpleExpSmoothing(df['Passengers'])# here we are passing data in constructer not in fit like other ML libs
df.index
df.index.freq = "MS"
df.index
ses = SimpleExpSmoothing(df['Passengers'], initialization_method='legacy-heuristic')
res = ses.fit(smoothing_level=alpha, optimized = False)
print(res)
res.predict(start = df.index[0], end = df.index[-1])
df['SES'] = res.predict(start = df.index[0], end= df.index[-1])
np.allclose(df['SES'],res.fittedvalues)
df.plot(figsize = (20,10))
# Ses output seems to shifted bt 1
df.head()
df['SES-1'] = df['SES'].shift(-1)
df.plot(figsize = (20,10));
df.head()
"""SES is the correct output according to the model and SES-1 not correct"""
#in the formula forcast is lagging behind 1 time innterval it should lag behind.
# if shift this and we need to shift everything.

""" Now we are ginng to treat it more like machine learning model"""

N_test = 12
train = df.iloc[:-N_test]
test = df.iloc[-N_test:]

ses = SimpleExpSmoothing(train['Passengers'],initialization_method = 'legacy-heuristic')
res = ses.fit()

# boolean series to index df rows
train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

df.loc[train_idx,'SESfitted'] = res.fittedvalues
df.loc[test_idx,'SESfitted'] = res.forecast(N_test)
df[['Passengers','SESfitted']].plot(figsize = (20,10))
res.params
