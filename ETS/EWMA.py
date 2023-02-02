import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df = pd.read_csv(path+fileName,index_col = 'Month', parse_dates = True)
df.head()
df.isna().sum()
df.plot()
alpha = 0.2 # we have randomly chosen this alpha value we can optimise it
df['EWMA'] = df['Passengers'].ewm(alpha = alpha, adjust= False).mean()
#adjust False is thier so that doesn't use any other calculations
type(df['Passengers'].ewm(alpha=alpha, adjust = False))
df.plot(figsize = (20,10))

manual_ewma = []
for x in df['Passengers'].to_numpy():
    if len(manual_ewma)>0:
        xhat = alpha*x + (1-alpha)*manual_ewma[-1]
    else:
        xhat = x
    manual_ewma.append(xhat)
df['Manual_EWMA'] = manual_ewma
df.head(10)
