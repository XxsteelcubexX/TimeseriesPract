import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df=pd.read_csv(path+fileName, index_col = 'Month', parse_dates=True)
df.plot(figsize = (14,5))

adfuller(df['Passengers'])

def adf(x):
    res = adfuller(x)
    print("Test-Statistic:",res[0])
    print("P-Value:",res[1])
    if res[1] <0.05:
        print("Stationary")
    else:
        print("Non-Stationary")

adf(df['Passengers'])

# testing the function with simulated noise
adf(np.random.randn(100))

# testing with gamma function

gmaFn = np.random.gamma(1,1,100)
plt.plot(gmaFn)
plt.hist(gmaFn)
adf(gmaFn)

# now coming back to our dataset
# taking the log

df['LogPassengers'] = np.log(df['Passengers'])
adf(df['LogPassengers'])

df['Diff'] = df['Passengers'].diff()
df['Diff'].plot()
adf(df['Diff'].dropna())
df['DiffLog'] = df['LogPassengers'].diff()
df['DiffLog'].plot()

adf(df['DiffLog'].dropna())

##### testing with  stock prices
fileName = 'sp500sub.csv'

stocks = pd.read_csv(path+fileName, index_col = 'Date', parse_dates= True)
stocks.head()

goog = stocks[stocks['Name'] == 'GOOG'] [['Close']]
goog.head()
goog['LogPrice'] = np.log(goog['Close'])
goog['LogRet'] = goog['LogPrice'].diff()
goog
goog['LogPrice'].plot()
goog['LogRet'].plot()
adf(goog['LogPrice'])
adf(goog['LogRet'].dropna())

sbux = stocks[stocks['Name'] == 'SBUX'][['Close']]
sbux['LogPrice'] = np.log(sbux['Close'])
sbux['LogRet'] = sbux['LogPrice'].diff()

sbux['LogPrice'].plot()
sbux['LogRet'].plot()
adf(sbux['LogPrice'])

adf(sbux['LogRet'].dropna())
