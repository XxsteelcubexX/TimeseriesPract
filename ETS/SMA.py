import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '~/Projects/Timeseries/Datasets/'
fileName = 'sp500_close.csv'

close = pd.read_csv(path+fileName, index_col= 0 , parse_dates= True)
close.head()
goog = close[['GOOG']].copy().dropna()
goog.head()
goog.plot()
# we are checking percentagechange for google by taking log and then adding Log
goog_ret = np.log(goog.pct_change(1)+1)
goog_ret.plot(figsize = (20,10))
#Moving Average
goog['SMA-10'] = goog["GOOG"].rolling(10).mean()
goog.head(20)
type(goog['GOOG'].rolling(10))
goog.plot(figsize= (20,10))
goog['SMA-50'] = goog["GOOG"].rolling(50).mean()
goog.plot(figsize= (20,10))

# multi-deminsional timeseries

goog_aapl = close[['GOOG','AAPL']].copy().dropna()
cov = goog_aapl.rolling(50).cov()
cov # this is 3 dimensional tensor in 2d database
# to get single covariance matrix for a particular date
cov.loc['2018-02-07'].to_numpy()
# Now calculating the log return for bith google and apple
goog_aapl_ret = np.log(1 + goog_aapl.pct_change(1))
goog_aapl_ret.head()
goog_aapl_ret['GOOG-SMA-50'] = goog_aapl_ret['GOOG'].rolling(50).mean()
goog_aapl_ret['AAPL-SMA-50'] = goog_aapl_ret['AAPL'].rolling(50).mean()
goog_aapl_ret.plot(figsize = (20,10))
cov = goog_aapl_ret[['GOOG-SMA-50','AAPL-SMA-50']].rolling(50).cov()
cov.tail()
corr = goog_aapl_ret[['GOOG-SMA-50','AAPL-SMA-50']].rolling(50).corr()
corr.tail(20)
