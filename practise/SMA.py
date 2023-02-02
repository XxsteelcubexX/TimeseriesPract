import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '~/Projects/Timeseries/Datasets/'
fileName = 'sp500_close.csv'
data = pd.read_csv(path+fileName,index_col= 0 , parse_dates= True)
data.head()
amzn = data[['AMZN']].dropna()
amzn.head()
amzn.plot(figsize = (20,10))
amzn_rect = np.log(amzn.pct_change(1)+1)
amzn_rect.plot(figsize = (20,10))
amzn['SMA_10'] = amzn[['AMZN']].rolling(10).mean()
amzn.plot(figsize = (30,15))
amzn['SMA_50'] = amzn[['AMZN']].rolling(50).mean()
amzn.plot(figsize = (30,15))
