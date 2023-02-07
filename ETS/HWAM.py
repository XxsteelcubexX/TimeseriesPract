import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df = pd.read_csv(path+fileName, index_col='Month', parse_dates = True)
df.index.freq = 'MS'

df.index

df['SMA_12'] = df[['Passengers']].rolling(12).mean()

alpha = 0.2
df['EWMA'] = df[['Passengers']].ewm(alpha = alpha, adjust = False).mean()

N_test = 12
train = df.iloc[:-N_test]
test = df.iloc[-N_test:]
train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

ses = SimpleExpSmoothing(train['Passengers'],initialization_method='legacy-heuristic')
res = ses.fit()
df.loc[train_idx,'SES'] = res.fittedvalues
df.loc[test_idx,'SES'] = res.forecast(N_test)

holt = Holt(train['Passengers'], initialization_method='legacy-heuristic')
holtRes = holt.fit()
df.loc[train_idx,'Holt'] = holtRes.fittedvalues
df.loc[test_idx,'Holt'] = holtRes.forecast(N_test)

# variation One

hw = ExponentialSmoothing(train['Passengers'],initialization_method='legacy-heuristic',
trend = 'add',seasonal='add',seasonal_periods=12)
hwRes = hw.fit()

df.loc[train_idx,'HWADD'] = hwRes.fittedvalues
df.loc[test_idx,'HWADD'] = hwRes.forecast(N_test)
df[['Passengers','Holt','HWADD']].plot(figsize = (30,10))

def rmse(y,t):
    return np.sqrt(np.mean((y-t)**2))

def mae(y,t):
    return np.mean(np.abs(y-t))


print("Train RMSE : ", rmse(train['Passengers'],hwRes.fittedvalues))
print("Test RMSE : ",rmse(test['Passengers'],hwRes.forecast(N_test)))
print("Train MAE : ", mae(train['Passengers'],hwRes.fittedvalues))
print("Test MAE : ", mae(test['Passengers'],hwRes.forecast(N_test)))

#variation Two

hw = ExponentialSmoothing(train['Passengers'],initialization_method='legacy-heuristic',
trend = 'add', seasonal='mul',seasonal_periods= 12)
hwRes = hw.fit()
df.loc[train_idx,'HWMUL'] = hwRes.fittedvalues
df.loc[test_idx,'HWMUL'] = hwRes.forecast(N_test)
df[['Passengers','Holt','HWADD','HWMUL']].plot(figsize = (30,10))z

print("Train RMSE : ", rmse(train['Passengers'],hwRes.fittedvalues))
print("Test RMSE : ",rmse(test['Passengers'],hwRes.forecast(N_test)))
print("Train MAE : ", mae(train['Passengers'],hwRes.fittedvalues))
print("Test MAE : ", mae(test['Passengers'],hwRes.forecast(N_test)))


#variation 3

hw = ExponentialSmoothing(train['Passengers'],initialization_method='legacy-heuristic',
trend = 'mul', seasonal='mul',seasonal_periods= 12)
hwRes = hw.fit()
df.loc[train_idx,'HW2XMUL'] = hwRes.fittedvalues
df.loc[test_idx,'HW2XMUL'] = hwRes.forecast(N_test)
df[['Passengers','HWADD','HWMUL','HW2XMUL']].plot(figsize = (30,10))
plt.show()

print("Train RMSE : ", rmse(train['Passengers'],hwRes.fittedvalues))
print("Test RMSE : ",rmse(test['Passengers'],hwRes.forecast(N_test)))
print("Train MAE : ", mae(train['Passengers'],hwRes.fittedvalues))
print("Test MAE : ", mae(test['Passengers'],hwRes.forecast(N_test)))
