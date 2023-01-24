import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, r2_score,mean_squared_error

try:
    path = '../Datasets/'
    fileName = 'SPY.csv'
    fileLoc = path + fileName
    df = pd.read_csv(fileLoc,index_col='Date',parse_dates = True)
except Exception as e:
    print(e)
    path = '/home/piyush/Projects/Timeseries/Datasets/'
    fileName = 'SPY.csv'
    fileLoc = path + fileName
    df = pd.read_csv(fileLoc,index_col='Date',parse_dates = True)

df.head()

# getting or naive forecasts
df['ClosePrediction'] = df['Close'].shift(1)
df.head()
y_true = df.iloc[1:]['Close']
y_pred = df.iloc[1:]['ClosePrediction']

# ███    ███ ███████ ████████ ██████  ██  ██████ ███████
# ████  ████ ██         ██    ██   ██ ██ ██      ██
# ██ ████ ██ █████      ██    ██████  ██ ██      ███████
# ██  ██  ██ ██         ██    ██   ██ ██ ██           ██
# ██      ██ ███████    ██    ██   ██ ██  ██████ ███████

#Main Idea: get a feel for how the values relate to one another. What's "good"?
#What's "bad"? If the R^2 is "good". will the MAE also be "good"?

(y_true - y_pred)

#1 SSE
sse = (y_true - y_pred).dot(y_true - y_pred)
print('#1 SSE : ',sse)

#2 MSE
mse = mean_squared_error(y_true,y_pred)
#MSE method 2
mse2 = (y_true-y_pred).dot(y_true-y_pred)/len(y_true)
print('#2 MSE :', mse , mse2)

#3 RMSE
rmse = mean_squared_error(y_true, y_pred, squared= False)
# method2
rmse2 = np.sqrt((y_true - y_pred).dot(y_true - y_pred)/len(y_true))
print('#3 RMSE :', rmse, rmse2)

#4 MAE
mae = mean_squared_error(y_true, y_pred)
print("#4 MAE", mae)

# Scale Independent Metrics below

#5 R^2
r2 = r2_score(y_true, y_pred)
print('#5 R^2 : ', r2)

#6 MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print('#6 MAPE', mape)

#7 SMAPE
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true)+np.abs(y_pred))/2
    ratio = numerator/denominator
    return ratio.mean()

print('#7 SMAPE', smape(y_true , y_pred))
