import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df = pd.read_csv(path+fileName, index_col='Month', parse_dates= True)
df.head()
df.plot(figsize=(14,5))
df['1stDiff'] = df['Passengers'].diff()
df.plot(figsize=(14,5))
df['LogPassengers'] = np.log(df['Passengers'])
df['LogPassengers'].plot(figsize=(14,5))

df.index.freq = 'MS'

Ntest = 12
train = df.iloc[:-Ntest]
test = df.iloc[-Ntest:]

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

arima = ARIMA(train['Passengers'],order=(1,0,0))
arimaResults = arima.fit()

df.loc[train_idx,'AR(1)'] = arimaResults.predict(
    start = train.index[0],end = train.index[-1])

df.columns
cols = ['Passengers','AR(1)']
df[cols].plot(figsize = (14,5))
predictionResult = arimaResults.get_forecast(Ntest)
forecast = predictionResult.predicted_mean
df.loc[test_idx,'AR(1)'] = forecast

df[cols].plot(figsize = (14,5))

type(predictionResult)

predictionResult.conf_int() # getting confidence intervals

def plot_fit_and_forecast(result):
    fig,ax = plt.subplots(figsize = (14,5))
    ax.plot(df['Passengers'], label = 'data')

    train_pred = result.fittedvalues
    ax.plot(train.index, train_pred, color = 'green', label = 'fitted')

    prediction_result = result.get_forecast(Ntest)
    conf_int = prediction_result.conf_int()
    lower,upper = conf_int['lower Passengers'], conf_int['upper Passengers']
    forecast = prediction_result.predicted_mean
    ax.plot(test.index, forecast, label = 'forecast')
    ax.fill_between(test.index, lower, upper , color = 'red', alpha = 0.3)
    ax.legend()

plot_fit_and_forecast(arimaResults)
#-----------------------------------------------------
arima = ARIMA(train['Passengers'], order = (10,0,0))
arimaResults = arima.fit()
plot_fit_and_forecast(arimaResults)

#-------------------------------------------
arima = ARIMA(train['Passengers'], order = (0,0,1))
arimaResults = arima.fit()
plot_fit_and_forecast(arimaResults)

######################################################
df['Log1stDiff'] = df['LogPassengers'].diff()
df['Log1stDiff'].plot(figsize = (14,5))

arima = ARIMA(train['Passengers'], order = (8,1,1))
arimaResults = arima.fit()
plot_fit_and_forecast(arimaResults)

# we are not plotting first row for difference as they will not present in the first row and will give errror
def plot_fit_and_forecast_int(results, d ,col = 'Passengers'):
    fig, ax = plt.subplots(figsize =(14,5))
    ax.plot(df[col], label = 'data')

    train_pred = results.predict(
        start = train.index[d], end = train.index[-1])

    ax.plot(train.index[d:], train_pred, color = 'green', label = 'fitted' )

    prediction_result = results.get_forecast(Ntest)
    conf_int = prediction_result.conf_int()
    lower , upper = conf_int[f'lower {col}'], conf_int[f'upper {col}']
    forecast = prediction_result.predicted_mean

    ax.plot(test.index, forecast, label = 'Forecast')
    ax.fill_between(test.index, lower, upper, color = 'red', alpha = 0.3)
    ax.legend()

arima = ARIMA(train['LogPassengers'], order=(8,1,1))
arimaResults = arima.fit()
plot_fit_and_forecast_int(arimaResults,1,col = 'LogPassengers')


arima = ARIMA(train['Passengers'], order=(12,1,0))
arimaResults = arima.fit()
plot_fit_and_forecast_int(arimaResults,1,col = 'Passengers')

arima = ARIMA(train['LogPassengers'], order=(12,1,0))
arimaResults = arima.fit()
plot_fit_and_forecast_int(arimaResults,1,col = 'LogPassengers')

def rmse(result, is_logged):
    forecast = result.forecast(Ntest)
    if is_logged:
        forecast = np.exp(forecast)

    t = test['Passengers']
    y = forecast
    return np.sqrt(np.mean((t-y)**2))

rmse(arimaResults,True)
