import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pprint import pprint

path = '~/Projects/Timeseries/Datasets/'
fileName = 'airline_passengers.csv'

df = pd.read_csv(path+fileName,index_col = 'Month', parse_dates = True)
df.index
df.index.freq = 'MS'

df.shape

#Assume that forecast horizon we care about is 12
# Validate over 10 steps
h=12
steps=10
Ntest = len(df)-h-steps+1
Ntest

#configure hyperparameters to try

trend_type_list = ['add','mul']
seasonal_type_list = ['add','mul']
damped_trend_list = [True,False]
init_method_list = ['estimated','heuristic','legacy-heuristic']
use_boxcox_list = [True, False, 0]# 0 is here for log transform

def walkForward(trend_type,seasonal_type,damped_trend,init_method,use_boxcox,debug = False):
    #stores errors
    errors = []
    seen_last = False
    steps_completed = 0

    for end_of_train in range(Ntest, len(df)-h +1):
        #we are just using index to get train and test data here.
        train = df.iloc[:end_of_train]
        test = df.iloc[end_of_train:end_of_train + h]

        if test.index[-1] == df.index[-1]:
            seen_last = True

        steps_completed +=1

        hw = ExponentialSmoothing(train['Passengers'],
        initialization_method= init_method,trend = trend_type,
        damped_trend=damped_trend,seasonal=seasonal_type,seasonal_periods=12,
        use_boxcox=use_boxcox)

        res_hw = hw.fit()

        fcast = res_hw.forecast(h)
        error = mean_squared_error(test['Passengers'],fcast)
        errors.append(error)

    if debug:
        print("seen_last : ", seen_last)
        print("steps_completed : ", steps_completed)

    return np.mean(errors)

#testing our function
walkForward('add','add',False,'legacy-heuristic',0,debug=True)

tuple_of_options = (trend_type_list,seasonal_type_list,damped_trend_list,
                    init_method_list,use_boxcox_list)
for x in itertools.product(*tuple_of_options):
    print(x)
best_score = float('inf')
best_option = None

piyLi = []
for x in itertools.product(*tuple_of_options):
    score = walkForward(*x)
    piyLi.append(([*x,score]))

    if score < best_score:
        print("Best Score So Far : ",score,'  |VARS|  ',x)
        best_score = score
        best_option = x

pprint(piyLi)

print("best_score : ", best_score)
print("best_options : ", best_option )
