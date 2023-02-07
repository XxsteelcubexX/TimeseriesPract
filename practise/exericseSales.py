import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import itertools

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

path = '/home/piyush/Projects/Timeseries/Datasets/'
fileName = 'perrin-freres-monthly-champagne.csv'
#option 2 df = pd.read_csv(path+fileName,index_col='Month', skipfooter = 2, parse_dates= True)

df = pd.read_csv(path+fileName,index_col='Month', parse_dates= True)
#df.tail()
df = df[:-2]
#df.tail()
df.index = pd.to_datetime(df.index)
df.index
df.index.freq = 'MS'
df.index
df.columns = ['Monthly_Sales']
df.plot(figsize = (30,10))
plt.show()

df.shape
h = 12
steps = 10
N_test = len(df)-h-steps+1

trend_List = ['add','mul']
seasonal_List = ['add','mul']
damped_trend_list = [True, False]
init_method_list = ['estimated','heuristic','legacy-heuristic']
use_boxcox_list = [True,False,0]
crash_counter = 0

def walkForward(trend,seasonal,damped_trend,init_method,use_boxcox,debug = False):

    errors = []
    last_seen = False
    steps_completed = 0
    global crash_counter

    for lastIndexOfTrain in range(N_test, len(df)-h +1):
        train = df.iloc[:lastIndexOfTrain]
        test = df.iloc[lastIndexOfTrain : lastIndexOfTrain+h]

        if test.index[-1] == df.index[-1]:
            last_seen = True

        steps_completed  += 1

        hw = ExponentialSmoothing(train['Monthly_Sales'],trend = trend,
            damped_trend=damped_trend, seasonal=seasonal,
            initialization_method=init_method,use_boxcox = use_boxcox)
        res_hw = hw.fit()

        fcast = res_hw.forecast(h)
        try:
            error = mean_squared_error(test['Monthly_Sales'],fcast)
        except Exception as e:
            print('\n\n\n\n',e)
            print(test['Monthly_Sales'],fcast)
            print(steps_completed, steps_completed)
            error = float('inf')
            crash_counter += 1


        errors.append(error)

    if debug :
        print("last_seen : ", last_seen)
        print("steps_completed : ", steps_completed)

    return np.mean(errors)

walkForward('add','add',False,'legacy-heuristic',False,debug = True)
options = (trend_List,seasonal_List,damped_trend_list,init_method_list,use_boxcox_list)

best_score = float('inf')
best_option  = None

piyLi = []

for x in itertools.product(*options):
    score = walkForward(*x)
    piyLi.append([*x,score])

    if score<best_score:
        print("Best Score So Far : ", score,' |Option| ',x)
        best_score = score
        best_option = x



def sortAsc(val):
    return val[-1]

piyLi.sort(key = sortAsc)

print("best_score : " , best_score)
print("best_option : ", best_option)


print('\n\n\ncrash_counter : ', crash_counter,'\n\n')
pprint(piyLi)

def insertBestModel(trend,seasonal,damped_trend,init_method,use_boxcox,debug = False):
    train = df[:-12]
    test = df[-12:]
    train_idx = df.index <= train.index[-1]
    test_idx = df.index >train.index[-1]

    hw = ExponentialSmoothing(train['Monthly_Sales'],trend = trend,
        damped_trend=damped_trend, seasonal=seasonal,
        initialization_method=init_method,use_boxcox = use_boxcox)
    res_hw = hw.fit()

    df.loc[train_idx,'HW'] = res_hw.fittedvalues
    df.loc[test_idx, 'HW'] = res_hw.forecast(h)

insertBestModel(*best_option)
df.plot(figsize = (30,10))

plt.show()
