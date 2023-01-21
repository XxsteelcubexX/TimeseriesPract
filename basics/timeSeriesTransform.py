import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox

try:
    path = '../Datasets/'
    fileName = 'airline_passengers.csv'
    fileLoc = path + fileName
    df = pd.read_csv(fileLoc,index_col='Month',parse_dates = True)
except Exception as e:
    print(e)
    path = '/home/piyush/Projects/Timeseries/Datasets/'
    fileName = 'airline_passengers.csv'
    fileLoc = path + fileName
    df = pd.read_csv(fileLoc,index_col='Month',parse_dates = True)

print(df.head())
df['Passengers'].plot(figsize = (20,8))
#plt.show()

# SquareRoot Transform

df['SQRTPassengers'] = np.sqrt(df['Passengers'])
df['SQRTPassengers'].plot(figsize = (20,8))
#plt.show()

#log Transform

df['LogPassengers'] = np.log(df['Passengers'])
df['LogPassengers'].plot(figsize = (20,8))
plt.show()

## boxcox Transform

data, lam = boxcox(df['Passengers']) #here we are using Scipy library
print(lam)
df['BoxcoxPassengers'] = data
df['BoxcoxPassengers'].plot(figsize = (20,8))
plt.show()

df['Passengers'].hist(bins=20)
df['SQRTPassengers'].hist(bins=20)
df['LogPassengers'].hist(bins=20)
df['BoxcoxPassengers'].hist(bins=20)
plt.show()
