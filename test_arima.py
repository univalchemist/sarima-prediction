import numpy
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from math import ceil
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import log

# Read from csv file
all_data = pd.read_csv('eurusd_d1_2016_1_2021_10.csv', usecols=[4,5], engine='python', skipfooter=1)
# drop rows with volume 0
all_data = all_data.drop(all_data[all_data.Volume == 0].index)
# drop column with volume
all_data = all_data.drop(all_data.columns[[1]], axis=1)
data = all_data.values

# For testing, will have data of 300 steps
data = data[0:300]

# Check if the data is stationary or not by using adfuller function. If the result is greater
# than 0.05(significance level)
# it is not stationary

result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Todo
# Find correct differencing

# split data
data_train = data[:275, :]
data_test = data[275:, :]

history = [x for x in data_train]
model = ARIMA(history, order=(1,1,1))
model_fit = model.fit()

# Forecast
forcast = model_fit.forecast(steps=25)

history = [x for x in data_train]
predictions = list()
for t in range(len(data_test)):
	model = ARIMA(history, order=(5,1,0)) #(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = data_test[t]
	history.append(numpy.array([yhat]))
	# history.append(obs)

# calculate MAE & RMSE
# print('ARIMA RMSE: %.3f' % sqrt(mean_squared_error(data_test, predictions)))
# print('ARIMA MAE: %.3f' % mean_absolute_error(data_test, predictions))
# print('ARIMA rsquare: %.3f' % r2_score(data_test, predictions))

plt.plot(data_test, label='actual')
plt.plot(predictions, label='forecast')
plt.title("ARIMA(1,1,1) Output")
plt.xlabel("Date") 
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()
# plt.savefig('aaa.png')
