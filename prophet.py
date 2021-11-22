import numpy
from pandas.core.frame import DataFrame
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
from fbprophet import Prophet
from datetime import timedelta, datetime
# Read from csv file
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
all_data = pd.read_csv('eurusd_d1_2011_1_2021_10.csv',
            usecols=[0,4],
            engine='python',
            )
print(all_data.head())
# all_data["Gmt time"] = pd.to_datetime(all_data["Gmt time"]).dt.strftime('%Y-%m-%d')
all_data['Gmt time'] = pd.to_datetime(all_data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f').dt.date
all_data = all_data.rename(columns = {"Gmt time":"ds","Close":"y"})
print(all_data.tail())

period_to_predict = 30
data = all_data[:-300]
print('-----------------data.tail()-----------------')
print(data.tail())

print('-----------------last_date-----------------')
last_date = data['ds'][-1:].values[0]
print(last_date)

# Trainding model
# m = Prophet(daily_seasonality = True)
m = Prophet()
m.fit(data)

# Predict next 30 days
future = m.make_future_dataframe(periods=30)
prediction = m.predict(future)

m.plot(prediction)
plt.title("Prediction using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Predict next period_to_predict days manually
# Define the period for which we want a prediction

# future = list()
# for i in range(1, period_to_predict + 1):
#     last_date = last_date + timedelta(1)
#     future.append(last_date)
# future = DataFrame(future)
# future.columns = ['ds']
# print('-----------------future dates----------------')
# print(future)
# # future['ds'] = pd.to_datetime(future['ds']) # if data is string, convert to datetime
# # use the model to make a forecast
# forecast = m.predict(future)
# # calculate MAE between expected and predicted values for next period_to_predict days
# y_true = all_data['y'][-period_to_predict:].values
# y_pred = forecast['yhat'].values
# mae = mean_absolute_error(y_true, y_pred)
# print('MAE: %.3f' % mae)
# # plot expected vs actual
# plt.plot(y_true, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.legend()
# plt.show()
