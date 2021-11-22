# forecast monthly births with random forest
from numpy import asarray
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
	# split dataset
    train, test = train_test_split(data, n_test)
	# seed history with training dataset
    history = [x for x in train]
	# step over each time-step in the test set
    for i in range(len(test)):
		# split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
        predictions.append(yhat)
		# add actual observation to history for the next loop
        print('-------------------------')
        print(test[i])
        print(yhat)
        history.append(test[i])
		# history.append(yhat)
		# summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions

def forecast_multistep():
    # load the dataset
    df = read_csv('dataset/eurusd_d1_2011_1_2021_10.csv', usecols=[4])
    i = 1
    st = i * 12
    en = st + 2000
    series = df[st:en]
    values = series.values
    train_values = values[:-6]
    # transform the time series data into supervised learning
    data = series_to_supervised(train_values, n_in=6)
    trainX, trainY = data[:, :-1], data[:, -1]

    # fit model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainY)

    temp_input = train_values[-6:]
    print('----------temp_input before-----------')
    print(temp_input)
    predictions = list()
    for i in range(6):
        row = temp_input.flatten()
        yhat = model.predict(asarray([row]))
        predictions.append(yhat[0])
        temp_input = np.append(temp_input, [yhat[0]])
        temp_input = temp_input[1:]
    actual = values[-6:]
    print('----------actual-----------')
    print(actual)
    print('----------predictions-----------')
    print(predictions)
    plt.plot(actual, label='actual')
    plt.plot(predictions, label='prediction')
    plt.legend()
    plt.show()
    return
if __name__ == "__main__":
    forecast_multistep()
    # load the dataset
    # df = read_csv('dataset/eurusd_d1_2011_1_2021_10.csv', usecols=[4])
    # for i in range(1):
    #     st = i * 12
    #     en = st + 300
    #     series = df[st:en]
    #     values = series.values
    #     # transform the time series data into supervised learning
    #     data = series_to_supervised(values, n_in=6)
    #     print('--------------------------data------------------')
    #     print(data)
    #     print(data[0])
    #     print(data[0][0])
    #     # evaluate
    #     mae, y, yhat = walk_forward_validation(data, 12)
    #     print('MAE: %.3f' % mae)
    #     # plot expected vs predicted
    #     plt.plot(y, label='Expected')
    #     plt.plot(yhat, label='Predicted')
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig("RandomForest{}.png".format(i))
    #     plt.close()