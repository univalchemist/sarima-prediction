import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from itertools import product
import warnings
from warnings import catch_warnings
from warnings import filterwarnings
import datetime
from ast import literal_eval

filterwarnings('ignore')

# n-step sarima forecast
def sarima_forecast(history, config, n_test):
	order, sorder = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make n step forecast
	# predict = model_fit.predict(start=len(history), end=len(history) + n_test - 1, dynamic=True)
	fcast = model_fit.forecast(steps=n_test)
    # forecast = model_fit.get_forecast(steps=n_test)
	return fcast

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
 
# forward validation for univariate data
def forward_validation(data, n_test, cfg):
    # split dataset
    train, test = train_test_split(data, n_test)
    # fit model and predict n step
    fcast = sarima_forecast(train, cfg, n_test)
    # estimate prediction error
    error = measure_rmse(test, fcast)
    return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
	# convert config to a key
    key = str(cfg)
	# show all warnings and fail on exception if debugging
    if debug:
        result = forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
			# never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = forward_validation(data, n_test, cfg)
        except:
            error = None
	# check for an interesting result
    if result is not None:
        print(' > Model[%s] %.5f' % (key, result))
    return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
	    # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
	    scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
    scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of sarima configs to try
def sarima_configs():
    models = list()
	# define config lists
    p_params = range(0, 3)
    d_params = range(0, 2)
    q_params = range(0, 3)
    P_params = range(0, 3)
    D_params = range(0, 2)
    Q_params = range(0, 3)
    m_params = [30, 40, 50, 60, 70]
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for P in P_params:
                    for D in D_params:
                        for Q in Q_params:
                            for m in m_params:
                                cfg = [(p,d,q), (P,D,Q,m)]
                                models.append(cfg)
    return models
def cal_duration(time1, time2):
    td = time2 - time1
    return td.total_seconds()

if __name__ == '__main__':
    # load dataset
    df = pd.read_csv('dataset/eurusd_h1_2016_1_2021_10.csv', usecols=[4])

    #Calculate the natural logarithm of the data so that the first difference 
    #gets the data percentage growth.
    df = np.log(df)
    # for i in range(2, 50):
    i = 1
    n_test = 24
    # all_data = df[i * n_test:( i * n_test) + 3000]
    all_data = df[:1000]
    data = all_data[:-n_test] # For test of prediction last predict_step steps

    # Check if the data is stationary or not by using adfuller function. If the result is greater
    # than 0.05(significance level)

    # order and seasonal order
    order = (1,1,1)
    s_order = (1,1,1,30)
    time1 = datetime.datetime.now()

    # Build Model to test the result with the data_test
    model_fit = SARIMAX(data, order=order, seasonal_order=s_order).fit()

    prediction = model_fit.forecast(steps=n_test)

    # data=np.exp(data)
    actual=np.exp(all_data[-n_test:len(all_data)])
    prediction= np.exp(prediction)
    time2 = datetime.datetime.now()
    print('------------duration-----------')
    print(cal_duration(time1, time2))

    #Plot prediction n steps ahead
    plt.plot(actual, label='actual')
    plt.plot(prediction, label='prediction')
    plt.title('Prediction vs Actual')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('{}test.png'.format(i))
    plt.close()
    # plt.show()