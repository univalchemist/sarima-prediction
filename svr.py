import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
dates = []
prices = []

# def get_data(filename):
#     with open(filename, 'r') as csvfile:
#         csvReader = csv.reader(csvfile)
#         print(csvReader)
#         length = 0
#         next(csvReader)
#         for row in csvReader:
#             print('------------------------', length)
#             dates.append(length)
#             prices.append(float(row[1]))
#             length = length + 1
#     return
def get_data(filename):
    dataframe = pd.read_csv(filename, usecols=[4,5], engine='python')
    # drop rows with volume 0
    dataframe = dataframe.drop(dataframe[dataframe.Volume == 0].index)
    # drop column with volume
    dataframe = dataframe.drop(dataframe.columns[[1]], axis=1)
    dataset = dataframe.values
    data = dataset[0:30]
    length = 0
    for row in data:
        dates.append(length)
        prices.append(float(row[0]))
        length = length + 1
    return
def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	# svr_lin = SVR(kernel= 'linear', C= 1e3, gamma = 'auto')
	# svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2, gamma = 'auto')
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	# svr_lin.fit(dates, prices)
	# svr_poly.fit(dates, prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	# plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	# plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	# return svr_rbf.predict(np.array(x).reshape(-1,1))[0], svr_lin.predict(np.array(x).reshape(-1,1))[0], svr_poly.predict(np.array(x).reshape(-1,1))[0]
	return svr_rbf.predict(np.array(x).reshape(-1,1))[0]

get_data('goog.csv') # calling get_data method by passing the csv file to it
print("Dates- ", dates)
print("Prices- ", prices)

predicted_price = predict_price(dates, prices, 29)  
print("\nThe stock open price for 29th Feb is:")
print("RBF kernel: $", str(predicted_price))