import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as holt_winters
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import sys
from os import path
import json
fileName = sys.argv[1]

import metrics

import time
from datetime import timedelta
start_time = time.time()

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):

		end_ix = i + n_steps

		if end_ix > len(sequence)-1:
			break

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "dataset", fileName))
series = pd.read_json(filepath)
rawData = ((series.iloc[:, 1]).values).reshape(-1,1)
# normalize dataset
scaler = MinMaxScaler(feature_range=(1, 2))
series.iloc[:, 1] = scaler.fit_transform(rawData)

dataset = series.iloc[:, 1]

model = holt_winters(
						dataset,
						seasonal='multiplicative',
						seasonal_periods=7
					)


hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
HW_Smoothen = pd.Series(hw_model.predict(start=dataset.index[0] , end = dataset.index[-1])).values

n_steps = 5

X, y = split_sequence(HW_Smoothen, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

lstm_units = 50

model = Sequential()
model.add(LSTM(
				lstm_units,
				activation="relu",
				input_shape=(n_steps, n_features)
			)
		)
model.add(Dense(1))

train_X, val_X, train_y, val_y = train_test_split(X, y)

model.compile(
				optimizer='adam',
				loss='mse',
			)


hist = model.fit(X, y, epochs=50, batch_size=64,
				validation_data=(val_X, val_y),  verbose=0)

finalfive = np.asarray(series.iloc[-5:, 1])
x_input = finalfive.reshape((1, n_steps, n_features))

yhat_nextday = model.predict(x_input, verbose=0)
yhat = model.predict(X, verbose=0)

# denormalize values
yhat_nextday = scaler.inverse_transform(yhat_nextday)
series.iloc[:-5, 1] = scaler.inverse_transform(((series.iloc[:-5, 1]).values).reshape(-1,1))
yhat = scaler.inverse_transform(yhat)

yhat_arr = list(yhat.flatten())

print('\nNext Day Prediction: ', round(yhat_nextday[0][0], 2))
metrics.evaluate(series.iloc[:-5, 1],yhat_arr)

plt.plot(yhat, label='Hybrid HW-LSTM Prediction')
plt.plot(series.iloc[:-5, 1], label='Actual Data')
plt.legend(loc='best')
plt.show()
