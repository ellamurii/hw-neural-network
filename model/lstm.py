from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
from os import path
import json
fileName = sys.argv[1]

import metrics

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
dataset = np.asarray(series.iloc[:, 1])
data = series.values

n_steps = 5

X, y = split_sequence(dataset, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(LSTM(50, activation=None, input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=50, verbose=0)

finalfive = np.asarray(series.iloc[-5:, 1])
x_input = finalfive.reshape((1, n_steps, n_features))

yhat_nextday = model.predict(x_input, verbose=0)
print('\nNext Day Prediction: ', round(yhat_nextday[0][0], 2))
yhat = model.predict(X, verbose=0)
yhat_arr = list(yhat.flatten())

metrics.evaluate(series.iloc[:-5, 1],yhat_arr)

plt.plot(yhat, label= 'LSTM Prediction')
plt.plot(dataset, label='Actual Data')
plt.legend(loc='best')
plt.show()
