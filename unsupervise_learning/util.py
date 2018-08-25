import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def getKaggleMNIST():
	data = pd.read_csv('train.csv').values.astype(np.float32)
	data = shuffle(data)
	train_X = data[:-1000, 1:] / 255
	train_Y = data[:-1000, 0].astype(np.int32)

	test_X = data[-1000:, 1:] / 255
	test_Y = data[-1000:, 0].astype(np.int32)
	return train_X, train_Y, test_X, test_Y

def error_rate(p,t):
	return np.mean(p != t)