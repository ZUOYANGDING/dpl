import numpy as np
import pandas as pd

def init_weight_bias(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)

#function for cnn
# def init_filter(shape, poolsz):
#     w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
#     return w.astype(np.float32)


def relu(X):
	return X * (X>0)


def sigmoid(A):
	return 1 / (1 + np.exp(-A))

def tan_h(A):
	return np.tanh(A)


def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
	cost = - (T * np.log(Y) + (1-T) * np.log(1-Y)).sum()
	return cost


def cost(T, Y):
	return -(T * np.log(Y)).sum()


def cost2(T, Y):
	N = len(T)
	return -np.log(Y[np.arange(N), T]).mean()


def error_rate(targets, predictions):
	return np.mean(targets != predictions)


def indicator(Y):
	N = len(Y)
	K = len(set(Y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, Y[i]] = 1
	return ind


def getData(balance_ones=True):
	X = []
	Y = []
	first_line = True
	for line in open('fer2013.csv'):
		if first_line:
			first_line = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])
	X = np.array(X) / 255.0
	Y = np.array(Y)

	if balance_ones:
		X0 = X[Y!=1, :]
		Y0 = Y[Y!=1]
		X1 = X[Y==1, :]
		X1 = np.repeat(X1, 9, axis=0)
		X = np.vstack([X0, X1])
		Y = np.concatenate((Y0, [1]*len(X1)))

	return X, Y


#function to get image
# def getImageData():
#     X, Y = getData()
#     N, D = X.shape
#     d = int(np.sqrt(D))
#     X = X.reshape(N, 1, d, d)
#     return X, Y


def getBinaryData():
	X = []
	Y = []
	first_line = True
	for line in open('fer2013.csv'):
		if first_line:
			first_line = False
		else:
			row = line.split(',')
			y = int(row[0])
			if y == 0 or y == 1:
				Y.append(y)
				X.append([int(p) for p in row[1].split()])
	X = np.array(X) / 255.0
	Y = np.array(Y)

	return X, Y


def cross_validation(model, X, Y, K):
	X, Y = shuffle(X, Y)
	size = len(Y) // K
	errors = []

	for k in range(K):
		x_train = np.concatenate(X[:(k*size), :], X[(k*size+size):, :])
		y_train = np.concatenate(Y[:(k*size)], Y[(k*size+size):])
		x_test = X[(k*size) : (k*size+size), :]
		y_test = Y[(k*size) : (k*size+size)]

		model.fit(x_train, y_train)
		error = model.score(x_test, y_test)
		errors.append(error)
	
	print("errors:", errors)
	return np.mean(errors)












