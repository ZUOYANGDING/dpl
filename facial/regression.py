import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getData, softmax, cost, indicator, error_rate

class LogisticModel(object):

	def __init__(self):
		pass

	def forward(self, X):
		pY = softmax(X.dot(self.W) + self.b)
		return pY

	def prediction(self, X):
		pY = self.forward(X)
		return np.argmax(pY, axis=1)

	def score(self, X, Y):
		prediction = self.prediction(X)
		return 1 - error_rate(Y, prediction)


	def fit(self, X, Y, learning_rate=1e-7, reg=0., epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		trainX = X[:-1000]
		trainY = Y[:-1000]
		validX = X[-1000:]
		validY = Y[-1000:]
		Tvalid = indicator(validY)
		Ttrain = indicator(trainY)

		N, D = trainX.shape
		K = len(set(trainY))
		self.W = np.random.randn(D, K) / np.sqrt(D)
		self.b = np.zeros(K)

		costs = []
		best_error_rate = 1

		for i in range(epochs):
			pY = self.forward(trainX)

			self.W -= learning_rate * (trainX.T.dot(pY-Ttrain) + reg*self.W)
			self.b -= learning_rate * ((pY - Ttrain).sum(axis=0) + reg*self.b)

			if i % 20 == 0:
				pY_valid = self.forward(validX)
				c = cost(Tvalid, pY_valid)
				costs.append(c)
				e = error_rate(validY, np.argmax(pY_valid, axis=1))
				print("i: ", i, " cost: ", c, " error_rate: ", e)

				if e < best_error_rate:
					best_error_rate = e

		print("best_error_rate is: ", best_error_rate)

		if show_fig:
			plt.plot(costs)
			plt.show()


def main():
	X, Y = getData()
	model = LogisticModel()
	model.fit(X, Y)
	print(model.score(X, Y))

if __name__ == '__main__':
	main()

