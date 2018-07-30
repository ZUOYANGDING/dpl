import numpy as np
import matplotlib.pyplot as plt
from util import getData, softmax, cost2, indicator, error_rate, tan_h
from sklearn.utils import shuffle

class ANN(object):

	def __init__(self, M):
		self.M = M

	def forward(self, X):
		Z = tan_h(X.dot(self.W1) + self.b1)
		pY = softmax(Z.dot(self.W2) + self.b2)
		return pY, Z


	def predict(self, X):
		pY, Z =self.forward(X)
		return np.argmax(pY, axis=1)

	def score(self, X, Y):
		prediciton = self.predict(X)
		return 1 - error_rate(Y, prediciton)


	def fit(self, X, Y, learning_rate=1e-6, reg=1e-6, epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		valid_X = X[-1000:]
		valid_Y = Y[-1000:]
		train_X = X[:-1000]
		train_Y = Y[:-1000]

		T_train = indicator(train_Y)
		T_valid = indicator(valid_Y)

		N, D = train_X.shape
		K = len(set(train_Y))

		self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
		self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
		self.b1 = np.zeros(self.M)
		self.b2 = np.zeros(K)

		costs = []
		best_error_rate = 1

		for i in range(epochs):
			pY, Z = self.forward(train_X)

			self.W2 -= learning_rate * (Z.T.dot(pY - T_train) + reg*self.W2)
			self.b2 -= learning_rate * ((pY - T_train).sum(axis=0) + reg*self.b2)

			dz = (pY - T_train).dot(self.W2.T) * (1 - Z*Z)
			self.W1 -= learning_rate * (train_X.T.dot(dz) + reg*self.W1)
			self.b1 -= learning_rate * (dz.sum(axis=0) + reg*self.b1)

			if i % 10 == 0:
				pY_valid, Z_valid = self.forward(valid_X)
				c = cost2(valid_Y, pY_valid)
				e = error_rate(valid_Y, np.argmax(pY_valid, axis=1))
				costs.append(c)
				print("i: ", i, " cost: ", c, " error: ", e)


				if e < best_error_rate:
					best_error_rate = e
		print("best_error_rate: ", e)


		if show_fig:
			plt.plot(costs)
			plt.show()


def main():
	X, Y = getData()
	model = ANN(200)
	model.fit(X, Y, reg = 0, show_fig=True)
	print(model.score(X, Y))

if __name__ == '__main__':
	main()
