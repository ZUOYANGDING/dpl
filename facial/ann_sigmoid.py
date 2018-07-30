import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate, tan_h

class ANN(object):
	def __init__(self, M):
		self.M = M

	def forward(self, X):
		Z = tan_h(X.dot(self.W1) + self.b1)
		pY = sigmoid(Z.dot(self.W2) + self.b2)
		return pY, Z

	def fit(self, X, Y, learning_rate= 5e-6, reg=1.0, epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		validX = X[-1000:, :]
		validY = Y[-1000:]
		trainX = X[:-1000, :]
		trainY = Y[:-1000]

		N, D = trainX.shape

		self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
		self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
		self.b1 = np.zeros(self.M)
		self.b2 = 0

		costs = []
		best_validation_error = 1

		for i in range(epochs):
			pY, Z = self.forward(trainX)
			
			self.W2 -= learning_rate * (Z.T.dot(pY-trainY) + reg*self.W2)
			self.b2 -= learning_rate * ((pY-trainY).sum() + reg*self.b2)
			
			dz = np.outer(pY-trainY, self.W2) * (1 - Z*Z)
			self.W1 -= learning_rate * (trainX.T.dot(dz) + reg*self.W1)
			self.b1 -= learning_rate * (np.sum(dz, axis=0) + reg*self.b1)

			if i % 20 == 0:
				pY_valid, Z_valid = self.forward(validX)
				c = sigmoid_cost(validY, pY_valid)
				costs.append(c)
				e = error_rate(validY, np.round(pY_valid))
				print("i: ", i, " cost: ", c, " error: ", e)
				if e <best_validation_error:
					best_validation_error = e
		print("best_validation_error: ", best_validation_error)

		if show_fig:
			plt.plot(costs)
			plt.show()


def main():
	X, Y = getBinaryData()

	X0 = X[Y==0, :]
	X1 = X[Y==1, :]
	X1 = np.repeat(X1, 9, axis=0)
	X = np.vstack([X0, X1])
	Y = np.concatenate(([0]*len(X0), [1]*len(X1)))

	model = ANN(100)
	model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
	main()


