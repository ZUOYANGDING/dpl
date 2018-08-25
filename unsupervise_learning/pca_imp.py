import numpy as np
import matplotlib.pyplot as plt
from util import getKaggleMNIST

def main():
	train_X, train_Y, test_X, test_Y = getKaggleMNIST()
	cov = np.cov(train_X.T)
	lamda, Q = np.linalg.eigh(cov)
	#sort lamda as descending order, where idx is index of the lamda
	idx = np.argsort(-lamda)
	lamda = lamda[idx]
	#set all lamdas as positive
	lamda = np.maximum(lamda, 0)
	Q = Q[:, idx]

	Z = train_X.dot(Q)
	plt.scatter(Z[:,0], Z[:,1], s=100, c=train_Y, alpha=0.3)
	plt.show()

	plt.plot(lamda)
	plt.title("Variance of each component")
	plt.show()

	plt.plot(np.cumsum(lamda))
	plt.title("Cumulative variance")
	plt.show()


if __name__ == '__main__':
	main()