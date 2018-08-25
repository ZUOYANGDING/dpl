import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from util import getKaggleMNIST

def main():
	train_X, train_Y, test_X, test_Y = getKaggleMNIST()
	pca = PCA()
	train_Z = pca.fit_transform(train_X)
	plt.scatter(train_Z[:,0], train_Z[:,1], s=100, c=train_Y, alpha=0.5)
	plt.show()

	plt.plot(pca.explained_variance_ratio_)
	plt.show()

	cumulative_variance = []
	last_variance = 0
	for v in pca.explained_variance_ratio_:
		cumulative_variance.append(last_variance + v)
		last_variance = cumulative_variance[-1]
	plt.plot(cumulative_variance)
	plt.show()


if __name__ == '__main__':
	main()