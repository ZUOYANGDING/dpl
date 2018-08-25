import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from util import getKaggleMNIST

def main():
	train_X, train_Y, _, _, = getKaggleMNIST()
	sample = 1000
	train_X = train_X[:sample]
	train_Y = train_Y[:sample]
	tsne = TSNE()
	Z = tsne.fit_transform(train_X)
	plt.scatter(Z[:,0], Z[:,1], s=100, c=train_Y, alpha=0.5)
	plt.show()

if __name__ == '__main__':
	main()