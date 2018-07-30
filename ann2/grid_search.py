import theano.tensor as T
from theano_ann import ANN
from until import get_spiral
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def main():
	X, Y = get_spiral()
	N, D = X.shape
	num_train = int(0.7*N)
	train_X = X[:num_train, :]
	train_Y = Y[:num_train]
	test_X = X[num_train:, :]
	test_Y = Y[num_train:]

	#hyperparamters need to search
	hidden_layer_size = [[300], [100, 100], [50, 50, 50]]
	learning_rate = [1e-4, 1e-3, 1e-2]
	reg_penalties = [0., 0.1, 1.0]

	best_validation_rate = 0
	best_hidden_layer_size = None
	best_learning_rate = None
	best_reg_penalties = None

	for hlz in hidden_layer_size:
		for lr in learning_rate:
			for rp in reg_penalties:
				model = ANN(hlz)
				model.fit(train_X, train_Y, learning_rate=lr, reg=rp, mu=0.99, epochs=3000, show_fig=False)
				validation_rate = model.score(test_X, test_Y)
				train_accuracy = model.score(train_X, train_Y)
				print("validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" % (validation_rate, train_accuracy, hlz, lr, rp))

				if validation_rate >  best_validation_rate:
					best_validation_rate = validation_rate
					best_hidden_layer_size = hlz
					best_learning_rate = lr
					best_reg_penalties = rp
	print("Best validation_accuracy:", best_validation_rate)
	print("Best settings:")
	print("hidden_layer_sizes:", best_hidden_layer_size)
	print("learning_rate:", best_learning_rate)
	print("l2:", best_reg_penalties)

if __name__ == '__main__':
	main()