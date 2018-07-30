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

	#initial hyperparameters
	M = 20
	num_hidden_layers = 2
	log_lr = -4
	log_reg = -2
	max_iter = 30

	best_M = M
	best_num_hidden_layers = num_hidden_layers
	best_validation_rate = 0
	best_hidden_layer_size = None
	best_learning_rate = None
	best_reg_penalty = None

	for i in range(max_iter):
		model = ANN([M] * num_hidden_layers)
		model.fit(train_X, train_Y, learning_rate=10**log_lr, reg=10**log_reg, mu=0.99, epochs=3000, show_fig=False)
		validation_rate = model.score(test_X, test_Y)
		train_accuracy = model.score(train_X, train_Y)
		print("validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" % (validation_rate, train_accuracy, [M]*num_hidden_layers, log_lr, log_reg))

		if validation_rate > best_validation_rate:
			best_M = M
			best_num_hidden_layers = num_hidden_layers
			best_validation_rate = validation_rate
			best_hidden_layer_size = [M] * num_hidden_layers
			best_learning_rate = log_lr
			best_reg_penalty = log_reg

		#update
		num_hidden_layers = best_num_hidden_layers + np.random.randint(-1, 2)
		M = best_M + np.random.randint(-1, 2)
		num_hidden_layers = max(10, num_hidden_layers)
		M = max(1, M)

		log_lr = best_learning_rate + np.random.randint(-1, 2)
		log_reg = best_reg_penalty + np.random.randint(-1, 2)
		log_lr = min(0, log_lr)
		log_reg = min(0, log_lr)

	print("Best validation_accuracy:", best_validation_rate)
	print("Best settings:")
	print("hidden_layer_sizes:", best_M*best_num_hidden_layers)
	print("learning_rate:", 10**best_learning_rate)
	print("l2:", 10**best_reg_penalty)

if __name__ == '__main__':
	main()