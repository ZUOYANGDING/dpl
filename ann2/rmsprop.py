import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from until import get_normalized_data, error_rate, cost, indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1

def main():
	max_iter = 20
	print_period = 50

	train_X, test_X, train_Y, test_Y = get_normalized_data()
	learning_rate = 0.00004
	reg = 0.01
	train_Y_ind = indicator(train_Y)
	test_Y_ind = indicator(test_Y)

	N, D = train_X.shape
	batch_size = 500
	batch_num = N // batch_size

	M = 300
	K = 10
	W1 = np.random.randn(D, M) / np.sqrt(D)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.zeros(K)

	#SAVE INITIAL WEIGHT AND BIAS
	W1_copy = W1.copy()
	b1_copy = b1.copy()
	W2_copy = W2.copy()
	b2_copy = b2.copy()

	#constant learning_rate
	lose_constant = []
	error_constant = []
	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

			pY, Z = forward(x, W1, W2, b1, b2)

			W2 -= learning_rate * (derivative_w2(Z, y, pY) + reg*W2)
			b2 -= learning_rate * (derivative_b2(y, pY) + reg*b2)
			W1 -= learning_rate * (derivative_w1(x, Z, y, pY, W2) + reg*W1)
			b1 -= learning_rate * (derivative_b1(Z, y, pY, W2) + reg*b1)

			if j % print_period == 0:
				p_test, Z_test = forward(test_X, W1, W2, b1, b2)
				l = cost(p_test, test_Y_ind)
				e = error_rate(p_test, test_Y)
				lose_constant.append(l)
				error_constant.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))


	#RMSprop
	W1 = W1_copy.copy()
	b1 = b1_copy.copy()
	W2 = W2_copy.copy()
	b2 = b2_copy.copy()

	learning_rate_0 = 0.001
	lose_non_costant = []
	error_non_constant = []
	cache_W1 = 1
	cache_W2 = 1
	cache_b1 = 1
	cache_b2 = 1
	decay_rate = 0.999
	eps = 1e-10

	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

			pY, Z = forward(x, W1, W2, b1, b2)
			gW2 = derivative_w2(Z, y, pY) + reg*W2
			cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
			W2 -= learning_rate_0 * gW2 / (np.sqrt(cache_W2) + eps)

			gb2 = derivative_b2(y, pY) + reg*b2
			cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
			b2 -= learning_rate_0 * gb2 / (np.sqrt(cache_b2) + eps)

			gW1 = derivative_w1(x, Z, y, pY, W2) + reg*W1
			cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
			W1 -= learning_rate_0 * gW1 / (np.sqrt(cache_W1) + eps)

			gb1 = derivative_b1(Z, y, pY, W2) + reg*b1
			cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
			b1 -= learning_rate_0 * gb1 / (np.sqrt(cache_b1) + eps)

			if j % print_period == 0:
				p_test, Z_test = forward(test_X, W1, W2, b1, b2)
				l = cost(p_test, test_Y_ind)
				e = error_rate(p_test, test_Y)
				lose_non_costant.append(l)
				error_non_constant.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))

	plt.plot(lose_constant, label="batch")
	plt.plot(lose_non_costant, label="non_constant")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()


