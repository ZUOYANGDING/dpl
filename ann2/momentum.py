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

	#batch
	loss_batch = []
	error_batch =[]
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
				loss_batch.append(l)
				error_batch.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))



	#momentum
	W1 = W1_copy.copy()
	b1 = b1_copy.copy()
	W2 = W2_copy.copy()
	b2 = b2_copy.copy()

	lose_momentum = []
	error_momentum = []
	mu = 0.9
	dW1 = 0
	dW2 = 0
	db1 = 0
	db2 = 0

	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range (batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]
			pY, Z = forward(x, W1, W2, b1, b2)
			# print("overflow?")
			gW2 = derivative_w2(Z, y, pY) + reg*W2
			gb2 = derivative_b2(y, pY) + reg*b2
			gW1 = derivative_w1(x, Z, y, pY, W2) + reg*W1
			gb1 = derivative_b1(Z, y, pY, W2) + reg*b1

			#UDPATE VELOCITIES
			dW2 = mu*dW2 - learning_rate*gW2
			db2 = mu*db2 - learning_rate*gb2
			dW1 = mu*dW1 - learning_rate*gW1
			db1 = mu*db1 - learning_rate*gb1

			#UPDATE WEIGHT
			W2 += dW2
			b2 += db2
			W1 += dW1
			b1 += db1

			if j % print_period == 0:
				p_test, Z_test = forward(test_X, W1, W2, b1, b2)
				l = cost(p_test, test_Y_ind)
				e = error_rate(p_test, test_Y)
				lose_momentum.append(l)
				error_momentum.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))


	#Nesterov momentum
	W1 = W1_copy.copy()
	b1 = b1_copy.copy()
	W2 = W2_copy.copy()
	b2 = b2_copy.copy()

	lose_nesterov = []
	error_nesterov = []
	mu = 0.9
	dW1 = 0
	db1 = 0
	dW2 = 0
	db2 = 0

	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(test_X, test_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]
			pY, Z = forward(x, W1, W2, b1, b2)

			gW2 = derivative_w2(Z, y, pY) + reg*W2
			gb2 = derivative_b2(y, pY) + reg*b2
			gW1 = derivative_w1(x, Z, y, pY, W2) + reg*W1
			gb1 = derivative_b1(Z, y, pY, W2) + reg*b1

			#update velocities
			dW2 = mu*dW2 - learning_rate*gW2
			db2 = mu*db2 - learning_rate*db2
			dW1 = mu*dW1 - learning_rate*gW1
			db1 = mu*db1 - learning_rate*gb1

			#update weight
			W2 += mu*dW2 - learning_rate*gW2
			b2 += mu*db2 - learning_rate*db2
			W1 += mu*dW1 - learning_rate*gW1
			b1 += mu*db1 - learning_rate*gb1

			if j % print_period == 0:
				p_test, Z_test = forward(test_X, W1, W2, b1, b2)
				l = cost(p_test, test_Y_ind)
				e = error_rate(p_test, test_Y)
				lose_nesterov.append(l)
				error_nesterov.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))


	
	plt.plot(loss_batch, label="batch")
	plt.plot(lose_momentum, label="momentum")
	plt.plot(lose_nesterov, label="Nesterov")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
