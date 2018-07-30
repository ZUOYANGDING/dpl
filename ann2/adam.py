import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from until import get_normalized_data, error_rate, cost, indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1

def main():
	max_iter = 10
	print_period = 50

	train_X, test_X, train_Y, test_Y = get_normalized_data()
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

	#1st moment
	mW1 = 0
	mW2 = 0
	mb1 = 0
	mb2 = 0

	#2nd moment
	vW1 = 0
	vW2 = 0
	vb1 = 0
	vb2 = 0

	#hyperparams
	learning_rate = 0.001
	beta1 = 0.99
	beta2 = 0.999
	eps = 1e-8

	#adam
	lose_adam = []
	error_adam = []
	t = 1
	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

			pY, Z = forward(x, W1, W2, b1, b2)

			#update gradient
			gW2 = derivative_w2(Z, y, pY) + reg*W2
			gb2 = derivative_b2(y, pY) + reg*b2
			gW1 = derivative_w1(x, Z, y, pY, W2) + reg*W1
			gb1 = derivative_b1(Z, y, pY, W2) + reg*b1

			#update 1st moment
			mW1 = beta1*mW1 + (1-beta1)*gW1
			mW2 = beta1*mW2 + (1-beta1)*gW2
			mb1 = beta1*mb1 + (1-beta1)*gb1
			mb2 = beta1*mb2 + (1-beta1)*gb2

			#update 2nd moment
			vW1 = beta2*vW1 + (1-beta2)*gW1*gW1
			vW2 = beta2*vW2 + (1-beta2)*gW2*gW2
			vb1 = beta2*vb1 + (1-beta2)*gb1*gb1
			vb2 = beta2*vb2 + (1-beta2)*gb2*gb2

			#bias correction
			correction_1 = 1 - beta1**t
			correction_2 = 1 - beta2**t
			mW1_hat = mW1 / correction_1
			mW2_hat = mW2 / correction_1
			mb1_hat = mb1 / correction_1
			mb2_hat = mb2 / correction_1

			vW1_hat = vW1 / correction_2
			vW2_hat = vW2 / correction_2
			vb1_hat = vb1 / correction_2
			vb2_hat = vb2 / correction_2

			#update t
			t += 1

			#update weight
			W2 -= learning_rate * mW2_hat / np.sqrt(vW2_hat + eps)
			b2 -= learning_rate * mb2_hat / np.sqrt(vb2_hat + eps)
			b1 -= learning_rate * mb1_hat / np.sqrt(vb1_hat + eps)
			W1 -= learning_rate * mW1_hat / np.sqrt(vW1_hat + eps)

			if j % print_period == 0:
				p_test, Z_test = forward(test_X, W1, W2, b1, b2)
				l = cost(p_test, test_Y_ind)
				e = error_rate(p_test, test_Y)
				lose_adam.append(l)
				error_adam.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))


	#RMSprop with momentum
	W1 = W1_copy.copy()
	b1 = b1_copy.copy()
	W2 = W2_copy.copy()
	b2 = b2_copy.copy()

	#hyperparams
	learning_rate = 0.001
	decay_rate = 0.999
	mu = 0.9
	eps = 1e-8

	#rmsprop cache
	cache_W1 = 1
	cache_W2 = 1
	cache_b1 = 1
	cache_b2 = 1

	#momentum
	dW1 = 0
	dW2 = 0
	db1 = 0
	db2 = 0

	lose_rmsprop_m = []
	error_rmsprop_m = []
	t = 1
	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

			pY, Z = forward(x, W1, W2, b1, b2)

			#udpate
			gW2 = derivative_w2(Z, y, pY) + reg*W2
			cache_W2 = decay_rate * cache_W2 + (1-decay_rate)*gW2*gW2
			dW2 = mu*dW2 - (1-mu) * learning_rate * gW2 / np.sqrt(cache_W2 + eps)
			W2 += dW2

			gb2 = derivative_b2(y, pY) + reg*b2
			cache_b2 = decay_rate * cache_b2 + (1-decay_rate)*gb2*gb2
			db2 = mu*db2 - (1-mu) * learning_rate * gb2 / np.sqrt(cache_b2 + eps)
			b2 += db2

			gW1 = derivative_w1(x, Z, y, pY, W2) + reg*W1
			cache_W1 = decay_rate * cache_W1 + (1-decay_rate)*gW1*gW1
			dW1 = mu*dW1 - (1-mu) * learning_rate * gW1 / np.sqrt(cache_W1 + eps)
			W1 += dW1

			gb1 = derivative_b1(Z, y, pY, W2) + reg*b1
			cache_b1 = decay_rate * cache_b1 + (1-decay_rate)*gb1*gb1
			db1 = mu*db1 - (1-mu) * learning_rate * gb1 / np.sqrt(cache_b1 + eps)
			b1 += db1
			# #update cache
			# cache_W1 = decay_rate * cache_W1 + (1-decay_rate)*gW1*gW1
			# cache_W2 = decay_rate * cache_W2 + (1-decay_rate)*gW2*gW2
			# cache_b1 = decay_rate * cache_b1 + (1-decay_rate)*gb1*gb1
			# cache_b2 = decay_rate * cache_b2 + (1-decay_rate)*gb2*gb2

			# #update momentum
			# dW2 = mu*dW2 + (1-mu) * learning_rate * gW2 / (np.sqrt(cache_W2) + eps)
			# db2 = mu*db2 + (1-mu) * learning_rate * gb2 / (np.sqrt(cache_b2) + eps)
			# dW1 = mu*dW1 + (1-mu) * learning_rate * dW1 / (np.sqrt(cache_W1) + eps)
			# db1 = mu*db1 + (1-mu) * learning_rate * db1 / (np.sqrt(cache_b1) + eps)

			# #update weights
			# W2 -= dW2
			# b2 -= db2
			# W1 -= dW1
			# b1 -= db1

			if j % print_period == 0:
				p_test, Z_test = forward(test_X, W1, W2, b1, b2)
				l = cost(p_test, test_Y_ind)
				e = error_rate(p_test, test_Y)
				lose_rmsprop_m.append(l)
				error_rmsprop_m.append(e)
				print("cost at itertion i=%d, j=%d: %.6f" % (i, j, l))
				print("error_rate: ", e)
	p_final, z_final = forward(test_X, W1, W2, b1, b2)
	print("final error_rate:", error_rate(p_final, test_Y))

	plt.plot(lose_adam, label="adam")
	plt.plot(lose_rmsprop_m, label="rmsprop with momentum")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()