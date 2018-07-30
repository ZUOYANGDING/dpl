import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from until import get_normalized_data, indicator

def relu(a):
	return a * (a>0)

def error_rate(prediction, target):
	return np.mean(prediction != target)

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
	W1_init = np.random.randn(D, M) / np.sqrt(D)
	b1_init = np.zeros(M)
	W2_init = np.random.randn(M, K) / np.sqrt(M)
	b2_init = np.zeros(K)

	#initialize theano variables
	thX = T.matrix('X')
	thT = T.matrix('T')
	W1 = theano.shared(W1_init, 'W1')
	W2 = theano.shared(W2_init, 'W2')
	b1 = theano.shared(b1_init, 'b1')
	b2 = theano.shared(b2_init, 'b2')

	#action function and softmax
	tZ = relu(thX.dot(W1) + b1)
	t_pY = T.nnet.softmax(tZ.dot(W2) + b2)

	#cost function and predition function
	cost = -(thT * T.log(t_pY)).sum() + reg*(W1*W1).sum() + reg*(W2*W2).sum() + reg*(b1*b1).sum() + reg*(b2*b2).sum()
	predition = T.argmax(t_pY, axis=1)

	#training
	update_b2 = b2 - learning_rate * T.grad(cost, b2)
	update_W2 = W2 - learning_rate * T.grad(cost, W2)
	update_b1 = b1 - learning_rate * T.grad(cost, b1)
	update_W1 = W1 - learning_rate * T.grad(cost, W1)

	train = theano.function(
		inputs=[thX, thT],
		updates=[(W1, update_W1), (W2, update_W2), (b1, update_b1), (b2, update_b2)])

	get_prediction = theano.function(
		inputs=[thX, thT],
		outputs=[cost, predition])

	costs = []
	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

			train(x, y)
			if j % print_period == 0:
				cost, test_pY = get_prediction(test_X, test_Y_ind)
				error = error_rate(test_pY, test_Y)
				print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost, error))
				costs.append(cost)
	plt.plot(costs)
	plt.show()


if __name__ == '__main__':
	main()