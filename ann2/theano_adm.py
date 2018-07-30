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
	# learning_rate = 0.00004
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

	#1st moment
	mW1_init = np.zeros((K, M))
	mW2_init = np.zeros((M, K))
	mb1_init = np.zeros((1,M))
	mb2_init = np.zeros((1,K))

	#2nd moment
	vW1_init = np.zeros((M,M))
	vW2_init = np.zeros((M,M))
	vb1_init = np.zeros((M,M))
	vb2_init = np.zeros((K,K))

	#hyperparams
	learning_rate = 0.001
	beta1 = 0.99
	beta2 = 0.999
	eps = 1e-8

	#other parameters
	t_init = 1


	#initialize theano variables
	thX = T.matrix('X')
	thT = T.matrix('T')
	W1 = theano.shared(W1_init, 'W1')
	W2 = theano.shared(W2_init, 'W2')
	b1 = theano.shared(b1_init, 'b1')
	b2 = theano.shared(b2_init, 'b2')
	mW1 = theano.shared(mW1_init, 'mW1')
	mW2 = theano.shared(mW2_init, 'mW2')
	mb1 = theano.shared(mb1_init, 'mb1')
	mb2 = theano.shared(mb2_init, 'mb2')
	vW1 = theano.shared(vW1_init, 'vW1')
	vW2 = theano.shared(vW2_init, 'vW2')
	vb1 = theano.shared(vb1_init, 'vb1')
	vb2 = theano.shared(vb2_init, 'vb2')
	t = theano.shared(t_init, 't')

	#action fuction
	tZ = relu(thX.dot(W1) + b1)
	t_pY = T.nnet.softmax(tZ.dot(W2) + b2)

	#cost and prediction function
	cost = -(thT * T.log(t_pY)).sum() + reg * ((W1*W1).sum() + (W2*W2).sum() + (b1*b1).sum() + (b2*b2).sum())
	prediction = T.argmax(t_pY, axis=1)

	#trainning
	#update gradient
	gW2 = T.grad(cost, W2)
	gb2 = T.grad(cost, b2)
	gW1 = T.grad(cost, W1)
	gb1 = T.grad(cost, b1)

	#update 1st moment
	update_mW1 = beta1*mW1 + (1-beta1)*gW1
	update_mW2 = beta1*mW2 + (1-beta1)*gW2
	update_mb1 = beta1*mb1 + (1-beta1)*gb1
	update_mb2 = beta1*mb2 + (1-beta1)*gb2

	#update 2nd moment
	update_vW1 = beta2*vW1 + (1-beta2)*gW1*gW1
	update_vW2 = beta2*vW2 + (1-beta2)*gW2*gW2
	update_vb1 = beta2*vb1 + (1-beta2)*gb1*gb1
	update_vb2 = beta2*vb2 + (1-beta2)*gb2*gb2

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

	#update
	update_t = t + 1
	update_W2 = W2 - learning_rate * mW2_hat / T.sqrt(vW2_hat + eps)
	update_b2 = b2 - learning_rate * mb2_hat / T.sqrt(vb2_hat + eps)
	update_b1 = b1 - learning_rate * mb1_hat / T.sqrt(vb1_hat + eps)
	update_W1 = W1 - learning_rate * mW1_hat / T.sqrt(vW1_hat + eps)

	train = theano.function(
		inputs=[thX, thT],
		updates=[(W1, update_W1), (W2, update_W2), (b1, update_b1), (b2, update_b2), 
		(mW1, update_mW1), (mW2, update_mW2), (mb1, update_mb1), (mb2, update_mb2), 
		(vW1, update_vW1), (vW2, update_vW2), (vb1, update_vb1), (vb2, update_vb2),
		(t, update_t)])

	get_prediciton = theano.function(
		inputs=[thX, thT],
		outputs=[cost, prediction])

	costs = []
	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

			train(x, y)
			if j % print_period == 0:
				cost, test_pY = get_prediciton(test_X, test_Y_ind)
				rror = error_rate(test_pY, test_Y)
				print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost, error))
				costs.append(cost)
	plt.plot(costs)
	plt.show()

if __name__ == '__main__':
	main()








