import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime
from benchmark import get_data

def relu(a):
	return a * (a>0)

def convpool(X, W, b, pool_size=(2, 2)):
	conv_out = conv2d(input=X, filters=W)
	pool_out = pool.pool_2d(
		input=conv_out,
		ws=pool_size,
		ignore_border=True,
		mode='max')
	# return relu(pool_out + b.dimshuffle('x', 0, 'x', 'x'))
	return T.tanh(pool_out + b.dimshuffle('x', 0, 'x', 'x'))

def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:]) / np.prod(poolsz))
	# w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[1:]))
	return w.astype(np.float32)

def rearrange(X):
	# input is (32, 32, 3, N)
    # output is (N, 3, 32, 32)
	return (X.transpose(3, 2, 0, 1) / 255).astype(np.float32)

def error_rate(p, t):
	return np.mean(p != t)

def main():
	train, test = get_data()
	train_X = rearrange(train['X'])
	train_Y = train['y'].flatten()-1
	train_X, train_Y = shuffle(train_X, train_Y)
	test_X = rearrange(test['X'])
	test_Y = test['y'].flatten()-1

	max_iter = 6
	print_period = 10
	lr = np.float32(0.0001)
	mu = np.float32(0.99)
	decay = np.float32(0.9)
	eps = np.float32(1e-10)
	reg = np.float32(0.01)
	N = train_X.shape[0]
	batch_sz = 500
	num_batch = N // batch_sz
	M = 500
	K = 10
	poolsz = (2, 2)

	W1_shape = (20, 3, 5, 5) #(num_feature_maps, num_color_channels, filter_width, filter_height)
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros(W1_shape[0], dtype=np.float32)

	W2_shape = (50, 20, 5, 5) #(num_feature_maps, old_num_feature_maps, filter_width, filter_height)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[0], dtype=np.float32)

	#ANN
	W3_init = np.random.randn(W2_shape[0]*5*5, M) / np.sqrt(W2_shape[0]*5*5 + M)
	b3_init = np.zeros(M, dtype=np.float32)
	W4_init = np.random.randn(M, K) / np.sqrt(M+K)
	b4_init = np.zeros(K, dtype=np.float32)

	#init theano variables
	X = T.tensor4('X', dtype='float32')
	Y = T.ivector('T')
	W1 = theano.shared(W1_init, 'W1')
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W2_init, 'W2')
	b2 = theano.shared(b2_init, 'b2')
	W3 = theano.shared(W3_init.astype(np.float32), 'W3')
	b3 = theano.shared(b3_init, 'b3')
	W4 = theano.shared(W4_init.astype(np.float32), 'W4')
	b4 = theano.shared(b4_init, 'b4')

	#forward
	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)
	pY = T.nnet.softmax(Z3.dot(W4) + b4)
	
	#test & prediction functions
	params = [W1, b1, W2, b2, W3, b3, W4, b4]
	rcost = reg * np.sum((p*p).sum() for p in params)
	cost = -(T.log(pY[T.arange(Y.shape[0]), Y])).mean() + rcost
	prediction = T.argmax(pY, axis=1)
	momentum = [theano.shared(
		np.zeros_like(p.get_value(), dtype=np.float32)) for p in params]
	catchs = [theano.shared(
		np.ones_like(p.get_value(), dtype=np.float32)) for p in params]
	
	#RMSProp
	updates = []
	grads = T.grad(cost, params)
	for p, g, m, c in zip(params, grads, momentum, catchs):
		updates_c = decay*c + (np.float32(1.0)-decay)*g*g
		updates_m = mu*m - lr*g / T.sqrt(updates_c + eps)
		updates_p = p + updates_m

		updates.append([c, updates_c])
		updates.append([m, updates_m])
		updates.append([p, updates_p])

	#init functions
	train_op = theano.function(inputs=[X, Y], updates=updates)
	prediction_op = theano.function(inputs=[X, Y], outputs=[cost, prediction])

	costs= []
	for i in range(max_iter):
		shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
		for j in range(num_batch):
			x = shuffle_X[j*batch_sz : (j*batch_sz+batch_sz), :]
			y = shuffle_Y[j*batch_sz : (j*batch_sz+batch_sz)]

			train_op(x, y)
			if j % print_period == 0:
				cost_val, p_val = prediction_op(test_X, test_Y)
				e = error_rate(p_val, test_Y)
				costs.append(cost_val)
				print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, e))
	plt.plot(costs)
	plt.show()


if __name__ == '__main__':
	main()









