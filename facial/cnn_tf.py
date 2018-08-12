import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import convolve2d
from scipy.io import loadmat
from sklearn.utils import shuffle

from benchmark import get_data

def error_rate(X, T):
	return np.mean(X != T)

def convpool(X, W, b):
	conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
	conv_out = tf.nn.bias_add(conv_out, b)
	pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return tf.nn.relu(pool_out)

def init_filter(shape):
	w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
	return w.astype(np.float32)

def rearrange(X):
	# input is (32, 32, 3, N)
	# output is (N, 32, 32, 3)
	return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)

def main():
	train, test = get_data()
	train_X = rearrange(train['X'])
	train_Y = train['y'].flatten() - 1
	train_X, train_Y = shuffle(train_X, train_Y)
	test_X = rearrange(test['X'])
	test_Y = test['y'].flatten() - 1
	del train
	del test

	max_iter = 6
	print_period = 10
	N = train_X.shape[0]
	batch_sz = 500
	num_batch = N // batch_sz
	train_X = train_X[:73000,]
	train_Y = train_Y[:73000]
	test_X = test_X[:26000,]
	test_Y = test_Y[:26000]

	#init weights and placeholders
	M = 500
	K = 10
	W1_shape = (5, 5, 3, 20)
	W1_init = init_filter(W1_shape)
	b1_init = np.zeros(W1_shape[-1], dtype=np.float32)
	W2_shape = (5, 5, 20, 50)
	W2_init = init_filter(W2_shape)
	b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

	W3_init = np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt(W2_shape[-1]*8*8 + M)
	b3_init = np.zeros(M, dtype=np.float32)
	W4_init = np.random.randn(M, K) / np.sqrt(M+K)
	b4_init = np.zeros(K, dtype=np.float32)

	inputs = tf.placeholder(tf.float32, shape=[batch_sz, 32, 32, 3], name='inputs')
	labels = tf.placeholder(tf.int32, shape=[batch_sz,], name='labels')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))
	W4 = tf.Variable(W4_init.astype(np.float32))
	b4 = tf.Variable(b4_init.astype(np.float32))

	#forward
	Z1 = convpool(inputs, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z2_shape = Z2.get_shape().as_list()
	Z2_re = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
	Z3 = tf.nn.relu(tf.matmul(Z2_re, W3) + b3)
	logits = tf.matmul(Z3, W4) + b4

	#init functions
	cost = tf.reduce_sum(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits,
			labels=labels))
	train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)
	predict_op = tf.argmax(logits, axis=1)

	costs = []
	W1_value = None
	W2_value = None
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		for i in range(max_iter):
			shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz : (j*batch_sz+batch_sz),]
				y = shuffle_Y[j*batch_sz : (j*batch_sz+batch_sz),]

				if len(x)==batch_sz:
					session.run(train_op, feed_dict={inputs: x, labels: y})
					if j % print_period==0:
						test_cost = 0
						prediction = np.zeros(len(test_X))
						for k in range(len(test_X) // batch_sz):
							Xtestbatch = test_X[k*batch_sz:(k*batch_sz + batch_sz),]
							Ytestbatch = test_Y[k*batch_sz:(k*batch_sz + batch_sz),]
							test_cost += session.run(cost, feed_dict={inputs: Xtestbatch, labels: Ytestbatch})
							prediction[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(
								predict_op, feed_dict={inputs: Xtestbatch})
						err = error_rate(prediction, test_Y)
						costs.append(test_cost)
						print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
		W1_value = W1.eval()
		W2_value = W2.eval()
	plt.plot(costs)
	plt.show()

	W1_value = W1_value.transpose(3, 2, 0, 1)
	W2_value = W2_value.transpose(3, 2, 0, 1)

	#input 3 chanels, output 20 chanels, use 8*8=64 grids and left final 4 empty
	grid = np.zeros((8*5, 8*5))
	m = 0
	n = 0
	for i in range(20):
		for j in range(3):
			grid[m*5:(m+1)*5, n*5:(n+1)*5] = W1_value[i, j]
			m += 1
			if m >= 8:
				m = 0
				n += 1
	plt.imshow(grid, cmap='gray')
	plt.title('W1')
	plt.show()

	#input 20, output 50, total is 1000. use 32*32=1024 grids and left final 24 empty
	grid = np.zeros((32*5, 32*5))
	m = 0
	n = 0
	for i in range(50):
		for j in range(20):
			grid[m*5:(m+1)*5, n*5:(n+1)*5] = W2_value[i, j]
			m += 1
			if m >= 32:
				m = 0
				n += 1
	plt.imshow(grid, cmap='gray')
	plt.title('W2')
	plt.show()


if __name__ == '__main__':
	main()





