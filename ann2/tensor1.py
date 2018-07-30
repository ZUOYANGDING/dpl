import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from until import get_normalized_data, indicator

def error_rate(target, prediction):
	return np.mean(target != prediction)

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

	M1 = 300
	M2 = 100
	K = 10
	W1_init = np.random.randn(D, M1) / np.sqrt(D)
	b1_init = np.zeros(M1)
	W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
	b2_init = np.zeros(M2)
	W3_init = np.random.randn(M2, K) / np.sqrt(M2)
	b3_init = np.zeros(K)

	#initialize tensorflow variables
	X = tf.placeholder(tf.float32, shape=(None, D), name='X')
	T = tf.placeholder(tf.float32, shape=(None, K), name='T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))

	#define model
	Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
	Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
	Y_temp = tf.matmul(Z2, W3) + b3

	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_temp, labels=T))
	train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)

	prediction_op = tf.argmax(Y_temp, axis=1)

	costs = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(max_iter):
			shuffle_X, shuffle_Y = shuffle(train_X, train_Y_ind)
			for j in range(batch_num):
				x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
				y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

				session.run(train_op, feed_dict={X: x, T: y})

				if j % print_period == 0:
					test_cost = session.run(cost, feed_dict={X: test_X, T: test_Y_ind})
					prediction = session.run(prediction_op, feed_dict={X: test_X})
					error = error_rate(prediction, test_Y)
					print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, error))
					costs.append(test_cost)


	plt.plot(costs)
	plt.show()


if __name__ == '__main__':
	main()



