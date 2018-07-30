import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import getData, getBinaryData, error_rate, init_weight_bias, indicator
from sklearn.utils import shuffle

class Hiddenlayer:
	def __init__(self, M1, M2, an_id):
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_bias(M1, M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.param = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):
	def __init__(self, hidden_layer_size):
		self.hidden_layer_size = hidden_layer_size

	def forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X):
		return tf.argmax(self.forward(X), axis=1)

	def fit(self, X, Y, learning_rate = 1e-2, mu=0.99, decay=0.999, reg=1e-3, epochs=10, batch_size=100, show_fig=False):
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
		reg = np.float32(reg)

		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		K = len(set(Y))
		Y = indicator(Y).astype(np.float32)
		train_X = X[:-1000, :]
		train_Y = Y[:-1000, :]
		test_X = X[-1000:, :]
		test_Y = Y[-1000:, :]
		test_Y_flat = np.argmax(test_Y, axis=1)

		N, D = train_X.shape
		# K = len(set(Y))
		M1 = D
		self.hidden_layers = []
		count = 0
		for M2 in self.hidden_layer_size:
			h = Hiddenlayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1 = M2
			count += 1
		W, b = init_weight_bias(M1, K)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		#store parameters
		self.param = [self.W, self.b]
		for h in self.hidden_layers:
			self.param += h.param

		#set functions and variables
		tf_X = tf.placeholder(tf.float32, shape=(None, D), name='X')
		tf_Y = tf.placeholder(tf.float32, shape=(None, K), name='T')
		action = self.forward(tf_X)
		rcost = reg * sum([tf.nn.l2_loss(p) for p in self.param])
		cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=action, labels=tf_Y)) + rcost
		train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost_fun)
		predict_fun = self.predict(tf_X)

		num_batch = N // batch_size
		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			for i in range(epochs):
				shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
				for j in range(num_batch):
					x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
					y = shuffle_Y[j*batch_size : (j*batch_size+batch_size), :]

					session.run(train_op, feed_dict={tf_X: x, tf_Y: y})
					if j % 20 == 0:
						c = session.run(cost_fun, feed_dict={tf_X: test_X, tf_Y: test_Y})
						p = session.run(predict_fun, feed_dict={tf_X: test_X})
						error = error_rate(test_Y_flat, p)
						print("i:", i, "j:", j, "nb:", num_batch, "cost:", c, "error_rate:", error)
						costs.append(c)
		if show_fig:
			plt.plot(costs)
			plt.show()


def main():
	X, Y = getData()
	model = ANN([2000, 1000, 500])
	model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
	main()











