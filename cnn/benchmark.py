import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

def get_data():
	train = loadmat('train_32x32.mat')
	test = loadmat('test_32x32.mat')
	return train, test

def flatten(X):
	N = X.shape[-1]
	flat = np.zeros((N, 3072))
	for i in range(N):
		flat[i] = X[:,:,:,i].reshape(3072)
	return flat

class Hiddenlayer(object):
	def __init__(self, M1, M2):
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
		b = np.zeros(M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.param = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)

	def forward_action(self, X):
		return tf.matmul(X, self.W) + self.b

class ANN(object):
	def __init__(self, hidden_layer_size):
		self.hidden_layer_size = hidden_layer_size

	def fit(self, train_X, train_Y, test_X, test_Y, learning_rate=1e-4, mu=0.9, decay=0.99, epochs=20, batch_size=500, show_fig=False):
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)

		N, D = train_X.shape
		K = len(set(train_Y))
		M1 = D
		self.hidden_layers = []
		for M2 in self.hidden_layer_size:
			h = Hiddenlayer(M1, M2)
			self.hidden_layers.append(h)
			M1 = M2
		h = Hiddenlayer(M1, K)
		self.hidden_layers.append(h)

		#init training
		inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
		labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
		train_logits = self.forward(inputs)
		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=train_logits,
				labels=labels))
		train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=mu).minimize(cost)

		#init predict
		test_logits = self.forward(inputs)
		prediction = self.predict(inputs)
		test_cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=test_logits,
				labels=labels))

		num_batch = N // batch_size
		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			for i in range(epochs):
				shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
				for j in range(num_batch):
					x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
					y = shuffle_Y[j*batch_size: (j*batch_size+batch_size)]

					session.run(train_op, feed_dict={inputs: x, labels: y})

					if j % 10 == 0:
						c = session.run(test_cost, feed_dict={inputs: test_X, labels: test_Y})
						p = session.run(prediction, feed_dict={inputs: test_X})
						e = self.error_rate(p, test_Y)
						costs.append(c)
						print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, c, e))
		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		Z = X
		for h in self.hidden_layers[:-1]:
			Z = h.forward(Z)
		Z = self.hidden_layers[-1].forward_action(Z)
		return Z

	def predict(self, X):
		pY = self.forward(X)
		return tf.argmax(pY, axis=1)

	def error_rate(self, P, T):
		return np.mean(P != T)






def main():
	train, test = get_data()
	train_X = flatten(train['X'].astype(np.float32) / 255.)
	train_Y = train['y'].flatten() - 1
	train_X, train_Y = shuffle(train_X, train_Y)
	test_X  = flatten(test['X'].astype(np.float32) / 255.)
	test_Y  = test['y'].flatten() - 1
	model = ANN([1000, 500])
	model.fit(train_X, train_Y, test_X, test_Y, show_fig=True)


if __name__ == '__main__':
	main()

