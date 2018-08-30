import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import getKaggleMNIST, error_rate

def init_weight(shape):
	W = np.random.randn(*shape) / np.sqrt(sum(shape))
	return W.astype(np.float32)

class Layer(object):
	def __init__(self, m1, m2):
		W = init_weight((m1, m2))
		b0 = np.zeros(m1, dtype=np.float32)
		b1 = np.zeros(m2, dtype=np.float32)
		self.W = tf.Variable(W)
		self.b0 = tf.Variable(b0)
		self.b1 = tf.Variable(b1)

	def set_session(self, session):
		self.session = session

	def forward(self, X):
		return tf.nn.sigmoid(tf.matmul(X, self.W) + self.b1)

	def forward_T(self, X):
		return tf.nn.sigmoid(tf.matmul(X, tf.transpose(self.W)) + self.b0)

class DeepAutoEncoder(object):
	def __init__(self, D, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes
		m_in = D
		self.hidden_layers = []
		for m_out in self.hidden_layer_sizes:
			layer = Layer(m_in, m_out)
			self.hidden_layers.append(layer)
			m_in = m_out
		self.build(D)

	def build(self, D):
		self.X_in = tf.placeholder(tf.float32, shape=(None, D))
		self.X_hat = self.forward(self.X_in)

		self.cost = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				labels=self.X_in,
				logits=self.X_hat))
		self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost)

	def set_session(self, session):
		self.session = session
		for layer in self.hidden_layers:
			layer.set_session(session)

	def transform(self, X):
		return self.session.run(self.X_hat, feed_dict={self.X_in: X})

	def forward(self, X):
		current_input = X
		for layer in self.hidden_layers:
			current_input = layer.forward(current_input)

		# self.map2center = self.transform(X)

		for i in range(len(self.hidden_layers)-1, -1, -1):
			current_input = self.hidden_layers[i].forward_T(current_input)

		return current_input

	def fit(self, X, epochs=50, batch_sz=100, show_fig=True):
		# lr = np.float32(learning_rate)
		# mu = np.float32(mu)
		N, D = X.shape
		num_batch = N // batch_sz
		# m_in = D
		# self.hidden_layers = []
		# for m_out in self.hidden_layer_sizes:
		# 	layer = Layer(m_in, m_out)
		# 	self.hidden_layers.append(layer)
		# 	m_in = m_out

		# self.X_in = tf.placeholder(tf.float32, shape=(None, D))
		# self.X_hat = self.forward(self.X_in)

		# cost = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(
		# 		labels=self.X_in,
		# 		logits=self.X_hat))
		# train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

		costs = []
		for i in range(epochs):
			shuffle_X = shuffle(X)
			print("epochs:", i)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz:(j*batch_sz+batch_sz)]
				_, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: x})

				if j % 100 == 0:
					print("j / num_batch", j, "/", num_batch, "cost:", c)
				costs.append(c)

		if show_fig:
			plt.plot(costs)
			plt.show()


def main():
	train_X, train_Y, test_X, test_Y = getKaggleMNIST()
	train_X = train_X.astype(np.float32)
	N, D = train_X.shape
	model = DeepAutoEncoder(D, [500, 300, 2])
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		model.set_session(session)
		model.fit(train_X)
		mapping = model.transform(train_X)
		plt.scatter(mapping[:,0], mapping[:, 1], c=train_Y, s=100, alpha=0.5)
		plt.show()

if __name__ == '__main__':
	main()

