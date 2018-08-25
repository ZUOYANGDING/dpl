import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import error_rate, getKaggleMNIST

class AutoEncoder(object):
	def __init__(self, D, M, an_id):
		self.D = D
		self.M = M
		self.id = an_id
		self.build(D, M)

	def set_session(self, session):
		self.session = session

	def build(self, D, M):
		self.W = tf.Variable(tf.random_normal(shape=(D, M)))
		self.bh = tf.Variable(np.zeros(M).astype(np.float32))
		self.b0 = tf.Variable(np.zeros(D).astype(np.float32))

		self.X_in = tf.placeholder(tf.float32, shape=(None, D))
		self.Z = self.forward_hidden(self.X_in)
		self.X_hat = self.forward_output(self.X_in)

		logits = self.forward_logits(self.X_in)
		self.cost = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				labels = self.X_in,
				logits = logits))
		self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost)

	def transform(self, X):
		return self.session.run(self.Z, feed_dict={self.X_in: X})
	
	def forward_hidden(self, X):
		Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.bh)
		return Z

	def forward_logits(self, X):
		Z = self.forward_hidden(X)
		Z = tf.matmul(Z, tf.transpose(self.W)) + self.b0
		return Z

	def forward_output(self, X):
		return tf.nn.sigmoid(self.forward_logits(X))


	# def prediction(self, X):
	# 	return self.session.run(self.X_hat, feed_dict={self.X_in: X})

	def fit(self, X, epochs=1, batch_sz=100, show_fig=False):
		N, D = X.shape
		num_batch = N // batch_sz
		costs = []
		print("pretraining : %s" % self.id)
		for i in range(epochs):
			print("epochs: ", epochs)
			shuffle_X = shuffle(X)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz:(j*batch_sz+batch_sz)]
				_, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: x})
				# c = self.session.run(self.cost, feed_dict={self.X_in: x})
				if j % 10 == 0:
					print("j / n_batches:", j, "/", num_batch, "cost:", c)
				costs.append(c)

		if show_fig:
			plt.plot(costs)
			plt.show()


class DNN(object):
	def __init__(self, D, hidden_layer_sizes, K, UnsupervisedModel=AutoEncoder):
		self.hidden_layers = []
		count = 0
		input_size = D
		for output_size in hidden_layer_sizes:
			ae = UnsupervisedModel(input_size, output_size, count)
			self.hidden_layers.append(ae)
			count += 1
			input_size = output_size
		self.build_final_layer(D, hidden_layer_sizes[-1], K)

	def set_session(self, session):
		self.session = session
		for layer in self.hidden_layers:
			layer.set_session(session)

	#init logistic layer
	def build_final_layer(self, D, M, K):
		self.W = tf.Variable(tf.random_normal(shape=(M, K)))
		self.b = tf.Variable(np.zeros(K).astype(np.float32))
		self.X = tf.placeholder(tf.float32, shape=(None, D))
		labels = tf.placeholder(tf.int32, shape=(None,))
		self.Y = labels
		logits = self.forward(self.X)
		self.cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits = logits,
				labels = labels))
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)
		self.prediction = tf.argmax(logits, axis=1)

	def forward(self, X):
		current_input = X
		for ae in self.hidden_layers:
			Z = ae.forward_hidden(current_input)
			current_input = Z
		return (tf.matmul(current_input, self.W) + self.b)

	def fit(self, train_X, train_Y, test_X, test_Y, pretrain=True, epochs=1, batch_sz=100):
		N = len(train_X)
		pretrain_epochs = 1
		if not pretrain:
			pretrain_epochs = 0

		current_input = train_X
		for ae in self.hidden_layers:
			ae.fit(current_input, epochs=pretrain_epochs)
			current_input = ae.transform(current_input)

		num_batch = N // batch_sz
		costs = []
		print("supervise training....")
		for i in range(epochs):
			print("epochs: ", i)
			shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz:(j*batch_sz+batch_sz)]
				y = shuffle_Y[j*batch_sz:(j*batch_sz+batch_sz)]

				self.session.run(self.train_op, feed_dict={self.X: x, self.Y: y})
				c, p = self.session.run((self.cost, self.prediction), feed_dict={self.X: test_X, self.Y: test_Y})
				# p = self.session.run(self.prediction, feed_dict={self.X: test_X, labels: test_Y})
				e = error_rate(p, test_Y)

				if j % 10 == 0:
					print("j / n_batches:", j, "/", num_batch, "cost:", c, "error:", e)
				costs.append(c)

		plt.plot(costs)
		plt.show()








def main():
	train_X, train_Y, test_X, test_Y = getKaggleMNIST()
	train_X = train_X.astype(np.float32)
	test_X = test_X.astype(np.float32)
	N, D = train_X.shape
	K = len(set(train_Y))
	model = DNN(D, [1000, 750, 500], K)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		model.set_session(session)
		model.fit(train_X, train_Y, test_X, test_Y, pretrain=True, epochs=10)

if __name__ == '__main__':
	main()
