import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from until import get_normalized_data

def init_weight(M1, M2):
	return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)

def init_weight_bias(M1, M2):
	W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
	b = np.zeros(M2)
	return W, b

class HiddenlayerBacthNorm(object):
	def __init__(self, M1, M2, activation_f):
		self.M1 = M1
		self.M2 = M2
		self.activation_f = activation_f
		W = init_weight(M1, M2).astype(np.float32)
		gamma = np.ones(M2).astype(np.float32)
		beta = np.zeros(M2).astype(np.float32)
		running_mean = np.zeros(M2).astype(np.float32)
		running_var = np.zeros(M2).astype(np.float32)

		self.W = tf.Variable(W)
		self.gamma = tf.Variable(gamma)
		self.beta = tf.Variable(beta)
		self.running_mean = tf.Variable(running_mean)
		self.running_var = tf.Variable(running_var)

	def forward(self, X, is_training, decay=0.9):
		activation = tf.matmul(X, self.W)
		if is_training:
			batch_mean, batch_var = tf.nn.moments(activation, [0])
			update_running_mean = tf.assign(
				self.running_mean,
				self.running_mean*decay + batch_mean*(1-decay))
			update_running_var = tf.assign(
				self.running_var,
				self.running_var*decay + batch_var*(1-decay))
			with tf.control_dependencies([update_running_mean, update_running_var]):
				out = tf.nn.batch_normalization(
					activation,
					batch_mean,
					batch_var,
					self.beta,
					self.gamma,
					1e-4)
		else:
			out = tf.nn.batch_normalization(
				activation,
				self.running_mean,
				self.running_var,
				self.beta,
				self.gamma,
				1e-4)
		return self.activation_f(out)

class Hiddenlayer(object):
	def __init__(self, M1, M2, activation_f):
		self.M1 = M1
		self.M2 = M2
		self.activation_f = activation_f
		W, b = init_weight_bias(M1, M2)
		
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

	def forward(self, X):
		out = tf.matmul(X, self.W) + self.b
		return self.activation_f(out)

class ANN(object):
	def __init__(self, hidden_layer_size):
		self.hidden_layer_size = hidden_layer_size

	def set_session(self, session):
		self.session = session

	def forward(self, X, is_training):
		Z = X
		for h in self.hidden_layers[:-1]:
			Z = h.forward(Z, is_training)
		Z = self.hidden_layers[-1].forward(Z)
		return Z

	def predict(self, X):
		return self.session.run(self.predict_op, feed_dict={self.inputs: X})
	
	def score(self, X, Y):
		pY = self.predict(X)
		return np.mean(pY == Y)

	def fit (self, train_X, train_Y, test_X, test_Y, activation=tf.nn.relu, learning_rate=1e-2, epochs=15, batch_sz=100, print_period=100, show_fig=True):
		train_X = train_X.astype(np.float32)
		train_Y = train_Y.astype(np.int32)

		N, D = train_X.shape
		K = len(set(train_Y))
		self.hidden_layers = []
		M1 = D
		for M2 in self.hidden_layer_size:
			h = HiddenlayerBacthNorm(M1, M2, activation)
			self.hidden_layers.append(h)
			M1 = M2
		h = Hiddenlayer(M1, K, lambda x: x)
		self.hidden_layers.append(h)

		if batch_sz is None:
			batch_sz = N

		inputs = tf.placeholder(tf.float32, shape=(None, D), name='X')
		labels = tf.placeholder(tf.int32, shape=(None,), name='Y')
		self.inputs = inputs

		logits = self.forward(inputs, is_training=True)
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits = logits,
			labels = labels))
		train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(cost)

		test_logits= self.forward(inputs, is_training=False)
		self.predict_op = tf.argmax(test_logits, axis=1)

		self.session.run(tf.global_variables_initializer())

		num_batch = N // batch_sz
		costs = []
		for i in range(epochs):
			shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz : (j*batch_sz+batch_sz), :]
				y = shuffle_Y[j*batch_sz: (j*batch_sz+batch_sz)]

				c, _, logits_out = self.session.run([cost, train_op, logits], feed_dict={inputs: x, labels: y})
				costs.append(c)
				if (j+1) % print_period == 0:
					acc = np.mean(y == np.argmax(logits_out, axis=1))
					print("epoch:", i, "batch:", j, "n_batches:", num_batch, "cost:", c, "acc: %.2f" % acc)
			print("Train acc:", self.score(shuffle_X, shuffle_Y), "Test acc:", self.score(test_X, test_Y))

		if show_fig:
			plt.plot(costs)
			plt.show()

def main():
	train_X, test_X, train_Y, test_Y = get_normalized_data()
	model = ANN([500, 300])
	session = tf.InteractiveSession()
	model.set_session(session)
	model.fit(train_X, train_Y, test_X, test_Y, show_fig=True)
	print("Train accuracy:", model.score(train_X, train_Y))
	print("Test accuracy:", model.score(test_X, test_Y))

if __name__ == '__main__':
	main()


