import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import init_weight, all_parity_pairs, indicator
from sklearn.utils import shuffle

class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.M1 = M1
		self.M2 = M2
		self.an_id = an_id
		W = init_weight(M1, M2)
		b = np.zeros(M2)
		self.W = tf.Variable(W, dtype=np.float32)
		self.b = tf.Variable(b, dtype=np.float32)
		self.param = [self.W, self.b]

	def set_session(self, session):
		self.session = session

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):
	def __init__(self, D, K, hidden_layer_sizes, learning_rate=1e-2, mu=0.99, reg=1e-12, decay=0.999):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.learning_rate = np.float32(learning_rate)
		self.mu = np.float32(mu)
		self.reg = np.float32(reg)
		self.decay = np.float32(decay)
		
		self.hidden_layers = []
		M1 = D
		count = 0
		for M2 in self.hidden_layer_sizes:
			layer = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(layer)
			M1 = M2
			count += 1
		self.W = (init_weight(M1, K)).astype(np.float32)
		self.b = (np.zeros(K)).astype(np.float32)

		self.params = [self.W, self.b]
		for layer in self.hidden_layers:
			self.params += layer.param

		self.build(D, K)

	def build(self, D, K):
		self.X_in = tf.placeholder(tf.float32, shape=(None, D))
		self.labels = tf.placeholder(tf.int32, shape=(None, K))
		logits = self.forward(self.X_in)
		r_cost = self.reg * sum([tf.nn.l2_loss(p) for p in self.params])
		self.cost = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(
				logits=logits, 
				labels=self.labels)) + r_cost
		self.train_op = tf.train.RMSPropOptimizer(
			self.learning_rate, decay=self.decay, momentum=self.mu).minimize(self.cost)
		self.prediction = self.predict(self.X_in)

	def set_session(self, session):
		self.session = session
		for layer in self.hidden_layers:
			layer.set_session(session)

	def fit(self, X, Y, epochs=400, batch_sz=20, print_period=1, show_fig=True):
		N, D = X.shape
		num_batch = N // batch_sz

		costs = []
		for i in range(epochs):
			shuffle_X, shuffle_Y = shuffle(X, Y)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz:(j*batch_sz+batch_sz)]
				y = shuffle_Y[j*batch_sz:(j*batch_sz+batch_sz)]

				_, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: x, self.labels: y})
				p = self.session.run(self.prediction, feed_dict={self.X_in: x})

				if j % print_period == 0:
					costs.append(c)
					e = np.mean(p != np.argmax(y, axis=1))
					print("i:", i, "j:", j, "cost:", c, "error_rate:", e)
		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		Z = X
		for layer in self.hidden_layers:
			Z = layer.forward(Z)
		return tf.nn.softmax(tf.matmul(Z, self.W) + self.b)

	def predict(self, X):
		Z = self.forward(X)
		return tf.argmax(Z, axis=1)


def main():
	X, Y = all_parity_pairs(12)
	X = X.astype(np.float32)
	Y = Y.astype(np.int32)
	K = len(set(Y))
	Y = (indicator(Y)).astype(np.float32)
	N, D = X.shape
	model = ANN(D, K, [1024, 1024])
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		model.set_session(session)
		model.fit(X, Y)

if __name__ == '__main__':
	main()



