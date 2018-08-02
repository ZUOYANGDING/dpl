import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from until import get_normalized_data
from sklearn.utils import shuffle

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

class ANN(object):
	def __init__(self, hiddenlayersize, p_keep):
		self.hidden_layer_size = hiddenlayersize
		self.drop_out = p_keep

	def forward(self, X):
		Z = X
		# Z = tf.nn.dropout(Z, self.drop_out[0])
		for h, d in zip(self.hidden_layers, self.drop_out[:-1]):
			Z = tf.nn.dropout(Z, d)
			Z = h.forward(Z)
			# Z = tf.nn.dropout(Z, d)
		Z = tf.nn.dropout(Z, self.drop_out[-1])
		return tf.matmul(Z, self.W) + self.b

	def test_forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		Z = tf.matmul(Z, self.W) + self.b
		return Z

	def predict(self, X):
		pY = self.test_forward(X)
		return tf.argmax(pY, axis=1)

	def error_rate(self, T, X):
		return np.mean(T != X)


	def fit(self, X, Y, test_X, test_Y, learning_rate=1e-4, mu=0.9, decay=0.9, epochs=15, batch_size=100, show_fig=False):
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)

		train_X = X.astype(np.float32)
		train_Y = Y.astype(np.int64)
		test_X = test_X.astype(np.float32)
		test_Y = test_Y.astype(np.int64)

		#inite weight
		N, D = train_X.shape
		K = len(set(train_Y))
		self.hidden_layers = []
		M1 = D
		for M2 in self.hidden_layer_size:
			h = Hiddenlayer(M1, M2)
			self.hidden_layers.append(h)
			M1 = M2
		W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
		b = np.zeros(K)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		#collecte parameters
		self.param = [self.W, self.b]
		for h in self.hidden_layers:
			self.param += h.param

		#init variables and functions
		inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
		labels = tf.placeholder(tf.int64, shape=(None,), name='labels' )
		logits = self.forward(inputs)
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits = logits,
			labels = labels))
		train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

		#init prediction and test function
		prediction = self.predict(inputs)
		test_logits = self.test_forward(inputs)
		test_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=test_logits,
			labels=labels))

		num_batches = N // batch_size
		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			for i in range(epochs):
				shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
				for j in range(num_batches):
					x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
					y = shuffle_Y[j*batch_size: (j*batch_size+batch_size)]

					session.run(train_op, feed_dict={inputs: x, labels: y})

					if j % 50 == 0:
						c = session.run(test_cost, feed_dict={inputs: test_X, labels: test_Y})
						p = session.run(prediction, feed_dict={inputs: test_X})
						costs.append(c)
						e = self.error_rate(test_Y, p)
						print("i:", i, "j:", j, "nb:", num_batches, "cost:", c, "error rate:", e)
		if show_fig:
			plt.plot(costs)
			plt.show()

def main():
	train_X, test_X, train_Y, test_Y = get_normalized_data()
	model = ANN([500, 300], [0.8, 0.5, 0.5])
	model.fit(train_X, train_Y, test_X, test_Y, show_fig=True)

if __name__ == '__main__':
	main()


		
