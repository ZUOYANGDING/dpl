import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getKaggleMNIST
from autoencoder_tf import DNN

class RBM(object):
	def __init__(self, D, M, an_id):
		self.D = D
		self.M = M
		self.an_id = an_id
		self.build(D, M)

	def set_session(self, session):
		self.session = session

	def build(self, D, M):
		self.W = tf.Variable(tf.random_normal(shape=(D, M)) * np.sqrt(2.0 / M))
		self.b = tf.Variable(np.zeros(D).astype(np.float32))
		self.c = tf.Variable(np.zeros(M).astype(np.float32))
		self.X_in = tf.placeholder(tf.float32, shape=(None, D))

		#forward
		p_h_given_v = tf.nn.sigmoid(tf.matmul(self.X_in, self.W) + self.c)
		self.p_h_given_v = p_h_given_v
		r = tf.random_uniform(shape=tf.shape(p_h_given_v))
		h_0 = tf.to_float(r < p_h_given_v)

		#backward reconstruction
		p_v_given_h = tf.nn.sigmoid(tf.matmul(h_0, tf.transpose(self.W)) + self.b)
		self.p_v_given_h = p_v_given_h
		r = tf.random_uniform(shape=tf.shape(p_v_given_h))
		v_1 = tf.to_float(r < p_v_given_h)

		cost = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(v_1))
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

		logits = self.forward_logits(self.X_in)
		self.cost_check = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				labels=self.X_in,
				logits=logits))

	def free_energy(self, v):
		b = tf.reshape(self.b, (self.D, 1))
		first_term = -tf.matmul(v, b)
		first_term = tf.reshape(first_term, (-1, ))
		second_term = -tf.reduce_sum(
			tf.nn.softplus(tf.matmul(v, self.W) + self.c),
			axis=1)
		return first_term + second_term

	def forward_hidden(self, X):
		Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)
		return Z

	def forward_logits(self, X):
		Z = self.forward_hidden(X)
		return tf.matmul(Z, tf.transpose(self.W)) + self.b

	def forward_output(self, X):
		return tf.nn.sigmoid(self.forward_logits(X))

	def transform(self, X):
		return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})

	def fit(self, X, epochs=1, batch_sz=100, show_fig=False):
		N, D = X.shape
		num_batch = N // batch_sz

		costs = []
		for i in range(epochs):
			print("pretrain.... %s" % self.an_id)
			shuffle_X = shuffle(X)
			for j in range(num_batch):
				x = shuffle_X[j*batch_sz:(j*batch_sz+batch_sz)]
				_, c = self.session.run((self.train_op, self.cost_check), feed_dict={self.X_in: x})

				if j % 10 == 0:
					print("j / n_batches:", j, "/", num_batch, "cost:", c)
				costs.append(c)
		if show_fig:
			plt.plot(costs)
			plt.show()

def main():
	train_X, train_Y, test_X, test_Y = getKaggleMNIST()
	train_X = train_X.astype(np.float32)
	test_X = test_X.astype(np.float32)
	N, D = train_X.shape
	K = len(set(train_Y))
	model = DNN(D, [1000, 750, 500], K, UnsupervisedModel=RBM)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		model.set_session(session)
		model.fit(train_X, train_Y, test_X, test_Y, pretrain=True, epochs=10)

if __name__ == '__main__':
	main()

