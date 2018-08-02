import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from until import get_normalized_data
from sklearn.utils import shuffle

class Hiddenlayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1, M2) * np.sqrt (2.0 / M1)
		b = np.zeros(M2)
		self.W = theano.shared(W, 'W_%s' % self.id)
		self.b = theano.shared(b, 'b_%s' % self.id)
		self.param = [self.W, self.b]

	def forward(self, X):
		return T.nnet.relu(X.dot(self.W) + self.b)

def rmsprop(cost, param, lr, mu, decay, eps):
	grads = T.grad(cost, param)
	updates = []
	for p, g in zip(param, grads):
		#update cache
		ones = np.ones_like(p.get_value())
		cache = theano.shared(ones)
		update_cache = decay*cache + (np.float32(1.0)-decay)*g*g

		#update momentum
		zeros = np.zeros_like(p.get_value())
		momentum = theano.shared(zeros)
		update_momentum = mu*momentum - lr*g / T.sqrt(update_cache + eps)

		#udpate weight
		update_p = p + update_momentum

		updates.append([cache, update_cache])
		updates.append([momentum, update_momentum])
		updates.append([p, update_p])
	return updates

class ANN(object):
	def __init__(self, hiddenlayers, p_keep):
		self.hiddenlayers = hiddenlayers
		self.drop_rate = p_keep

	def forward_train(self, X):
		Z = X
		for h, d in zip(self.hidden_layer, self.drop_rate[:-1]):
			mask = self.rng.binomial(n=1, p=d, size=Z.shape)
			Z = mask * Z
			Z = h.forward(Z)
		mask = self.rng.binomial(n=1, p=self.drop_rate[-1], size=Z.shape)
		Z = mask*Z
		Z = T.nnet.softmax(Z.dot(self.W) + self.b)
		return Z

	def forward_predict(self, X):
		Z = X
		for h, d in zip(self.hidden_layer, self.drop_rate[:-1]):
			Z = h.forward(d * Z)
		Z = T.nnet.softmax((self.drop_rate[-1]*Z).dot(self.W) + self.b)
		return Z

	def predict(self, X):
		pY = self.forward_predict(X)
		return T.argmax(pY, axis=1)

	def error_rate(self, T, P):
		return np.mean(T != P)

	def fit(self, X, Y, test_X, test_Y, learning_rate=1e-4, mu=0.9, decay=0.9, eps=1e-10, epochs=8, batch_sz=100, show_fig=False):
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
		eps = np.float32(eps)

		train_X = X.astype(np.float32)
		train_Y = Y.astype(np.int32)
		test_X = test_X.astype(np.float32)
		test_Y = test_Y.astype(np.int32)
		self.rng = RandomStreams()

		#initialize hiddenlayers
		N, D = train_X.shape
		K = len(set(train_Y))
		M1 = D
		self.hidden_layer = []
		count = 0
		for M2 in self.hiddenlayers:
			h = Hiddenlayer(M1, M2, count)
			self.hidden_layer.append(h)
			count += 1
			M1 = M2
		W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
		b = np.zeros(K)
		self.W = theano.shared(W, 'W_logreg')
		self.b = theano.shared(b, 'b_logreg')

		#get parameters
		self.param = [self.W, self.b]
		for h in self.hidden_layer:
			self.param += h.param

		#init X, Y
		thX = T.fmatrix('X')
		thY = T.ivector('Y')

		#init train functions
		pY_train = self.forward_train(thX)
		cost_train = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))
		updates = rmsprop(cost_train, self.param, learning_rate, mu, decay, eps)
		train_op = theano.function(inputs=[thX, thY], updates=updates)

		#init predict & evaluation functions
		pY_predict = self.forward_predict(thX)
		cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]), thY]))
		prediction = self.predict(thX)
		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost_predict, prediction])

		num_batchs = N // batch_sz
		costs = []
		for i in range(epochs):
			shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
			for j in range(num_batchs):
				x = shuffle_X[j*batch_sz : (j*batch_sz+batch_sz), :]
				y = shuffle_Y[j*batch_sz : (j*batch_sz+batch_sz)]

				train_op(x, y)
				if j % 50 == 0:
					c, p = cost_predict_op(test_X, test_Y)
					costs.append(c)
					e = self.error_rate(test_Y, p)
					print("i:", i, "j:", j, "nb:", num_batchs, "cost:", c, "error rate:", e)
		if show_fig:
			plt.plot(costs)
			plt.show()


def main():
	train_X, test_X, train_Y, test_Y = get_normalized_data()
	model = ANN([500, 300], [0.8, 0.5, 0.5])
	model.fit(train_X, train_Y, test_X, test_Y, show_fig=True)


if __name__ == '__main__':
	main()








