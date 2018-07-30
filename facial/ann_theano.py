import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import getData, getBinaryData, error_rate, relu, init_weight_bias
from sklearn.utils import shuffle

def rmsprop(cost, param, lr, mu, decay, eps):
	grads = T.grad(cost, param)
	updates = []
	for p, g in zip(param, grads):
		#update cache
		ones = np.ones_like(p.get_value(), dtype=np.float32)
		cache = theano.shared(ones)
		update_cache = decay*cache + (np.float32(1.0)-decay)*g*g

		#update momentum
		zeros = np.zeros_like(p.get_value(), dtype=np.float32)
		momentum = theano.shared(zeros)
		update_momentum = mu*momentum - lr*g / T.sqrt(update_cache + eps)

		#udpate weight
		update_p = p + update_momentum

		updates.append([cache, update_cache])
		updates.append([momentum, update_momentum])
		updates.append([p, update_p])
	return updates

class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_bias(M1, M2)
		self.W = theano.shared(W, 'W_%s' % self.id)
		self.b = theano.shared(b, 'b_%s' % self.id)
		self.param = [self.W, self.b]

	def forward(self, X):
		return relu(X.dot(self.W) + self.b)

class ANN(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes

	def th_forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return T.nnet.softmax(Z.dot(self.W)+self.b)
	
	def th_predict(self, X):
		pY = self.th_forward(X)
		return T.argmax(pY, axis=1)

	def fit(self, X, Y, learning_rate=1e-3, mu=0.9, decay=0.9, reg=0, eps=1e-10, epochs=100, batch_size=30, show_fig=False):
		learning_rate = np.float32(learning_rate)
		mu = np.float32(mu)
		decay = np.float32(decay)
		reg = np.float32(reg)
		eps = np.float32(eps)
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)
		train_X = X[:-1000, :]
		train_Y = Y[:-1000]
		valid_X = X[-1000:, :]
		valid_Y = Y[-1000:]

		N, D = train_X.shape
		K = len(set(train_Y))
		self.hidden_layers = []
		M1 = D
		count = 0
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			count += 1
			M1 = M2
		W, b = init_weight_bias(M1, K)
		self.W = theano.shared(W, 'W_out')
		self.b = theano.shared(b, 'b_out')

		self.param = [self.W, self.b]
		for h in self.hidden_layers:
			self.param += h.param

		#set up functions and variables
		thX = T.fmatrix('X')
		thY = T.ivector('Y')
		pY = self.th_forward(thX)

		rcost = reg*T.sum([(p*p).sum() for p in self.param])
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
		prediction = self.th_predict(thX)

		self.predict_opt = theano.function(inputs=[thX], outputs=prediction)
		cost_predict_opt = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		updates = rmsprop(cost, self.param, learning_rate, mu, decay, eps)
		train_op = theano.function(inputs=[thX, thY], updates=updates)

		num_batch = N // batch_size
		cost_array = []
		for i in range(epochs):
			shuffle_X, shuffle_Y = shuffle(train_X, train_Y)
			for j in range(num_batch):
				x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
				y = shuffle_Y[j*batch_size : (j*batch_size+batch_size)]

				train_op(x, y)
				if j%20 == 0:
					c, p = cost_predict_opt(valid_X, valid_Y)
					cost_array.append(c)
					e = error_rate(valid_Y, p)
					print("i:", i, "j:", j, "nb:", num_batch, "cost:", c, "error rate:", e)
		if show_fig:
			plt.plot(cost_array)
			plt.show()

def main():
	X, Y = getData()
	model = ANN([2000, 1000])
	model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
	main()