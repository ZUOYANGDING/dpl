import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from until import get_transformed_data, forward, error_rate, cost, gradW, gradB, indicator

# from util import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator

def main():
	train_X, test_X, train_Y, test_Y = get_transformed_data()
	N, D = train_X.shape
	train_Y_ind = indicator(train_Y)
	test_Y_ind = indicator(test_Y)

	#full
	W = np.random.randn(D, 10) / np.sqrt(D)
	b = np.zeros(10)
	LL =[]
	learning_rate = 0.0001
	reg = 0.01
	t0 = datetime.now()

	for i in range(50):
		pY = forward(train_X, W, b)

		W += learning_rate * (gradW(train_Y_ind, pY, train_X) - reg*W)
		b += learning_rate * (gradB(train_Y_ind, pY) - reg*b)

		p_test = forward(test_X, W, b)
		ll = cost(p_test, test_Y_ind)
		LL.append(ll)

		error = error_rate(p_test, test_Y)
		if i % 10 == 0:
			print("cost as iteration %d: %6f" % (i,ll))
			print("error_rate is ", error)
	py_test = forward(test_X, W, b)
	print("Final error_rate is ", error_rate(py_test, test_Y))
	print("time cost for full GD ", datetime.now() - t0)

	#stochastic
	W = np.random.randn(D, 10) / np.sqrt(D)
	b = np.zeros(10)
	LL_st = []
	learning_rate = 0.0001
	reg = 0.01
	t0 = datetime.now()

	for i in range(50):
		#only go through one path of the dataset, more pathes will perform better, but take too long
		shuffled_X, shuffled_Y_ind = shuffle(train_X, train_Y_ind)
		not_printed = True
		for j in range(min(N, 500)):
			#Only apply 500 samples of the dataset, else take too long, but perform better
			x = train_X[j, :].reshape(1, D)
			y = train_Y_ind[j, :].reshape(1, 10)
			pY = forward(x, W, b)

			W += learning_rate * (gradW(y, pY, x) - reg*W)
			b += learning_rate * (gradB(y, pY) - reg*b)

			p_test = forward(test_X, W, b)
			ll = cost(p_test, test_Y_ind)
			LL_st.append(ll)

			error = error_rate(p_test, test_Y)
			if i % 10 == 0:
				if not_printed:
					not_printed = False
					print("cost as iteration %d: %6f" % (i,ll))
					print("error_rate is ", error)
	py_test = forward(test_X, W, b)
	print("Final error_rate is ", error_rate(py_test, test_Y))
	print("time cost for stochastic GD ", datetime.now() - t0)



	#batch
	W = np.random.randn(D, 10) / np.sqrt(D)
	b = np.zeros(10)
	LL_batch = []
	learning_rate = 0.0001
	reg = 0.01
	t0 = datetime.now()
	batch_size = 500
	batch_num = N // batch_size

	for i in range(50):
		shuffle_X, shuffled_Y_ind = shuffle(train_X, train_Y_ind)
		not_printed = True
		for j in range(batch_num):
			x = shuffle_X[j*batch_size : (j*batch_size+batch_size), :]
			y = shuffled_Y_ind[j*batch_size : (j*batch_size+batch_size), :]
			pY = forward(x, W, b)

			W += learning_rate * (gradW(y, pY, x) - reg*W)
			b += learning_rate * (gradB(y, pY) - reg*b)

			p_test = forward(test_X, W, b)
			ll = cost(p_test, test_Y_ind)
			LL_batch.append(ll)

			error = error_rate(p_test, test_Y)
			if i % 10 == 0:
				if not_printed:
					not_printed = False
					print("cost as iteration %d: %6f" % (i,ll))
					print("error_rate is ", error)
	py_test = forward(test_X, W, b)
	print("Final error_rate is ", error_rate(py_test, test_Y))
	print("time cost for batch GD ", datetime.now() - t0)

	x1 = np.linspace(0, 1, len(LL))
	plt.plot(x1, LL, label="full")
	x2 = np.linspace(0, 1, len(LL_st))
	plt.plot(x2, LL_st, label="stochastic")
	x3 = np.linspace(0, 1, len(LL_batch))
	plt.plot(x3, LL_batch, label="batch")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()