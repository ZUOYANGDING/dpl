import numpy as np

def init_weight(m1, m2):
	return np.random.randn(m1, m2) / np.sqrt(m1+m2)

def all_parity_pairs(nbit):
	N = 2**nbit
	N_total = N + (100 - (N%100))
	X = np.zeros((N_total, nbit))
	Y = np.zeros(N_total)
	for i in range(N_total):
		k = i % N
		for j in range(nbit):
			if k % (2**(j+1)) != 0:
				k -= 2**j
				X[i, j] = 1
		Y[i] = X[i].sum() % 2
	return X, Y

def indicator(Y):
	N = len(Y)
	K = len(set(Y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, Y[i]] = 1
	return ind