import numpy as np

def forward(X, W1, W2, b1, b2):
	# #sigmoid
	# Z = 1 / (1 + np.exp(-(X.dot(W1) + b1)))

	# #tanh
	# Z = np.tanh(X.dot(W1) + b1)

	#relu
	Z = X.dot(W1) + b1
	Z[Z<0] = 0

	A = Z.dot(W2) + b2
	expA = np.exp(A)
	pY = expA / expA.sum(axis=1, keepdims=True)
	return pY, Z


def derivative_w2(Z, T, Y):
	return Z.T.dot(Y - T)

def derivative_b2(T, Y):
	return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
	# #sigmoid
	# dz = (Y - T).dot(W2.T) * (Z * (1 - Z))

	# #tanh
	# dz = (Y - T).dot(W2.T) * (1 - Z*Z)

	#relu
	dz = (Y - T).dot(W2.T) * (Z > 0)

	return X.T.dot(dz)

def derivative_b1(Z, T, Y, W2):
	# #sigmoid
	# dz = (Y - T).dot(W2.T) * (Z * (1-Z))

	# #tanh
	# dz = (Y - T).dot(W2.T) * (1 - Z * Z)

	#relu
	dz = (Y - T).dot(W2.T) * (Z > 0)

	return dz.sum(axis=0)


