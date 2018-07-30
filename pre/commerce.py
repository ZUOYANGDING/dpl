import numpy as np
from process import get_data

X, Y  = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
W2 = np.random.randn(M, K)
b1 = np.random.randn(M)
b2 = np.random.randn(K)

def soft_max(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def forward_prection(X, W1, W2, b1, b2):
	Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
	# Z = np.tanh(X.dot(W1) + b1)
	A = Z.dot(W2) + b2
	return soft_max(A)

def predict_accuracy(Y, P):
	return np.mean(Y == P)

prediction = forward_prection(X, W1, W2, b1, b2)
P = np.argmax(prediction, axis = 1)

assert(len(P)==len(Y))

print("predict accuracy is ", predict_accuracy(Y, P))
