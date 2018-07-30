import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

X1 = np.random.randn(Nclass, 2) + np.array([0,-2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])

X = np.vstack([X1, X2, X3])
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

# plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
# plt.show()

D = 2
M = 3
K = 3

W1 = np.random.randn(D, M)
W2 = np.random.randn(M, K)
b1 = np.random.randn(M)
b2 = np.random.randn(K)

def predict_given_x(X, W1, W2, b1, b2):
	Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	P = expA / expA.sum(axis=1, keepdims=True)
	return P

def prediction_accuracy(Y, P):
	total = 0
	correct = 0
	for i in range(len(Y)):
		total += 1
		if Y[i] == P[i]:
			correct += 1
	return float(correct) / float(total)


predition = predict_given_x(X, W1, W2, b1, b2)

P = np.argmax(predition, axis=1)

assert(len(P)==len(Y))

print("classification rate ", prediction_accuracy(Y, P))