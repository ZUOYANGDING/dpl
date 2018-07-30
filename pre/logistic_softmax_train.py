import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_data

def y2indecater(Y, K):
	N = len(Y)
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, Y[i]] = 1
	return ind


X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
train_ind = y2indecater(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
test_ind = y2indecater(Ytest, K)

W = np.random.randn(D,K)
b = np.zeros(K)

def soft_max(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
	Z = X.dot(W) + b
	return soft_max(Z)

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)

def classificaiton_accuracy(P, Y):
	return np.mean(P == Y)

def cross_entropy(T, pY):
	return -np.mean(T * np.log(pY))


train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
	pY_train = forward(Xtrain, W, b)
	pY_test = forward(Xtest, W, b)

	train_cost = cross_entropy(train_ind, pY_train)
	test_cost = cross_entropy(test_ind, pY_test)
	train_costs.append(train_cost)
	test_costs.append(test_cost)

	W -= learning_rate * Xtrain.T.dot(pY_train - train_ind)
	b -= learning_rate * (pY_train - train_ind).sum(axis=0)

	if i % 1000 == 0:
		print(i, train_cost, test_cost)

print("Final train classification_rate:", classificaiton_accuracy(predict(pY_train), Ytrain))
print("Final test classification_rate:", classificaiton_accuracy(predict(pY_test), Ytest))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()