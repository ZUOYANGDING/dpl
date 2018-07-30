import numpy as np
import matplotlib.pyplot as plt


def predict_given_x(X, W1, W2, b1, b2):
	Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	P = expA / expA.sum(axis=1, keepdims=True)
	return P, Z

def prediction_accuracy(Y, P):
	total = 0
	correct = 0
	for i in range(len(Y)):
		total += 1
		if Y[i] == P[i]:
			correct += 1
	return float(correct) / float(total)

def derivative_w2(T, output, hidden):
	ret = hidden.T.dot(T - output)
	return ret

def derivative_w1(T, output, W2, hidden, X):
	dz = (T - output).dot(W2.T) * hidden * (1 - hidden) #(N*K).dot(M*K.T) * N*M * N*M
	ret = X.T.dot(dz)
	return ret

def derivative_b2(T, output):
	return (T - output).sum(axis=0)

def derivative_b1(T, W2, hidden, output):
	ret = ((T - output).dot(W2.T) * hidden * (1 - hidden)).sum(axis=0)
	return ret

def cost(T, output):
	total_cost  = T * np.log(output)
	return total_cost.sum()

def main():
	Nclass = 500
	D = 2
	M = 3
	K = 3

	X1 = np.random.randn(Nclass, 2) + np.array([0,-2])
	X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
	X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])

	X = np.vstack([X1, X2, X3])
	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	N = len(Y)
	T = np.zeros((N, K))
	for i in range(N):
		T[i, Y[i]] = 1

	W1 = np.random.randn(D,M)
	W2 = np.random.randn(M,K)
	b1 = np.random.randn(M)
	b2 = np.random.randn(K)

	learning_rate = 10e-7
	costs = []

	for propb in range(100000):
		output, hidden  = predict_given_x(X, W1, W2, b1, b2)

		if propb%100 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis=1)
			accurate = prediction_accuracy(Y, P)
			print ("cost:", c, "classfication accuracy: ", accurate)
			costs.append(c)

		W1 += learning_rate * derivative_w1(T, output, W2, hidden, X)
		W2 += learning_rate * derivative_w2(T, output, hidden)
		b1 += learning_rate * derivative_b1(T, W2, hidden, output)
		b2 += learning_rate * derivative_b2(T, output)

	plt.plot(costs)
	plt.show()





if __name__ == '__main__':
	main()
