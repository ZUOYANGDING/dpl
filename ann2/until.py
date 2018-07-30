import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_spiral():
    # Idea: radius -> low...high
    #           (don't start at 0, otherwise points will be "mushed" at origin)
    #       angle = low...high proportional to radius
    #               [0, 2pi/6, 4pi/6, ..., 10pi/6] --> [pi/2, pi/3 + pi/2, ..., ]
    # x = rcos(theta), y = rsin(theta) as usual

    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi*i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    # convert into cartesian coordinates
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    # inputs
    X = np.empty((600, 2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()

    # add noise
    X += np.random.randn(600, 2)*0.5

    # targets
    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)
    return X, Y

 
def get_transformed_data():
	print("Reading and transforming data....")

	df = pd.read_csv("train.csv")
	data = df.values.astype(np.float32)
	np.random.shuffle(data)

	X = data[:, 1:]
	Y = data[:, 0].astype(np.int32)

	train_X = X[:-1000]
	train_Y = Y[:-1000]
	test_X = X[-1000:]
	test_Y = Y[-1000:]

	#center data
	mu = train_X.mean(axis=0)
	train_X = train_X - mu
	test_X = test_X - mu

	#transform data
	pca = PCA()
	train_Z = pca.fit_transform(train_X)
	test_Z = pca.transform(test_X)

	# plot_cumulative_variance(pca)
	train_Z = train_Z[:, :300]
	test_Z = test_Z[:, :300]

	#normalize data
	mu = train_Z.mean(axis=0)
	std = train_Z.std(axis=0)
	train_Z = (train_Z - mu) / std
	test_Z = (test_Z - mu) / std

	return train_Z, test_Z, train_Y, test_Y


def get_normalized_data():
	print("Reading and normalize data.....")

	df = pd.read_csv("train.csv")
	data = df.values.astype(np.float32)
	np.random.shuffle(data)
	X = data[:, 1:]
	Y = data[:, 0]

	train_X = X[:-1000]
	train_Y = Y[:-1000]
	test_X = X[-1000:]
	test_Y = Y[-1000:]

	#normalize the data
	mu = train_X.mean(axis=0)
	std = train_X.std(axis=0)
	#there are some std is 0
	np.place(std, std==0, 1)
	train_X = (train_X - mu) / std
	test_X = (test_X - mu) / std

	return train_X, test_X, train_Y, test_Y


def plot_cumulative_variance(pca):
	P = []
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p + P[-1])
	plt.plot(P)
	plt.show()
	return P


def forward(X, W, b):
	a = X.dot(W) + b
	expA = np.exp(a)
	pY = expA / expA.sum(axis=1, keepdims=True)
	return pY


def predict(pY):
	prediction = np.argmax(pY, axis=1)
	return prediction


def error_rate(pY, target):
	prediction = predict(pY)
	return np.mean(prediction != target)


def cost(pY, target):
	total = target * np.log(pY)
	return -total.sum()


def gradW(target, pY, X):
	return X.T.dot(target - pY)

def gradB(target, pY):
	return (target - pY).sum(axis=0)


def indicator(Y):
	N = len(Y)
	Y = Y.astype(np.int32)
	ind = np.zeros((N, 10))
	for i in range(N):
		ind[i, Y[i]] = 1
	return ind


def benchmark_full():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    print("Performing logistic regression...")
    # lr = LogisticRegression(solver='lbfgs')


    # convert Ytrain and Ytest to (N x K) matrices of indicator variables
    N, D = Xtrain.shape
    Ytrain_ind = indicator(Ytrain)
    Ytest_ind = indicator(Ytest)

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    # reg = 1
    # learning rate 0.0001 is too high, 0.00005 is also too high
    # 0.00003 / 2000 iterations => 0.363 error, -7630 cost
    # 0.00004 / 1000 iterations => 0.295 error, -7902 cost
    # 0.00004 / 2000 iterations => 0.321 error, -7528 cost

    # reg = 0.1, still around 0.31 error
    # reg = 0.01, still around 0.31 error
    lr = 0.00004
    reg = 0.01
    for i in range(500):
        p_y = forward(Xtrain, W, b)
        # print "p_y:", p_y
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)
        
        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradB(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()


def benchmark_pca():
    Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    Ytrain_ind = np.zeros((N, 10))
    for i in range(N):
        Ytrain_ind[i, Ytrain[i]] = 1

    Ntest = len(Ytest)
    Ytest_ind = np.zeros((Ntest, 10))
    for i in range(Ntest):
        Ytest_ind[i, Ytest[i]] = 1

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    # D = 300 -> error = 0.07
    lr = 0.0001
    reg = 0.01
    for i in range(200):
        p_y = forward(Xtrain, W, b)
        # print "p_y:", p_y
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradB(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()


if __name__ == '__main__':
    benchmark_pca()
    #benchmark_full()



