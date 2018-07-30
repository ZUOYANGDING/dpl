import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 500
X = np.random.random((N, 2))*4 - 2 # in between (-2, +2)
Y = X[:,0]*X[:,1] # makes a saddle shape
# note: in this script "Y" will be the target,
#       "Yhat" will be prediction

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], Y)
# plt.show()

# make a neural network and train it
D = 2
M = 100 # number of hidden units

# layer 1
W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)

# layer 2
W2 = np.random.randn(M) / np.sqrt(M)
c = 0

def forward(X, W1, W2, b1, c):
	Z = np.tanh(X.dot(W1) + b1)
	pY = Z.dot(W2) + c
	return pY, Z

def derivative_w1(W2, Z, Y, pY):
	dz = np.outer(Y - pY, W2) * (1 - Z*Z)
	return X.T.dot(dz)

def derivative_b1(W2, Z, Y, pY):
	dz = np.outer(Y - pY, W2) * (1 - Z * Z)
	return dz.sum(axis=0)

def derivative_w2(Z, Y, pY):
	return (Y - pY).dot(Z)

def derivative_c(Y, pY):
	return (Y - pY).sum(axis=0)

def update(W1, W2, b1, c, Y, pY):
	learning_rate = 1e-4
	regulization_rate = 0.0001
	W1 += learning_rate * derivative_w1(W2, Z, Y, pY) - regulization_rate*W1
	W2 += learning_rate * derivative_w2(Z, Y, pY) - regulization_rate*W2
	b1 += learning_rate * derivative_b1(W2, Z, Y, pY) - regulization_rate*b1
	c += learning_rate * derivative_c(Y, pY) - regulization_rate*c
	return W1, W2, b1, c

def get_cost(pY, Y):
	return ((Y - pY)**2).mean()


costs = []
for i in range(2000):
	pY, Z = forward(X, W1, W2, b1, c)
	cost = get_cost(pY, Y)
	costs.append(cost)
	W1, W2, b1, c = update(W1, W2, b1, c, Y, pY)

	if i % 25 == 0:
		 print(cost)

# plot the costs
plt.plot(costs)
plt.show()


# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
pY, Z = forward(Xgrid, W1, W2, b1, c)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], pY, linewidth=0.2, antialiased=True)
plt.show()


# plot magnitude of residuals
Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - pY)

plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
plt.show()
