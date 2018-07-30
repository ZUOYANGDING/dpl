import tensorflow as tf
import numpy as np

Nclass = 500
D = 2
M = 3
K = 3

X1 = np.random.randn(Nclass, 2) + np.array([0,-2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])

X = np.vstack([X1, X2, X3]).astype(np.float32)
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

N = len(Y)
T = np.zeros((N, K))
for i in range(N):
	T[i, Y[i]] = 1

def init_weight(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X, W1, W2, b1, b2):
	Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
	return tf.matmul(Z, W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weight([D, M])
W2 = init_weight([M, K])
b1 = init_weight([M])
b2 = init_weight([K])

pY = forward(tfX, W1, W2, b1, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfY, logits=pY))

train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict = tf.argmax(pY, 1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100000):
	sess.run(train, feed_dict={tfX:X, tfY:T})
	pred = sess.run(predict, feed_dict={tfX:X, tfY:T})

	if i%10000 == 0:
		print(np.mean(Y == pred))
