from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt

M = np.random.randn(10000, 2)

M[:,1] = 2*M[:,1] + 5
# plt.scatter(M[:,0], M[:,1])
# plt.axis('equal')
# plt.show() 

#sample 2D matrix with cov=0.8, mean1=0, mean2=2, var1=1, var2=3#
cov = np.array([[1, 0.8], [0.8, 3]])
mu = np.array([0,2])
r = mvn.rvs(mean=mu, cov=cov, size=10000)
# plt.scatter(r[:,0], r[:,1])
# plt.axis('equal')
# plt.show()

#also can be done with np

r = np.random.multivariate_normal(mean=mu, cov=cov, size=10000)
plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
plt.show()