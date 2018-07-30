import numpy as np

# M1 = np.matrix([[1,2], [3,4]])

# print(M1)

# M2 = np.array(M1)

# print(M2)
# print(M2.T)

# M1 = np.zeros(10)
# print(M1)

# M2 = np.zeros((10, 10))
# print(M2)

#random number 10*10 matrix#
# R = np.random.random((10,10))
# print(R)

#random numbers in normal distribution 10*10 matrix#
# N = np.random.randn(10,10)
# print(N)
# print(N.mean())
# print(N.var())

#inverse matrix#
# M1 = np.matrix([[1,2], [3,4]])
# inversM1 = np.linalg.inv(M1)
# print(inversM1)

#det of matrix#
# detM1 = np.linalg.det(M1)
# print(detM1)

#diagonal elements of matrix
# digM1 = np.diag(M1)
# print(digM1)
# print(np.diag([1,2]))

#outer & inner product#
# a = np.array([1,2])
# b = np.array([3,4])
# print(np.outer(a,b))
# print(np.inner(a,b))

#trace (sum of diagnal elements of matrix)#
# M1 = np.matrix([[1,2], [3,4]])
# print(np.trace(M1))


#eigenvalue and eigen vectors#
N = np.random.randn(100,3)
#to get the cov of N we need to transpose the N first#
cov = np.cov(N.T)
print(np.linalg.eig(cov))




