import numpy as np

#solve Ax=b, which equal to np.linalg.inv(A).dot(b)#
A = np.array([[1,2], [3,4]])
b = np.array([1,2])
print(np.linalg.solve(A, b))