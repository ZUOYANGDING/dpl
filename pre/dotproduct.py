import numpy as np

a = np.array([1,2])
b = np.array([2,1])

# dot = 0
# for e,f in zip(a, b):
# 	dot += e*f

# print(dot)

# print(a*b)

# print(np.sum(a*b))

# print(np.dot(a, b))

# print((a*b).sum())

#dot product#
print(a.dot(b))

cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cosangle)

print(angle)
print(cosangle)

