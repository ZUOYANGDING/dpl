import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('lena.png')

#with black edge
# def convolve2d(X, W):
# 	m1, m2 = X.shape
# 	n1, n2 = W.shape
# 	Y = np.zeros((m1+n1-1, m2+n2-1))
# 	for i in range(m1+n1-1):
# 		for ii in range(n1):
# 			for j in range(m2+n2-1):
# 				for jj in range(n2):
# 					if i >= ii and j >= jj and i-ii<m1 and j-jj<m2:
# 						Y[i,j] += W[ii,jj] * X[i-ii, j-jj]
# 	return Y

#not with the black edge
def convolved2d_e(X, W):
	m1, m2 = X.shape
	n1, n2 = W.shape
	Y = np.zeros((m1+n1-1, m2+n2-1))
	for i in range(m1):
		for j in range(m2):
			Y[i:i+n1, j:j+n2] += X[i, j] * W
	ret = Y[n1//2:-n1//2+1, n2//2:-n2//2+1]
	assert(ret.shape == X.shape)
	return ret

bw = img.mean(axis=2)
print(bw.shape)

#create Guassin filter
W = np.zeros((20, 20))
for i in range(20):
	for j in range(20):
		dist = (i-9.5)**2 + (j-9.5)**2
		W[i,j] = np.exp(-dist / 50.)
out = convolved2d_e(bw, W)
plt.imshow(out, cmap='gray')
plt.show()
# print(out.shape)
# print(bw.shape)
# print(img.shape)

out = np.zeros(img.shape)
W /= W.sum()
for i in range(3):
	out[:,:,i] = convolved2d_e(img[:,:,i], W)
plt.imshow(out)
plt.show()