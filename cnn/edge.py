import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('lena.png')
bw = img.mean(axis=2)

Hx = np.array([
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0 ,1]], dtype=np.float32)

Hy = np.array([
	[-1, -2, -1],
	[0 ,0, 0],
	[1, 2, 1]], dtype=np.float32)

Gx = convolve2d(bw, Hx)
plt.imshow(Gx, cmap='gray')
plt.show()

Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap='gray')
plt.show()

# Gradient magnitude
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()

# The gradient's direction
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()