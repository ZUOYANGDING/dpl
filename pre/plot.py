import matplotlib.pyplot as plt
import numpy as np

#generate 10 numbers from 0 to 10#
a = np.linspace(0, 10, 10)
b = np.sin(a)

# plt.plot(a,b)
# plt.scatter(a, b)
R = np.random.randn(1000)
plt.hist(R, bins=20)
plt.show()
