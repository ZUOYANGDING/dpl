from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

#pdf of 0 in normal distribution with mean=0, stv=10#
#norm.pdf(0) means pdf of 0 in normal distribution with mean=0, stv=1#
#  

# b = np.random.randn(10)
# print(norm.pdf(b))

#log pdf#
# print(norm.logpdf(b))

#cdf and log of cdf#
# print(norm.cdf(b))
# print(norm.logcdf(b))

#sample 10000 data with mean=5, stv=10#
c = 10*np.random.randn(10000) + 5
plt.hist(c, bins=20)
plt.show()
