import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

M = df.as_matrix()
#get the first column from row 1 to the end#
im = M[0, 1:]
#reshape it to 28*28 2D matrix#
im = im.reshape(28, 28)

#show the image, set as gray means 0 is color black, and 255 is color white#
plt.imshow(255-im, cmap='gray')
plt.show()