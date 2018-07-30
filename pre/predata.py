import numpy as np
import pandas as pd

# X = []
# for line in open("data_2d.csv"):
# 	row = line.split(',')
# 	sample = list(map(float, row))
# 	X.append(sample)
# X = np.array(X)
# print(X.shape) 

X = pd.read_csv("data_2d.csv", header = None)
# print(type(X))
# print(X.info())
# print(X.head())
M = X.as_matrix()

#choose first column of X#
# print(X[0])

#choose first row of X#
# print(X.ix[0])

#choose first 2 columns#
print(X[[0,2]])

#choose rows that first columns less than 5#
print(X[X[0]<5])