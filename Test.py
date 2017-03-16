import numpy as np, matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import scipy.stats

data = np.random.randint(90, 110, 100)
data_out = [35, 10, 80, 75, 111, 214, 185]

X = np.concatenate((data, data_out))
np.random.shuffle(X)
#
# plt.plot(X, 'ro')
# plt.show()

norm = scipy.stats.norm(np.mean(X), np.std(X))
#
# plt.plot(X, norm.pdf(X), 'ro')
# plt.show()

print(norm.pdf(X))
print(X[np.where(norm.pdf(X) < 0.01)])
print(X[np.where(norm.pdf(X) < 0.02)])
print(X[np.where(norm.pdf(X) < 0.023)])
