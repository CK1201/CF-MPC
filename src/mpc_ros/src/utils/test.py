import imp


import numpy as np
# a = np.arange(12).reshape((4,3))
# print(a)
# print(np.gradient(a, axis=0))
# print(np.gradient(a, np.array([0.1, 0.3, 0.6, 1]), axis=0))

a = np.array([0.1,0.2,0.3, 0.4, 0.5,0.5]) * 2
a = np.concatenate((a[np.newaxis,:] / 2, np.ones((1,6)) * 0.5), axis=0)
# print(a)
a = a.min(0)[:,np.newaxis]
# print(a)

p1 = np.array([1,1,1])
p2 = np.array([3,4,5])
dp = np.repeat((p2 - p1)[np.newaxis,:], repeats=6, axis=0)
a = np.repeat(a, repeats=3, axis=1)
print(a)
print(dp)
print(dp * a)