import imp


import numpy as np
a = np.arange(12).reshape((4,3))
print(a)
print(np.gradient(a, axis=0))
print(np.gradient(a, np.array([0.1, 0.3, 0.6, 1]), axis=0))