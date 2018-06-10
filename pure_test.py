import numpy as np

a = [[5,5,5],[5,5,5],[5,5,5] ]

a = np.array(a, dtype=np.uint8)

b = np.array(a / 2.1 , dtype=np.uint8)
print(a)
print(b)