import numpy as np

A = 18*np.exp(-1.4*3)

P = np.pi * 0.3**2/(4*np.pi * 32**2)

print(P)
print(A*P*10**6)
print(1.03*A*P*10**6)