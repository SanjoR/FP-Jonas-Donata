import numpy as np
from uncertainties import ufloat
import scipy.constants as const
Ds = np.genfromtxt("../Data/blau_Abstand_B0.txt", unpack=True)
ds = np.genfromtxt("../Data/blau_Abstand_B1009.txt", unpack=True)

DLambda = 27.0 #pm

dLambda = 1/2 * ds/Ds *DLambda

for i in dLambda:
    print(round(i,2))

lam = ufloat(dLambda.mean(),dLambda.std())
print()
print(lam)

lambdaa=480*10**-9
mu_B= 9.27*10**-24
B=1.009


g = const.c*const.h/(lambdaa**2)* (lam*10**-12) /(mu_B*B)

print(g)