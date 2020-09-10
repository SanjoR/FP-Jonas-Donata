import numpy as np
from uncertainties import ufloat
import scipy.constants as const
Ds = np.genfromtxt("../Data/95_Abstand.txt", unpack=True)
ds = np.genfromtxt("../Data/97_Abstand.txt", unpack=True)

DLambda = 48.9 #pm

dLambda = 1/2 * ds/Ds *DLambda

for i in dLambda:
    print(round(i,2))

lam = ufloat(dLambda.mean(),dLambda.std())
print()
print(lam)
print()

lambdaa=643.8*10**-9
mu_B= 9.27*10**-24
B=ufloat(394,9)/1000


g = const.c*const.h/(lambdaa**2)* (lam*10**-12) /(mu_B*B)

print(g)