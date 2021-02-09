import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.constants as const
import sys

if sys.argv[1]== "100":
    Maxima = np.genfromtxt("../Data/Beigungmax_100.txt",unpack=True)
    Abstand = 25 
    a = 1/(100*1000)
if sys.argv[1]== "600":
    Maxima= np.genfromtxt("../Data/Beigungmax_600.txt",unpack=True)
    Abstand = 5
    a = 1/(600*1000)

#print(a)

Maxima[Maxima<0] *= -1


Maxima = np.sort(Maxima)
Maxima = Maxima[1:]
for i in Maxima:
    print(i)
print()
phi = np.arctan(Maxima/Abstand)
for i in phi:
    print(round(i,3))

lambda_normal = []

print()
n=1
for i in range(0,len(Maxima),2):
    print(round(a*np.sin(phi[i])/(n)*10**9,1))
    print(round(a*np.sin(phi[i+1])/(n)*10**9,1))
    lambda_normal = np.append(lambda_normal,a*np.sin(phi[i])/(n))
    lambda_normal = np.append(lambda_normal,a*np.sin(phi[i+1])/(n))
    n+=1

lambdaa = ufloat(np.mean(lambda_normal)*10**9,np.std(lambda_normal)*10**9)
A = ufloat(639,4)
B = ufloat(664,14)
print()
print()
print(lambdaa)
print()
print((A+B)/2)

