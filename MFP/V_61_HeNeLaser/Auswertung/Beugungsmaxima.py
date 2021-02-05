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
    Abstand = 5 
    a = 1/(100*1000)
if sys.argv[1]== "600":
    Maxima= np.genfromtxt("../Data/Beigungmax_600.txt",unpack=True)
    Abstand = 25
    a = 1/(600*1000)

Max_test_1 = Maxima[Maxima<0]*-1
Max_test_2 = Maxima[Maxima>0]

Maxima[Maxima<0] *= -1
Maxima_abstand = np.abs(np.diff(Maxima))/100


print(Maxima_abstand)
Maxima = np.sort(Maxima)
Maxima = Maxima[1:]
phi = np.arctan(Maxima/Abstand)
for i in phi:
    print(i)
n=1
for i in range(0,len(Maxima),2):
    print(i)
    print(n)
    print(a*np.sin(phi[i])/(n))
    print(a*np.sin(phi[i+1])/(n))
    n+=1

print()
lambda_test_1 = []
lambda_test_2 = []
phi_test_1 = np.arctan(Max_test_1/Abstand)
phi_test_2 = np.arctan(Max_test_2/Abstand)
for i in range(len(phi_test_1)):
    print(a*np.sin(phi_test_1[i])/(i+1))
    lambda_test_1 = np.append(lambda_test_1,a*np.sin(phi_test_1[i])/(i+1))
print()
for i in range(len(phi_test_2)):
    print(a*np.sin(phi_test_2[i])/(i+1)*10**9)
    lambda_test_2 = np.append(lambda_test_2,a*np.sin(phi_test_2[i])/(i+1) *10**9 )

print()
print(np.mean(lambda_test_1))
print(np.mean(lambda_test_2))


