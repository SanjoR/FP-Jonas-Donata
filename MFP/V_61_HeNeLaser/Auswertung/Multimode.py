import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.constants as const


L_Peaks = np.genfromtxt("../Data/Stabi_konkav_konkav.txt", unpack = True)


print(L_Peaks)

L = L_Peaks[:1,][0] /100
L_Peaks = np.delete(L_Peaks,(0),axis=0)
Peaks_1 = L_Peaks[:,0][:3]
Peaks_2 = L_Peaks[:,1][:5]
Peaks_3 = L_Peaks[:,2][:5]
Peaks_4 = L_Peaks[:,3]
print()
print("Länge : 73cm")
for i in range(len(Peaks_1)):
    print()
    print(const.c*(i+1) /(2*L[0])/10**6, Peaks_1[i])


print()
print("Länge : 84.8cm")
for i in range(len(Peaks_2)):
    print()
    print(const.c*(i+1) /(2*L[1])/10**6, Peaks_2[i])

    print()
print("Länge : 135cm")
for i in range(len(Peaks_3)):
    print()
    print(const.c*(i+1) /(2*L[2])/10**6, Peaks_3[i])

print()
print("Länge : 176.3cm")
for i in range(len(Peaks_4)):
    print()
    print(const.c*(i+1) /(2*L[3])/10**6, Peaks_4[i])



def linear(x,a,b):
    return(a*x+b)



x_plot_1 = np.arange(len(Peaks_1))+1
x_plot_2 = np.arange(len(Peaks_2))+1
x_plot_3 = np.arange(len(Peaks_3))+1
x_plot_4 = np.arange(len(Peaks_4))+1

params_1,covma_1 = curve_fit(linear,x_plot_1,Peaks_1)
params_2,covma_2 = curve_fit(linear,x_plot_2,Peaks_2)
params_3,covma_3 = curve_fit(linear,x_plot_3,Peaks_3)
params_4,covma_4 = curve_fit(linear,x_plot_4,Peaks_4)

errors_1 = np.sqrt(np.diag(covma_1))
errors_2 = np.sqrt(np.diag(covma_2))
errors_3 = np.sqrt(np.diag(covma_3))
errors_4 = np.sqrt(np.diag(covma_4))

params_error_1 = unp.uarray(params_1,errors_1)
params_error_2 = unp.uarray(params_2,errors_2)
params_error_3 = unp.uarray(params_3,errors_3)
params_error_4 = unp.uarray(params_4,errors_4)

a_1 = const.c/(2*L[0])/10**6
a_2 = const.c/(2*L[1])/10**6
a_3 = const.c/(2*L[2])/10**6
a_4 = const.c/(2*L[3])/10**6

y_plot_1 = a_1*x_plot_1
y_plot_2 = a_2*x_plot_2
y_plot_3 = a_3*x_plot_3
y_plot_4 = a_4*x_plot_4


fig, axis = plt.subplots(2,2 ,sharex = True , sharey = True)
axis[0,0].plot(x_plot_1,y_plot_1,"r-",label="Theoriegrade L = 73.0cm")
axis[0,0].plot(x_plot_1,linear(x_plot_1,*params_1),"r--", label ="Ausgleichsgrade L = 73.0cm")
axis[0,0].plot(x_plot_1, Peaks_1,"rx",label="Messdaten L= 73.0cm")
axis[0,0].grid()
axis[0,0].set_ylabel(r"$\nu$ / MHz")
axis[0,0].legend(loc="best")

axis[0,1].plot(x_plot_2,y_plot_2,"g-",label="Theoriegrade L = 84.8cm")
axis[0,1].plot(x_plot_2,linear(x_plot_2,*params_2),"g--", label ="Ausgleichsgrade L = 84.8cm")
axis[0,1].plot(x_plot_2, Peaks_2,"gx",label="Messdaten L= 84.8cm")
axis[0,1].grid()
axis[0,1].legend(loc="best")

axis[1,0].plot(x_plot_3,y_plot_3,"b-",label="Theoriegrade L = 135.0cm")
axis[1,0].plot(x_plot_3,linear(x_plot_3,*params_3),"b--", label ="Ausgleichsgrade L = 135.0cm")
axis[1,0].plot(x_plot_3, Peaks_3,"bx",label="Messdaten L= 135.0cm")
axis[1,0].set_ylabel(r"$\nu$ / MHz")
axis[1,0].set_xlabel(r"n")
axis[1,0].grid()
axis[1,0].legend(loc="best")

axis[1,1].plot(x_plot_4,y_plot_4,"k-",label="Theoriegrade L = 176.3cm")
axis[1,1].plot(x_plot_4,linear(x_plot_4,*params_4),"k--", label ="Ausgleichsgrade L = 176.3cm")
axis[1,1].plot(x_plot_4, Peaks_4,"kx",label="Messdaten L= 176.3cm")
axis[1,1].set_xlabel(r"n")
axis[1,1].grid()
axis[1,1].legend(loc="best")

#plt.xlabel("n")
#plt.ylabel(r"$\nu$ / MHz")

plt.show()


print()
print("Länge : 73cm")
print()
print("Theoriewert a :", a_1)
print("a,b")
for i in params_error_1:
    print(i)

print()
print("Länge : 84.3cm")
print()
print("Theoriewert a :", a_2)
print("a,b")
for i in params_error_2:
    print(i)

print()
print("Länge : 135cm")
print()
print("Theoriewert a :", a_3)
print("a,b")
for i in params_error_3:
    print(i)

print()
print("Länge : 176.3cm")
print()
print("Theoriewert a :", a_4)
print("a,b")
for i in params_error_4:
    print(i)

print("a in m/s*10**6")