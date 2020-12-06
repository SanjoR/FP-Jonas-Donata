import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.constants as const 

data_f = np.genfromtxt("echo_gradient_fft.txt")
data = np.genfromtxt("../Data/scope_7.csv", delimiter=",")

peaks,_ = find_peaks(-data_f[:,1], 50)


print(data_f[:,0][peaks][1]-data_f[:,0][peaks][0])


fig, (ax1,ax2) = plt.subplots(1,2)


ax1.plot(data[:,0][data[:,0]>data[:,0][np.argmax(data[:,1])]],data[:,1][data[:,0]>data[:,0][np.argmax(data[:,1])]])
ax2.plot(data_f[:,0],data_f[:,1])
ax2.plot(data_f[:,0][peaks],data_f[:,1][peaks],"rx")
ax2.set_xlim(-25000,25000)

ax1.set_title("Messdaten")
ax2.set_title("Fouriertransformation")
plt.savefig("../latex-template/figure/Fouriertrafo.pdf")
#plt.show()
#
#plt.plot(data_f[:,0],data_f[:,1])
#plt.plot(data_f[:,0][peaks],data_f[:,1][peaks],"rx")
#plt.xlim(-25000,25000)
#plt.show()plt
TD = -57341
d = 0.0044
df = data_f[:,0][peaks][1]-data_f[:,0][peaks][0]
D = -3/2 * 1/(TD* (2*np.pi * df/d) **2 )

print(D)
eta = 1.0087/1000 #http://www.wasser-wissen.de/abwasserlexikon/v/viskositaet.htm
r = const.k *(22.2+273)/(6 * np.pi * D * eta)
print(eta)


print(r)
m = 18.01528 / const.N_A /1000
print(m)

rho = 997

r_hcp = ( (m*0.74/(4/3 * np.pi *rho)) )**(1/3)
print(r_hcp)