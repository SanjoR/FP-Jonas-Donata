import numpy as np 
import matplotlib.pyplot as plt

I = np.genfromtxt("../Messdaten/I_0_Messung.Spe",unpack = True)


I = I[10:]
I = I[:230]
x = np.arange(len(I))


a = 127

test = np.abs(x-a)
b = I[test==test.min()]
a = x[test==test.min()]

plt.figure()
#plt.hist(C,bins = 300,range=(10,230),histtype = "step")
plt.plot([x[I==I.max()],x[I==I.max()]],[0,I.max()],"b--",label="Photopeak 662 keV")
plt.plot([a,a],[0,b],"r--",label = "Comptonkante 478 keV")
plt.plot(x, I, "k-",label = "Messdaten" )
plt.grid()
plt.legend(loc="best")
plt.xlabel("Kanal")
plt.ylabel("Ereignisse")
plt.savefig("../latex-template/figure/Leer.pdf")
plt.show()