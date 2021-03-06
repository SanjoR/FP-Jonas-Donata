import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt

z,B = np.genfromtxt("../Data/BFeld_Data.txt", unpack=True)

print(B.max()/1000)
plt.figure()
plt.plot(z,B)
plt.plot(z[B.argmax()],B.max(),"rx",label =f"Maximalwert = {B.max()} mT")
plt.legend(loc="best")
plt.xlabel("z / mm")
plt.ylabel("B / mT")
plt.savefig("../latex-template/figure/BFeld_plot.pdf")
plt.show()
plt.close()