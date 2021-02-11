import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 350, 10000)
y = ( 1 - x/140)*(1-x/280)


plt.figure()
plt.plot(x, y)

plt.xlabel(r"d")
plt.ylabel(r"g")
plt.legend(loc='best')
plt.savefig("d_quad.pdf")
plt.show()
plt.close()
x = np.linspace(0, 350, 10000)
y = 1-x/280

plt.plot(x, y)

plt.xlabel(r"d")
plt.ylabel(r"g")
plt.legend(loc='best')

plt.savefig('dlin.pdf')

