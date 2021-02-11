import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,400, 10000)
y = ( 1 - x/140)*(1-x/140)


plt.figure()

plt.plot(x, y)
plt.grid()

plt.xlabel(r"d / cm")
plt.ylabel(r"$g_1 \cdot g_2$")
plt.title(r"zwei konkave Reflektorspiegel")
plt.savefig("../latex-template/figure/d_quad.pdf", bbox_inches='tight')

plt.show()
plt.close()

x = np.linspace(0, 400, 10000)
y = 1-x/140

plt.plot(x, y)
plt.grid()

plt.xlabel(r"d / cm")
plt.ylabel(r"$g_1 \cdot g_2$")
plt.title(r"ein konkaver und ein flacher Reflektorspiegel")
plt.savefig('../latex-template/figure/dlin.pdf', bbox_inches='tight')
plt.show()
