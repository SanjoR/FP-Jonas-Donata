import numpy as np
import matplotlib.pyplot as plt

b1 = 1400



def g1g2(d):
    return(1-d/b1)
 

d_plot= np.linspace(0,2000,10000)
print(g1g2(1))

plt.plot(d_plot,g1g2(d_plot))
plt.show()
