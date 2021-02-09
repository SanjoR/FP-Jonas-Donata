import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.special import eval_hermite as H_n


print(ufloat(0.074,0.007)*180/np.pi)
print(180/np.pi)