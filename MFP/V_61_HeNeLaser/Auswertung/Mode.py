import numpy as np 
import scipy.constants as const


lambda_s = 633
f_s = const.c/(633*10**-9)

v= np.sqrt(8*const.k *(22+273)/(np.pi*20.18*const.m_p) )

lambda_b = (v/f_s)*10**9

print(lambda_b)