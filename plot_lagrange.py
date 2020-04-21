import matplotlib.pyplot as plt
import numpy as np
from mpmath import *
import sys

file = sys.argv[-1]
nb_values = sys.argv

mp.dps = 5000
a = np.load('lagrange/lag_'+file.split('/')[1], params) 
b = np.load('lagrange/lag_iter_'+file.split('/')[1], res)

x = np.arange()
