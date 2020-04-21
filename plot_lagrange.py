import matplotlib.pyplot as plt
import numpy as np
from mpmath import *
import sys

file = sys.argv[-1]
nb_values = int(sys.argv[-2])

mp.dps = 5000
a = np.load('lagrange/lag_'+file.split('/')[1]) [0]
b = np.load('lagrange/lag_iter_'+file.split('/')[1])[0]

x = np.arange(nb_values)/nb_values
xx = np.arange(255)/255.
y = np.polyval(a, x)
yx = np.polyval(a, xx)
yy = np.polyval(b, x)
yyxx = np.polyval(b, xx)

plt.plot(x, y, color='plum')
plt.scatter(xx, yx, color='lavender')
plt.xticks(np.arange(0,1,1/255))
plt.yticks(np.arange(0,1,1/255))
plt.grid(linestyle='-.')
plt.savefig('lagrange/plot_lagrange.png')

plt.plot(x, yy, color='plum')
plt.scatter(xx, yyxx, color='lavender')
plt.xticks(np.arange(0,1,1/255))
plt.yticks(np.arange(0,1,1/255))
plt.grid(linestyle='-.')
plt.savefig('lagrange/plot_lagrange_gradiants.png')
