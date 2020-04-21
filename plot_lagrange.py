import matplotlib.pyplot as plt
import numpy as np
from mpmath import *
import sys

file = sys.argv[-1]
nb_values = int(sys.argv[-2])

mp.dps = 1000
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
plt.xticks(np.arange(0,1,15/255), ['15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255'])
plt.yticks(np.arange(0,1,15/255), ['15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255'])

plt.grid(linestyle='-.')
plt.savefig('lagrange/plot_lagrange.png')
plt.clf()
plt.plot(x, yy, color='plum')
plt.scatter(xx, yyxx, color='lavender')
plt.xticks(np.arange(0,1,15/255), ['15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255'])
plt.yticks(np.arange(0,1,15/255), ['15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255'])
plt.grid(linestyle='-.')
plt.savefig('lagrange/plot_lagrange_gradiants.png')
