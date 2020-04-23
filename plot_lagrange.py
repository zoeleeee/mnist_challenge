import matplotlib.pyplot as plt
import numpy as np
from mpmath import *
import sys

file = sys.argv[-1]
nb_values = int(sys.argv[-2])

mp.dps = 1000
aa = np.load('lagrange/lag_'+file.split('/')[1])
bb = np.load('lagrange/lag_iter_'+file.split('/')[1])

for i in range(aa.shape[0]):
	a, b = aa[i], bb[i]

	x = np.array([mpf(val) for val in range(nb_values)])/nb_values
	xx = np.array([mpf(val) for val in range(256)])/255.
	yy = np.polyval(b, x)
	yyxx = np.polyval(b, xx)
        print(yyxx)
        print(yyxx.astype(np.float32))
	print(len(yyxx), [float(v) for v in yyxx])
	y = np.polyval(a, x)
	yx = np.polyval(a, xx)
	print(len(yx), [int(v*255) for v in yx])

	plt.clf()
	plt.ylim(-1,2)
	plt.plot(x, y, color='plum', lw=.3)
	plt.scatter(xx, yx, color='indigo', s=.5, alpha=1)
	plt.xticks(np.arange(0,1.0001,15/255), ['0/255', '15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255', '255/255'], fontsize='x-small', rotation=270)
	plt.yticks(np.arange(0,1.0001,15/255), ['0/255', '15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255', '255/255'], fontsize='x-small')

	plt.grid(linestyle='-')
	plt.savefig('lagrange/lagrange_16_{}.png'.format(i))
	plt.clf()

	#plt.ylim(-1e5,1e5)
	plt.yscale('symlog')
	plt.plot(x, yy, color='plum', lw=.3)
	plt.scatter(xx, [float(v) for v in yyxx], color='indigo', s=.5, alpha=1)
	plt.xticks(np.arange(0,1.0001,15/255), ['0/255', '15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255', '255/255'], fontsize='x-small', rotation=270)
	plt.yticks(fontsize='x-small')
	# plt.yticks(np.arange(0,1,15/255), ['15/255', '30/255', '45/255', '60/255', '75/255', '90/255', '105/255', '120/255', '135/255', '150/255', '165/255', '180/255', '195/255', '210/255', '225/255', '240/255'])
	plt.grid(axis='x', linestyle='-')
	plt.savefig('lagrange/lagrange_gradiants_16_{}.png'.format(i))
