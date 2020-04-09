from mpmath import *
import numpy as np
from functools import reduce
import operator
mp.dps = 1000
n = 256

np.random.seed(777)
#xx = range(n)
#yy = np.random.permutation(xx)

x = range(n)
y = list(np.load('permutation/256_256.1_permutation.npy'))

xx = np.array([mpf(val) for val in x])
yy = np.array([mpf(str(val)) for val in y])


coef, s = [], []
#256*256
coef.append([yy[i] / reduce(operator.mul, [xx[i]-xx[j] if i!=j else 1 for j in range(len(xx))]) for i in range(len(xx))])
s.append(mpf(sum(xx)))
coef.append([s[0] - xx[i] for i in range(n)])
s.append(mpf(sum([s[0]*xx[j]-xx[j]**2 for j in range(n)]))/2)
coef.append([s[1] - xx[i]*coef[1][i] for i in range(n)])

for i in range(3, n):
	sign = 1 if i%2==0 else -1
	s.append(mpf(sum([s[i-2]*xx[j]-(xx[j]**2)*coef[i-2][j] for j in range(n)]))/i)
	coef.append([s[i-1]-xx[j]*coef[i-1][j] for j in range(n)])

coef = np.array(coef)
param = [sum(coef[i] * coef[0]) if i!=0 else sum(coef[0]) for i in range(len(coef))]
with open("lagrange_weights.txt", "w") as f:
    for s in param:
        f.write(str(s) +"\n")
np.save('lagrange_weights.npy', param)
#np.save('lagrange_weights_sign.h5', np.sign(param))
print(param)

for j, v in enumerate(xx):
	tmp = sum([(v**i)*param[n-i-1] if (n-1)%2 == i%2 else -1*(v**i)*param[n-i-1] for i in range(n)])
	print(tmp==yy[j], float(tmp)-float(yy[j]))

## TEST
# def parameters(n):
# 	# x = np.arange(n)
# 	coef, s = [], []
# 	coef.append([1]*n)
# 	s.append(sum(np.arange(n)))
# 	coef.append([s[0]-i for i in range(n)])
# 	for i in range(2,n):
# 		s.append(sum([s[i-2]*j-(j**2)*coef[i-2][j] for j in range(n)])/i)
# 		coef.append([s[i-1]-j*coef[i-1][j] for j in range(n)])
# 	return coef, np.sum(coef, axis=-1)

# from itertools import combinations
# for d in range(n):
# 	sum([c for c in combinations(range(n),d)])
