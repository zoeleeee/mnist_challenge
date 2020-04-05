from mpmath import *
import numpy as np
from functools import reduce
import operator
mpf.dps = 100
n = 256

x = np.arange(n)
y = np.load('permutation/256_256.1_permutation.npy')

xx = np.array([mpf(val) for val in x]) / 255.
yy = np.array([mpf(val) for val in y]) / 255.

coef, s = [], []
#256*256
coef.append([yy[i] / reduce(operator.mul, [xx[i]-xx[j] if i!=j else 1 for j in range(len(xx))]) for i in range(len(xx))])
s.append(mpf(sum(x)))
coef.append(-[s[0] - i for i in range(n)])
s.append(mpf(sum([s[0]*j-j**2 for j in range(n)]))/2)
coef.append([s[1] - i*coef[1] for i in range(n)])

for i in range(3, n):
	sign = 1 if i%2==0 else -1
	s.append(mpf(sum([s[i-2]*j-(j**2)*coef[i-2][j] for j in range(n)]))/i)
	coef.append(sign*[s[i-1]-j*coef[i-1][j] for j in range(n)])

coef = np.array(coef)
param = [sum(coef[i] * coef[0]) if i！=0 else sum(coef[0]) for i in range(len(coef))]
print(param)

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
