from mpmath import *
import numpy as np
from functools import reduce
import operator
import sys
mp.dps = 1000
n = 256

np.random.seed(777)
#xx = range(n)
#yy = np.random.permutation(xx)

x = range(n)
file = sys.argv[-1]
y = np.load(file).reshape(256,-1).transpose((1,0))

# xx = np.array([mpf(val) for val in x])
# yy = np.array([mpf(val) for val in x])
# yy = np.array([mpf(str(val)) for val in y])
# xx = [mpf(int(reduce(operator.add, [bin(int(v))[2:].zfill(8) for v in c]), 2)) for c in y]


# coef, s = [], []
# #256*256
# coef.append([yy[i] / reduce(operator.mul, [xx[i]-xx[j] if i!=j else 1 for j in range(len(xx))]) for i in range(len(xx))])
# s.append(mpf(sum(xx)))
# coef.append([s[0] - xx[i] for i in range(n)])
# s.append(mpf(sum([s[0]*xx[j]-xx[j]**2 for j in range(n)]))/2)
# coef.append([s[1] - xx[i]*coef[1][i] for i in range(n)])

# for i in range(3, n):
#   sign = 1 if i%2==0 else -1
#   s.append(mpf(sum([s[i-2]*xx[j]-(xx[j]**2)*coef[i-2][j] for j in range(n)]))/i)
#   coef.append([s[i-1]-xx[j]*coef[i-1][j] for j in range(n)])

# coef = np.array(coef)
# param = [sum(coef[i] * coef[0]) if i!=0 else sum(coef[0]) for i in range(len(coef))]

# for j, v in enumerate(xx):
#   tmp = sum([(v**i)*param[n-i-1] if (n-1)%2 == i%2 else -1*(v**i)*param[n-i-1] for i in range(n)])
#   print(tmp==yy[j], tmp, float(tmp)-float(yy[j]))

# param = [param[i]*-1 if (n-1)%2==i%2 else param[i] for i in range(n)]
# for j, v in enumerate(xx):
#     tmp = np.polyval(param, v)
#     print(tmp==yy[j], tmp, float(tmp)-float(yy[j]))
#param = [param[i]*(n-i-1) if (n-1)%2==i%2 else -1*param[i]*(n-i-1) for i in range(n-1)]
# with open("lagrange_weights.txt", "w") as f:
#     for s in param:
#         f.write(str(s) +"\n")
# np.save('lagrange_weights.npy', param)

xx = np.array([mpf(val) for val in x]) / 255.
yyy = np.array([mpf(str(val)) for val in y.reshape(-1)]).reshape(y.shape) / 255.
res, params = [], []
for t in range(yyy.shape[0]):
    print('Iteration:', t)
    coef, s = [], []
    yy = yyy[t]
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
    param = [param[i]*-1 if (n-1)%2==i%2 else param[i] for i in range(n)]
    params.append(param)
    #param = [param[i]*(n-i-1) for i in range(n-1)]
    for j, v in enumerate(xx):
        tmp = np.polyval(param, v)
        if np.around(tmp*255) != y[t][j]:
            print(y[t][j], np.around(tmp*255))
        # print(int(tmp*255), int(tmp*255)==y[t][j])#(float(tmp)-float(yy[j]))
    param = [param[i]*(n-i-1)*-1 if (n-1)%2==i%2 else param[i]*(n-i-1) for i in range(n-1)]
    res.append(param)
np.save('lagrange/lag_'+file.split('/')[1], params) 
np.save('lagrange/lag_iter_'+file.split('/')[1], res)
#np.save('lagrange/lag_forward_'+file.split('/')[1], res)

#np.save('lagrange_weights_sign.h5', np.sign(param))
# print(param)


## TEST
# def parameters(n):
#   # x = np.arange(n)
#   coef, s = [], []
#   coef.append([1]*n)
#   s.append(sum(np.arange(n)))
#   coef.append([s[0]-i for i in range(n)])
#   for i in range(2,n):
#       s.append(sum([s[i-2]*j-(j**2)*coef[i-2][j] for j in range(n)])/i)
#       coef.append([s[i-1]-j*coef[i-1][j] for j in range(n)])
#   return coef, np.sum(coef, axis=-1)

# from itertools import combinations
# for d in range(n):
#   sum([c for c in combinations(range(n),d)])
