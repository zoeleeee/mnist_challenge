import numpy as np 
import sys
import os
from numpy import linalg as LA

files = os.listdir('advs')
metric = sys.argv[-1]
sign = sys.argv[-2]
attack = sys.argv[-3]
lst = []
for f in files:
    if f.find('mnist') == -1:
        continue
    if not f.endswith('show.npy'):
        continue
    if f.find(attack) == -1:
        continue
    if f.find(sign) == -1:# and f.find('high') == -1:# and f.find('mix') == -1:
        continue
    if f.find(metric) == -1:
        continue
    lst.append(f)
list.sort(lst)
#lst = lst[-4:]
print(lst)
idxs = np.load('data/final_random_1000_correct_idxs.npy')+60000
imgs = np.load('data/mnist_data.npy')[idxs].astype(np.float32) / 255.
labs = np.load('data/mnist_labels.npy')[idxs]
print(imgs.shape, labs.shape)
#print(imgs[999]==0)
cnt = 0
num = 1000
for f in lst:
    cnt = 0
    print(f)
    f = os.path.join('advs', f)
    b = np.load(f).astype(np.float32) / 255.
    a = np.load(f[:-8]+'label.npy')
    if len(a) == num:
        np.save(f[:-8]+'idxs.npy', np.arange(cnt, cnt+num))
        cnt += num
        continue

    pos, tmp = [], []
    mindis, i = np.inf, 0
    for j in range(cnt, cnt+num):
        v = labs[j]
        if i >= len(a):
            break
#        print(i, v, a[i])
        if v == a[i]:
            if len(tmp) > 0:
                pos.append(np.random.permutation(tmp)[0])
                tmp = []
#            print(v, j)
#            mindis = LA.norm((b[i]-imgs[j]).reshape(-1), np.inf) if metric == 'linf' else  LA.norm((b[i]-imgs[j]).reshape(-1), 2)
            mindis = np.max(np.absolute(b[i]-imgs[j])) if metric == 'linf' else  np.sum((b[i]-imgs[j])**2)
            tmp = [j]
#            print(v, a[i-1], pos, mindis, tmp)
            i += 1
        elif i > 0 and v == a[i-1]:
            dis = np.max(np.absolute(b[i]-imgs[j])) if metric == 'linf' else  np.sum((b[i]-imgs[j])**2)
            if dis < mindis:
                mindis = dis
                tmp = [j]
            elif dis == mindis:
                tmp.append(j)
#            print(i, v, a[i-1], dis, mindis)
        
    if len(tmp) > 0:
        pos.append(np.random.permutation(tmp)[0])
    print(len(a), len(pos))
    assert len(a) == len(pos) #print(len(a), len(pos))
    np.save(f[:-8]+'idxs.npy', pos)
    cnt += num
    
