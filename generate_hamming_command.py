import numpy as np 
import os
path = 'preds'
files = os.listdir(path)
lst = []
for f in files:
    if f.find('_0_HASH') == -1:
        continue
    if f.find('CW') == -1:
        continue
    if f.find('low')==-1 and f.find('high')==-1 and f.find('mix')==-1:
        continue
    if f.endswith('show.npy'):
        lst.append(f)

for f in lst:
    strs = f.split('_0_HASH_')
    print(strs)
    a = np.load(os.path.join(path, strs[0]+'_0_HASH_'+strs[1]))
    b = np.load(os.path.join(path, strs[0]+'_20_HASH_'+strs[1]))
    c = np.load(os.path.join(path, strs[0]+'_40_HASH_'+strs[1]))
    d = np.load(os.path.join(path, strs[0]+'_60_HASH_'+strs[1]))
    np.save(os.path.join(path, strs[0]+'_80_HASH_'+strs[1]), np.hstack((a,b,c,d)))
    

    


