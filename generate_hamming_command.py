import numpy as np 
import os

fils = os.listdir('preds')
lst = []
for f in files:
    if f.find('_0_HASH') == -1:
        continue
    if f.find('MIM') == -1:
        continue
    if f.find('low')==-1 and f.find('high')==-1 and f.find('mix')==-1:
        continue
    if f.endswith('show.npy'):
        lst.append(f)

for f in lst:
    strs = f.split('_0_')
    a = np.load(strs[0]+'_0_'+strs[1])
    b = np.load(strs[0]+'_20_'+strs[1])
    c = np.load(strs[0]+'_40_'+strs[1])
    d = np.load(strs[0]+'_60_'+strs[1])
    np.save(strs[0]+'_80_'+strs[1], np.hstack((a,b,c,d)))
    

    


