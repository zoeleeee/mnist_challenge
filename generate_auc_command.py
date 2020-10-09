import os
import sys

path = sys.argv[-1]
files = os.listdir(path)
test_file = ['/home/zhuzby/mnist_challenge/preds/pred_256.16_mnist_20_hash_80_HASH_origin.npy',
            '/home/zhuzby/mnist_challenge/preds/pred_256.32_mnist_20_hash_80_HASH_origin.npy']
lst = []
for f in files:
    if f.find('_80_') == -1:
        continue
    if f.find('_HASH_') == -1:
        continue
    if f.find('CW') == -1:
        continue
#    if f.find('linf') == -1:
#       continue
    if f.find('256.16') == -1 and f.find('256.32') == -1:
        continue
    if f.find('low')==-1 and f.find('high')==-1 and f.find('mix')==-1:
        continue
    if f.endswith('show.npy'):
        lst.append(f)

fin = open('tmp3.sh', 'w')
gpu_id = 1
for f in lst:
    if f.find('256.16') != -1:
        idx = 0
        conf = 'configs/20_256.16_HASH_0.json'
    elif f.find('256.32') != -1:
        idx = 1
        conf = 'configs/20_256.32_HASH_0.json'
    for s in [.5, .6, .7, .8, .9]:
        fin.write('echo {}, {}'.format(s, os.path.join(path,f)))
        fin.write('CUDA_VISIBLE_DEVICES={} python get_auc.py {} {} {} {}\n'.format(gpu_id, s, test_file[idx], os.path.join(path,f), conf))
fin.close()
