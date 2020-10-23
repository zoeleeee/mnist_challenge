import os
import sys

files = os.listdir('advs')
lst = []
for f in files:
    if f.find('CW')==-1:# and f.find('low')==-1 and f.find('high')==-1 and f.find('mix')==-1:
        continue
    if f.endswith('show.npy'):
        lst.append(f)

print(len(lst), len(lst)/2)
fin = open('tmp5.sh', 'w')
gpu_id = 2
for i, a in enumerate(lst):
#    if gpu_id == 0 and i*2-len(lst) >= 0:
#        fin.close()
#        gpu_id = 1
#        fin = open('tmp2.sh', 'w')
    str1 = 'advs/'+a
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 16, 0))
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 16, 20))
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 16, 40))
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 16, 60))
    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 32, 0))
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 32, 20))
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 32, 40))
#    fin.write('CUDA_VISIBLE_DEVICES={} python keras_multi_eval.py 16 HASH {} configs/20_256.{}_HASH_{}.json\n'.format(gpu_id, str1, 32, 60))
fin.close()


