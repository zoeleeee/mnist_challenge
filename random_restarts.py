#python random_restarts.py linf MIM low at
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import json
from hamming_eval import hamming_idxs

model_id = sys.argv[-1]
_type = sys.argv[-2]
attack = sys.argv[-3]
metric = sys.argv[-4]
sign = sys.argv[-5]

files = os.listdir('advs')
lst = []
for f in files:
    if not f.endswith('show.npy'):
        continue
    if f.find(attack) != -1 and f.find(_type) != -1 and f.find(metric) != -1:
        lst.append(f)
print(lst)
if model_id.find('config') != -1:
    
    with open(model_id) as config_file:
        config = json.load(config_file)
    model_dir = config['model_dir']
    
    for s in [.5, .6, .7, .8, .9]:
        valid = 10
        num = 0
        res = np.zeros((10, 1000))
        for f in lst:
            label_path = os.path.join('advs', f[:-8]+'label.npy')
            idxs = np.load(os.path.join('advs', f[:-8]+'idxs.npy')).astype(np.int)
            scores = np.load('preds/pred_{}_{}'.format(model_dir.split('/')[1]+'_'+sign, f.split('/')[-1]))
            pred_dist, correct_idxs, error_idxs = hamming_idxs(scores, config, s, label_path)
    #        print(np.sum(res), len(correct_idxs), len(error_idxs))
    #        print(len(idxs), error_idxs)
            valid = min(10, np.max(pred_dist))
            for t in range(valid):
                res[t][idxs[error_idxs[pred_dist[error_idxs] <= t]]] = 2
        #        print(np.sum(res))
                if num == 0:
                    res[t][idxs[correct_idxs[pred_dist[correct_idxs] <= t]]] = 1
                else:
                    correct_tmp = np.zeros(1000)
                    correct_tmp[idxs[correct_idxs[pred_dist[correct_idxs] <= t]]]  = 1
                    if len(scores) == res.shape[-1]:
                        res[t] = np.array([2 if v == 2 else np.logical_and(v, correct_tmp[i]) for i,v in enumerate(res)])
                    else:
                        res[t][idxs[correct_idxs[pred_dist[correct_idxs] <= t]]] = 1
                num += 1
    #        print(np.sum(res))
        
        for i in  range(valid):
            print('{}_{}: {}_{}_{}_{} acc: {} / err: {} / dec: {}'.format(s, i, model_id, _type, attack, metric, np.sum(res[i]==1), np.sum(res[i]==2), np.sum(res[i]==0)))

else:
    res = np.ones(1000)

    if model_id == 'at':
        if metric == 'linf':
            model = torch.load('cifar10_linf_at.pth')
        elif metric == 'l2':
            model = torch.load('cifar10_l2_at.pth')
    elif model_id == 'nat':
        model = torch.load('cifar10_nat.pth')
    model.eval()
    for path in lst:
        print(path)
        path = os.path.join('data', path)
        x_test = np.load(path).astype(np.float32)
        y_test = np.load(path[:-8]+'label.npy')
        idx_test = np.load(path[:-8]+'idxs.npy')
        x_test /= 255.
#        print(x_test.shape, y_test.shape, idx_test.shape, np.max(idx_test), np.min(idx_test), np.max(x_test))

        x_test = torch.Tensor(x_test)
        test_dataset = TensorDataset(x_test, torch.Tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=100, num_workers=0, shuffle=False)
        
        pred_labs = []
        for im, label in test_loader:
            _, preds = torch.max(model(im.cuda())[0], dim=1)
            pred_labs.append(preds.cpu().numpy())
        pred_labs = np.hstack(pred_labs)
        print(pred_labs.shape)
#        print(np.sum(pred_labs-y_test))
        res[idx_test] = np.logical_and(pred_labs==y_test, res[idx_test])

    print('{}_{}_{}_{} acc: {}'.format(model_id, _type, attack, metric, np.mean(res.astype(np.float32))))
