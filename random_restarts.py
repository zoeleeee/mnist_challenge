#python random_restarts.py linf MIM low at
from tensorflow import keras
import numpy as np
import sys
import os
import json
from hamming_eval import hamming_idxs

def cal_auc(advs, test_dist, t):
    advs_pred = np.zeros(advs_pred.shape)
    advs_pred[advs==0] = 1
    test_pred = np.zeros(test_dist.shape)
    test_pred[test_dist>t] = 1
    y_pred = np.concatenate((advs_pred, test_pred),axis=-1)
  return roc_auc_score(y_true, y_pred)

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
    
    
    test_file = 'preds/pred_{}_{}'.format(model_dir.split('/')[1]+'_'+sign, 'origin.npy')
    idxs = np.load('data/final_random_1000_correct_idxs.npy')
    scores = np.load(test_file)[idxs]
    for s in [.5, .6, .7, .8, .9]:
        test_dist, _, _ = hamming_idxs(scores,config,s)
        valid = 10
        num = 0
        res = np.zeros((10, 1000))
        tot_idxs = []
        for f in lst:
            label_path = os.path.join('advs', f[:-8]+'label.npy')
            idxs = np.load(os.path.join('advs', f[:-8]+'idxs.npy')).astype(np.int)
            scores = np.load('preds/pred_{}_{}'.format(model_dir.split('/')[1]+'_'+sign, f.split('/')[-1]))
            pred_dist, correct_idxs, error_idxs = hamming_idxs(scores, config, s, label_path)
            if len(tot_idxs) == 0:
                tot_idxs = idxs
            else:
                tot_idxs = np.union1d(tot_idxs, idxs)
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
                    if attack == 'CW' and _type=='mix':
                        res[t][idxs[correct_idxs[pred_dist[correct_idxs] <= t]]] = 1
                    else:
                        res[t] = np.array([2 if v == 2 else np.logical_and(v, correct_tmp[i]) for i,v in enumerate(res[t])])
            num += 1
    #        print(np.sum(res))
        
        for i in  range(valid):
            acc = np.sum(res[i]==1)/len(tot_idxs)
            err = np.sum(res[i]==2)/len(tot_idxs)
            det = 1-acc-err
            print('{}_{}: {}_{}_{}_{} auc:{} / acc: {} / err: {} / dec: {}'.format(s, i, model_id, _type, attack, metric, cal_auc(res[i], test_dist, i), acc, err, det))

else:
    res = np.ones(1000)

    if model_id == 'at':#linf
        model = keras.models.load_model('models/adv_trained.h5')
    elif model_id == 'nat':
        model = keras.models.load_model('models/natural.h5')
    idxs = []
    for path in lst:
        print(path)
        path = os.path.join('advs', path)
        x_test = np.load(path).astype(np.float32)
        y_test = np.load(path[:-8]+'label.npy')
        idx_test = np.load(path[:-8]+'idxs.npy')
        x_test /= 255.
        idxs = idx_test if len(idxs) == 0 else np.union1d(idxs, idx_test)

        output = model.predict(x_test, batch_size=100)
        pred_labs = np.argmax(output, axis=-1)
        res[idx_test] = np.logical_and(pred_labs==y_test, res[idx_test])

    print('{}_{}_{}_{} acc: {}'.format(model_id, _type, attack, metric, np.mean(res.astype(np.float32))))
