import numpy as np
from sklearn.metrics import roc_auc_score
import sys
from hamming_eval import hamming_idxs
import json
import os

def cal_auc(advs_dist, test_dist, t):
  y_true = np.concatenate((np.ones(advs_dist.shape), np.zeros(test_dist.shape)),axis=-1)
  advs_pred = np.zeros(advs_dist.shape)
  advs_pred[advs_dist>t] = 1
  test_pred = np.zeros(test_dist.shape)
  test_pred[test_dist>t] = 1
  y_pred = np.concatenate((advs_pred, test_pred),axis=-1)
  return roc_auc_score(y_true, y_pred)

def main():
  with open(sys.argv[-1]) as config_file:
    config = json.load(config_file)
  advs_file = sys.argv[-2]
  test_file = sys.argv[-3]
  s = eval(sys.argv[-4])
  label_path = os.path.join('advs',advs_file.split('_HASH_')[-1][:-8]+'label.npy')

  scores = np.load(advs_file)
  advs_dist, _, _ = hamming_idxs(scores,config,s,label_path)
  idxs = np.load('data/final_random_1000_correct_idxs.npy')
  scores = np.load(test_file)[idxs]
  test_dist, _, _ = hamming_idxs(scores,config,s)
  ts = max(np.max(advs_dist), np.max(test_dist))



  y_true = np.concatenate((np.ones(advs_dist.shape), np.zeros(test_dist.shape)),axis=-1)
  print(y_true.shape)
  auc_scores = []
  tprs, fprs = [], []
  for t in range(int(ts)):
    advs_pred = np.zeros(advs_dist.shape)
    advs_pred[advs_dist>t] = 1
    test_pred = np.zeros(test_dist.shape)
    test_pred[test_dist>t] = 1
    y_pred = np.concatenate((advs_pred, test_pred),axis=-1)
    auc_scores.append(roc_auc_score(y_true, y_pred))
  #  print(t, auc_scores[-1])

    nb_advs_advs = np.sum(advs_pred)
    nb_test_advs = np.sum(test_pred)
    nb_advs_test = len(advs_pred) - nb_advs_advs
    nb_test_test = len(test_pred) - nb_test_advs
    nb_pos = len(advs_pred)
    nb_neg = len(test_pred)
    tprs.append(nb_advs_advs/nb_pos)
    fprs.append(nb_test_advs/nb_pos)
  #  print(nb_advs_advs, nb_test_advs, nb_advs_test, nb_test_test, nb_pos, nb_neg)
    # print(t, 'tpr:', , 'fpr:', , 'auc:', roc_auc_score(y_true,y_pred), 'fnr:', nb_advs_test/nb_neg, 'tnr:', nb_test_test/nb_neg)
  print('max auc:', np.argmax(auc_scores), np.max(auc_scores))
  tprs, fprs = np.array(tprs), np.array(fprs)
  for fpr in [.01, .05, .1]:
    idxs = np.arange(len(fprs))[fprs<fpr]
    idx = idxs[np.argmax(tprs[idxs])]
    print('FPR<'+str(fpr)+':', 'fpr:',fprs[idx], 'tpr:',tprs[idx])

if __name__ == '__main__':
  main()