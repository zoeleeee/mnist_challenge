import numpy as np
import sys
import json
from hamming_eval import hamming_idxs
from utils import load_data
conf = sys.argv[-2]
scores = np.load(sys.argv[-1])


with open(conf) as config_file:
  config = json.load(config_file)

nb_labels = config['num_labels']
model_dir = config['model_dir']
rep = np.load('2_label_permutation.npy')[:nb_labels].T

imgs, labels, input_shape = load_data(config['permutation'], nb_labels)
labels = labels[60000:]

advs_label = np.load('advs_targeted_labels.npy')

nat_labs = np.zeros(scores.shape).astype(np.float32)
nat_labs[scores>=0.5] = 1.

preds, preds_dist, preds_score = [], [], []

for i in range(len(nat_labs)):
	tmp = np.repeat([nat_labs[i]], rep.shape[0], axis=0)
	dists = np.sum(np.absolute(tmp-rep), axis=1)
	min_dist = np.min(dists)
	pred_labels = np.arange(len(dists))[dists==min_dist]
	pred_scores = [np.sum([scores[i][k] if rep[j][k]==1 else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
	pred_label = pred_labels[np.argmax(pred_scores)]
	preds.append(pred_label)
	preds_dist.append(dists[pred_label])
	preds_score.append(np.max(pred_scores))

preds = np.array(preds)
preds_dist = np.array(preds_dist)

print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))

adv_idxs = np.arange(len(nat_labs))[labels != advs_label]
nat_dist, correct_idxs, error_idxs = hamming_idxs(np.load('preds/pred_{}_origin.npy'.format(model_dir.split('/')[1])), config)

ts = np.arange(np.max(preds_dist))

for t in ts:
	corr_idx = correct_idxs[nat_dist[correct_idxs]<t]
	idxs = np.intersect1d(adv_idxs, corr_idx)

	print('{} advs acc: {} / {} = {:.2f}%'.format(t, np.sum(preds_dist[idxs[preds[idxs] == advs_label[idxs]]] < t), len(idxs), np.sum(preds_dist[idxs[preds[idxs] == advs_label[idxs]]] < t) / len(idxs)*100))
	print('{} acc: {} / {} = {:.2f}%'.format(t, np.sum(preds_dist[idxs[preds[idxs] == labels[idxs]]] < t), len(idxs), np.sum(preds_dist[idxs[preds[idxs] == labels[idxs]]] < t) / len(idxs)*100))
