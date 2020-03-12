import numpy as np
import sys
import json
from hamming_eval import hamming_idxs

conf = np.load(sys.argv[-2])
scores = np.load(sys.argv[-1])


with open(conf) as config_file:
  config = json.load(config_file)

nb_labels = config['num_labels']
model_dir = config['model_dir']
rep = np.load('2_label_permutation.npy')[:nb_labels].T

imgs, nat_labs, input_shape = load_data(config['permutation'], nb_labels)
advs_label = np.load('advs_targeted_labels.npy')

nat_labs = np.zeros(scores.shape).astype(np.float32)
nat_labs[nat_scores>=0.5] = 1.

preds, preds_dist, preds_score = [], [], []

for i in range(len(preds)):
	tmp = np.repeat([nat_labels[i]], rep.shape[0], axis=0)
	dists = np.sum(np.absolute(tmp-rep), axis=1)
	min_dist = np.min(dist)
	pred_labels = np.arange(len(dists))[dists==min_dist]
	pred_scores = [np.sum([scores[i][k] if samples[i][k] == rep[j][k] else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
	pred_label = pred_labels[np.argmax(pred_scores)]
	preds.append(pred_label)
	preds_dist.append(dists[pred_label])
	preds_score.append(np.max(pred_scores))

preds = np.array(preds)
preds_dist = np.array(preds_dist)

print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))

adv_idxs = np.arange(len(labs))[labs != advs_label]
nat_dist, correct_idxs, error_idxs = hamming_idxs(np.load('preds/pred_{}_origin.npy'.format(model_dir)), config)

ts = np.arange(np.max(preds_dist))

for t in ts:
	corr_idx = correct_idxs[nat_dist[correct_idxs]<t]
	idxs = np.intersect1d(adv_idxs, corr_idx)

	print('{} acc: {} / {} = {}'.format(t, np.sum(preds_dist[preds[idxs] == labels[idxs]] < t), len(idxs), np.sum(preds_dist[preds[idxs] == labels[idxs]] < t) / len(idxs)))
	print('{} err: {} / {} = {}'.format(t, np.sum(preds_dist[preds[idxs] != labels[idxs]] < t), len(idxs), np.sum(preds_dist[preds[idxs] != labels[idxs]] < t) / len(idxs)))



