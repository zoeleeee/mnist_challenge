import numpy as np
import sys
import json
from scipy.special import expit

from utils import load_data

def hamming_idxs(scores, config, _type, t):
	res = []
	#nb_labels = config['num_labels']
	rep = np.load('2_label_permutation.npy')[config['start_label']:config['start_label']+scores.shape[-1]].T
	#tmp = np.load('2_label_permutation.npy')[:config['num_labels']].T
	#np.random.seed(0)
	#rep = np.random.permutation(tmp)
	#while rep.shape[-1] < scores.shape[-1]:
	#	np.random.seed(rep.shape[-1])
	#	rep = np.hstack((rep, np.random.permutation(tmp)))

	imgs, labels, input_shape = load_data(config['permutation'], scores.shape[-1])
	labels = labels[60000:60000+len(scores)]
	print(t)
	nat_labels = np.zeros(scores.shape).astype(np.float32)
	nat_labels[scores>=t] = 1.
	if t == 1-t:
		nat_labels[scores<t] = -1.
	else:
		nat_labels[scores<=1-t] = -1.
	rep[rep==0] = -1
#        print(nat_labels[0])
#        print(rep[labels[0]])
	preds, preds_dist, preds_score = [], [], []

	for i in range(len(nat_labels)):
		tmp = np.repeat([nat_labels[i]], rep.shape[0], axis=0)
		dists = np.sum(np.absolute(tmp-rep), axis=-1)
		min_dist = np.min(dists)
		pred_labels = np.arange(len(dists))[dists==min_dist]
		pred_scores = [np.sum([scores[i][k] if rep[j][k]==1 else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
		pred_label = pred_labels[np.argmax(pred_scores)]
		preds.append(pred_label)
		preds_dist.append(dists[pred_label])
		preds_score.append(np.max(pred_scores))

	preds = np.array(preds)
	preds_dist = np.array(preds_dist)
	print(preds)
	correct_idxs = np.arange(len(preds))[preds == labels]
	error_idxs = np.arange(len(preds))[preds != labels]
	return preds_dist, correct_idxs, error_idxs

if __name__ == '__main__':
	with open(sys.argv[-1]) as config_file:
	  config = json.load(config_file)
	name = sys.argv[-2].split('/')[-1][5:]
	_type = sys.argv[-3]
	t = eval(sys.argv[-4])
	#model_dir = config['model_dir']
	scores = np.load('preds/pred_{}'.format(name))
#	if np.max(scores) > 1:
#		scores = expit(scores)
	# labels = np.load('preds/labels_{}'.format(name))
	print(scores.shape)
	preds_dist, correct_idxs, error_idxs = hamming_idxs(scores, config, _type, t)
	print(preds_dist.shape)
	print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))

	ts = np.arange(np.max(preds_dist)+1)
	for t in ts:
		print(t, 'acc:', np.sum(preds_dist[correct_idxs] < t+1) / len(scores))
		print(t, 'err:', np.sum(preds_dist[error_idxs] < t+1) / len(scores))




