import numpy as np
import sys

conf = np.load(sys.argv[-2])
scores = np.load(sys.argv[-1])
with open(conf) as config_file:
  config = json.load(config_file)

nb_labels = config['num_labels']
rep = np.load('2_label_permutation.npy')[:nb_labels].T

imgs, labs, input_shape = load_data(config['permutation'], nb_labels)


nat_labels = np.zeros(scores.shape).astype(np.float32)
nat_labels[nat_scores>=0.5] = 1.

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

ts = np.arange(np.max(preds_dist))
for t in ts:
	print(t, 'acc:', np.sum(preds_dist[preds == labels] < t) / len(labels))
	print(t, 'err:', np.sum(preds_dist[preds != labels] < t) / len(labels))


