import sys
import numpy as np
file = sys.argv[-1]

with open(file) as f:
	cnt = f.readlines()

count = []
distortion = []
calibration = []
linf = []
for line in cnt:
	if line.startswith('Adversarial Example Found Successfully:'):
		count.append(int(line.split(' ')[-1]))
		distortion.append(eval(line.split(' ')[-5]))
	elif line.startswith('1 Predicted label'):
		calibration.append(eval(line.split(' ')[-3]))
		linf.append(eval(line.split(' ')[-2]))
print('count:', np.mean(count), np.median(count), np.min(count), np.max(count))
print('distortion:', np.mean(distortion), np.median(distortion), np.min(distortion), np.max(distortion))
print('calibration:', np.mean(calibration), np.median(calibration), np.min(calibration), np.max(calibration))
print('linf:', np.mean(linf), np.median(linf), np.min(linf), np.max(linf))
