import numpy as np 

for i in range(60):
	print(i)
	perms = []
	for j in range(256*256*256):
		perms.append(np.random.permutation(np.arange(256)))
	np.save('perm/{}.npy'.format(i), perms)

