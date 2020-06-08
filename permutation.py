import numpy as np

imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
new_data = []
for t in range(16):
    print(t)
    np.random.seed(t)
    perms = []
    for j in range(256*256*256):
        perms.append(np.random.permutation(np.arange(256)))
    tmp = np.array([[[perms[a[i-1][j-1]][a[i-1][j]][a[i][j-1]][a[i][j]] for i in range(1, len(a[j]), 1)] for j in range(1, len(a), 1)] for a in imgs])
    print(tmp.size())
    new_data.append(tmp)
np.save('data/window_mnist_data.npy', np.arrray(new_data).transpose((1,2,3,0)))


