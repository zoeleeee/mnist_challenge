import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image

file = sys.argv[-1]
label = int(sys.argv[-2])
idx = file.split('_')[-2]
imgs = np.load(file)
for i, img in enumerate(imgs):
    #print(np.max(img), np.min(img))
    #tmp = np.expand_dims((img*255).astype(np.uint8), axis=0)
   # print(np.max(tmp), np.min(tmp), tmp.shape)
 #@   im = Image.fromarray(tmp)
#    im.save('mnist_pics/pil#{}_{}.png'.format(idx,(label+i)%10))
    plt.cla()
#    img = (img*255).astype(np.int)
    print(img.shape)
    plt.imshow(img.reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.savefig("mnist_pics/fig#{}_{}.png".format(idx, (label+i)%10), format='png')

