import keras
import numpy as np
import os
import copy
import time
import math
import hashlib
from Crypto.Util.Padding import pad
from Crypto.Cipher import AES


def permutate_labels(labels, path='2_label_permutation.npy'):
    order = np.load(path)
    labs = [order[i] for i in labels]
    return np.array(labs)

def load_data(path, nb_labels=-1, one_hot=False):
    labels = np.load('data/mnist_labels.npy').astype(np.int)
    if one_hot:
        labels = keras.utils.to_categorical(labels, nb_labels)
    #labs = np.load('mnist_labels.npy')
    #label_permutation = np.load('2_label_permutation.npy')[:int(nb_labels)].T
    #labels = np.array([label_permutation[i] for i in labs])
    order = np.load(path)#'256_65536_permutation.npy'
    if os.path.exists('data/{}_mnist_data.npy'.format(path.split('_')[1])):
        imgs = np.load('data/{}_mnist_data.npy'.format(path.split('_')[1]))
        input_shape = (28, 28, int(path.split('_')[1].split('.')[-1]))
    # if eval(path.split('/')[-1].split('_')[1]) == 65536:
    #   input_shape = (28, 28, 1)
    #   imgs = np.load('65536_mnist_data.npy')
    # elif eval(path.split('/')[-1].split('_')[1]) == 256.2:
    #   imgs = np.load('256_2_mnist_data.npy')
    #   input_shape = (28, 28, 2)
    elif len(order.shape) > 1:
        input_shape = (28, 28, int(path.split('_')[1].split('.')[-1]))
        imgs = np.transpose(np.load('data/mnist_data.npy').astype(np.int), (0,2,3,1))
        #imgs = np.clip(np.transpose(np.load('mnist_data.npy').astype(np.float32)+1, (1,0,2,3))[0], 0, 255).astype(np.int)

        # tmp = np.array([copy.deepcopy(imgs) for i in np.arange(int(path.split('_')[1].split('.')[-1]))])
        samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs])
        imgs = samples.astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
        # np.save('data/{}_mnist_data.npy'.format(path.split('_')[1]),imgs)
    elif len(order.shape) == 1:
        input_shape = (28, 28, 1)
        imgs = np.transpose(np.load('data/mnist_data.npy'), (0,2,3,1)) 
        samples = np.array([[[[order[a] for a in b] for b in c] for c in d] for d in imgs])
        imgs = samples.astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
        # np.save('data/{}_mnist_data.npy'.format(path.split('_')[1]), imgs)
    return imgs, labels, input_shape

def extend_data(path, imgs):
    if np.max(imgs) <= 1:
        imgs *= 255
    order = np.load(path)
    imgs = imgs.astype(np.int)
    samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs]).astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
    return samples

def order_extend_data(order, imgs, basis=255):
    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.astype(np.int)
    samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs]).astype(np.float32) / basis
    return samples

def two_pixel_perm_img(nb_channal, imgs):
    np.random.seed(0)
    perms = []
    for j in range(256):
        perm = []
        for i in range(nb_channal):
            perm.append(np.random.permutation(np.arange(256)))
        perms.append(perm)
    perms = np.array(perms).transpose((0,2,1)).astype(np.float32)/255.

    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.reshape(-1, 784).astype(np.int)
    #print(imgs.shape)
    imgs = np.array([[perms[a[i]][a[i+1]] for i in range(0, len(a), 2)] for a in imgs]).reshape(-1, 28, 14, nb_channal)
    return imgs

def two_pixel_perm(nb_channal, model_dir):
    np.random.seed(0)
    perms = []
    for j in range(256):
        perm = []
        for i in range(nb_channal):
            perm.append(np.random.permutation(np.arange(256)))
        perms.append(perm)

    perms = np.array(perms).transpose((0,2,1)).astype(np.float32)/255.
    # print(perms.shape)

    imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1)).reshape(-1, 784)
    imgs = np.array([[perms[a[i]][a[i+1]] for i in range(0, len(a), 2)] for a in imgs]).reshape(-1, 28, 14, nb_channal)
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir+'_two'

def two_pixel_perm_sliding_img(nb_channal, imgs, seed):
    np.random.seed(seed)
    perms = []
    for j in range(256):
        perm = []
        for i in range(nb_channal):
            perm.append(np.random.permutation(np.arange(256)))
        perms.append(perm)

    perms = np.array(perms).transpose((0,2,1)).astype(np.float32)/255.
    # print(perms.shape)

    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.transpose((3,0,1,2))[0].astype(np.int)
    print(imgs.shape)
    imgs = np.array([[[perms[b[i-1]][b[i]] for i in range(1, len(b), 1)] for b in a] for a in imgs])
    print(imgs.shape)
    return imgs

def two_pixel_perm_sliding(nb_channal, model_dir, seed):
    np.random.seed(seed)
    perms = []
    for j in range(256):
        perm = []
        for i in range(nb_channal):
            perm.append(np.random.permutation(np.arange(256)))
        perms.append(perm)

    perms = np.array(perms).transpose((0,2,1)).astype(np.float32)/255.
    # print(perms.shape)

    imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
    imgs = np.array([[[perms[b[i-1]][b[i]] for i in range(1, len(b), 1)] for b in a] for a in imgs])
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir+'_slide'

def four_pixel_perm_sliding_img(nb_channal, imgs, seed):
    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.transpose((3,0,1,2))[0].astype(np.int)

    if nb_channal ==16:
        m = hashlib.md5
    elif nb_channal == 32:
        m = hashlib.sha256

    new_data = []
    for a in imgs:
        img = []
        for i in range(0, len(a), 1):
            tmp = []
            for j in range(3, len(a[i]), 1):
                b = m((str(seed)+str(a[i][j])+str(a[i][j-1])+str(a[i][j-2])+str(a[i][j-3])).encode('utf-8')).hexdigest()
                tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
            img.append(tmp)
        new_data.append(img)
    imgs = np.array(new_data).astype(np.float32)/255.
    
    return imgs

def four_pixel_perm_sliding_img_AES(nb_channal, imgs, seed, input_bytes):
    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.transpose((3,0,1,2))[0].astype(np.int)

    if nb_channal == 16:
        key = os.urandom(nb_channal)
    new_data = []
    for a in imgs:
        img = []
        for i in range(0, len(a), 1):
            tmp = []
            for j in range(input_bytes-1, len(a[i]), 1):
                string = str(seed)
                for t in range(input_bytes):
                    string += str(a[i][j-t])
                if nb_channal ==16: 
                    meg = pad(string.encode(), nb_channal)
                    b, _ = AES.new(key, AES.MODE_EAX).encrypt_and_digest(meg)
                    tmp.append(list(b))
                elif nb_channal == 32:
                    b = hashlib.sha256(string.encode('utf-8')).hexdigest()
                    tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
            img.append(tmp)
        new_data.append(img)
    imgs = np.array(new_data).astype(np.float32)/255.
    
    return imgs

def four_pixel_perm_sliding(nb_channal, model_dir, seed):
    imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
    if nb_channal ==16:
        m = hashlib.md5
    elif nb_channal == 32:
        m = hashlib.sha256

    new_data = []
    for a in imgs:
        img = []
        for i in range(0, len(a), 1):
            tmp = []
            for j in range(3, len(a[i]), 1):
                b = m((str(seed)+str(a[i-1][j-1])+str(a[i-1][j])+str(a[i][j-1])+str(a[i][j])).encode('utf-8')).hexdigest()
                tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
            img.append(tmp)
        new_data.append(img)
    
    imgs = np.array(new_data).astype(np.float32)/255.
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir+'_slide4'

def four_pixel_perm_sliding_AES(nb_channal, model_dir, seed, input_bytes):
    imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
    if nb_channal == 16:
        key = os.urandom(nb_channal)
    new_data = []
    for a in imgs:
        img = []
        for i in range(0, len(a), 1):
            tmp = []
            for j in range(input_bytes-1, len(a[i]), 1):
                string = str(seed)
                for t in range(input_bytes):
                    string += str(a[i][j-t])
                if nb_channal ==16:
                    meg = pad(string.encode(), nb_channal)
                    b, _ = AES.new(key, AES.MODE_EAX).encrypt_and_digest(meg)
                    tmp.append(list(b))
                elif nb_channal == 32:
                    b = hashlib.sha256(string.encode('utf-8')).hexdigest()
                    tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
                
            img.append(tmp)
        new_data.append(img)
    
    imgs = np.array(new_data).astype(np.float32)/255.
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir+'_slide4'

def window_perm_sliding(nb_channal, model_dir, seed):
    imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
    if nb_channal ==16:
        m = hashlib.md5
    elif nb_channal == 32:
        m = hashlib.sha256

    new_data = []

    for a in imgs:
        img = []
        for i in range(1, len(a), 1):
            tmp = []
            for j in range(1, len(a[i]), 1):
                b = m((str(seed)+str(a[i-1][j-1])+str(a[i-1][j])+str(a[i][j-1])+str(a[i][j])).encode('utf-8')).hexdigest()
                tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
            img.append(tmp)
        new_data.append(img)
    
    imgs = np.array(new_data).astype(np.float32)/255.
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir+'_window'

def window_perm_sliding_AES(nb_channal, model_dir, seed, input_bytes):
    imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
    if nb_channal == 16:
        key = os.urandom(nb_channal)
    new_data = []
    for a in imgs:
        img = []
        for i in range(int(math.sqrt(input_bytes))-1, len(a), 1):
            tmp = []
            for j in range(int(math.sqrt(input_bytes))-1, len(a[i]), 1):
                string = str(seed)
                for t in range(int(math.sqrt(input_bytes))):
                    for l in range(int(math.sqrt(input_bytes))):
                        string += str(a[i-t][j-l])
                if nb_channal ==16:
                    meg = pad(string, nb_channal)
                    b, _ = AES.new(key, AES.MODE_EAX).encrypt_and_digest(meg)
                    tmp.append(list(b))
                elif nb_channal == 32:
                    b = hashlib.sha256(string.encode('utf-8')).hexdigest()
                    tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
                
            img.append(tmp)
        new_data.append(img)
    
    imgs = np.array(new_data).astype(np.float32)/255.
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir+'_window'

def window_perm_sliding_img(nb_channal, imgs, seed):
    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.transpose((3,0,1,2))[0].astype(np.int)
    if nb_channal ==16:
        m = hashlib.md5
    elif nb_channal == 32:
        m = hashlib.sha256

    new_data = []
#    st = time.time()
    for a in imgs:
        img = []
        for i in range(1, len(a), 1):
            tmp = []
            for j in range(1, len(a[i]), 1):
                b = m((str(seed)+str(a[i-1][j-1])+str(a[i-1][j])+str(a[i][j-1])+str(a[i][j])).encode('utf-8')).hexdigest()
                tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
            img.append(tmp)
        new_data.append(img)
 #       print('time:', time.time()-st)
    
    imgs = np.array(new_data).astype(np.float32)/255.
    print('image shape:', imgs.shape)
    return imgs

def window_perm_sliding_img_AES(nb_channal, imgs, seed, input_bytes):
    if np.max(imgs) <= 1:
        imgs *= 255
    imgs = imgs.transpose((3,0,1,2))[0].astype(np.int)
    if nb_channal == 16:
        key = os.urandom(nb_channal)
    new_data = []
    for a in imgs:
        img = []
        for i in range(int(math.sqrt(input_bytes))-1, len(a), 1):
            tmp = []
            for j in range(int(math.sqrt(input_bytes))-1, len(a[i]), 1):
                string = str(seed)
                for t in range(int(math.sqrt(input_bytes))):
                    for l in range(int(math.sqrt(input_bytes))):
                        string += str(a[i-t][j-l])
                if nb_channal ==16:
                    meg = pad(string, nb_channal)
                    b, _ = AES.new(key, AES.MODE_EAX).encrypt_and_digest(meg)
                    tmp.append(list(b))
                elif nb_channal == 32:
                    b = hashlib.sha256(string.encode('utf-8')).hexdigest()
                    tmp.append([int(b[t:t+1], 16) for t in range(0,nb_channal*2,2)])
            img.append(tmp)
        new_data.append(img)
    imgs = np.array(new_data).astype(np.float32)/255.
    return imgs

def diff_perm_per_classifier(st_lab, nb_channal, model_dir):
    np.random.seed(st_lab)
    perm = []
    for i in range(nb_channal):
        perm.append(np.random.permutation(np.arange(256)))
    perm = np.array(perm).transpose((1,0)).astype(np.float32)
    imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
    imgs = order_extend_data(perm, imgs)
    labels = np.load('data/mnist_labels.npy')
    input_shape = imgs.shape
    return imgs, labels, input_shape, model_dir

def diff_perm_per_classifier_img(st_lab, nb_channal, imgs):
    np.random.seed(st_lab)
    perm = []
    for i in range(nb_channal):
        perm.append(np.random.permutation(np.arange(256)))
    perm = np.array(perm).transpose((1,0)).astype(np.float32)
    # imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
    imgs = order_extend_data(perm, imgs)
    return imgs

def show_image(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
    
