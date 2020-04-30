import time
import random 
#import minpy.numpy as np
from tensorflow import keras
from utils import extend_data
import sys
import json
import numpy as np

rep_labels = np.load('2_label_permutation.npy')
conf = sys.argv[-1]
with open(conf) as config_file:
    config = json.load(config_file)

def predict(models, img, t=0):
    img = np.clip(img, 0, 1)*255
    img = extend_data(config['permutation'], np.array([img]))
    scores = np.hstack([m.predict(img) for m in models])[0]
    #print(scores.shape)

    nat_labels = np.zeros(scores.shape).astype(np.float32)
    nat_labels[scores>=0.5] = 1.
    rep = rep_labels[:len(scores)].T
    tmp = np.repeat([nat_labels], rep.shape[0], axis=0)
    dists = np.sum(np.absolute(tmp-rep), axis=-1)
    min_dist = np.min(dists)
    pred_labels = np.arange(len(dists))[dists==min_dist]
    pred_scores = [np.sum([scores[k] if rep[j][k]==1 else 1-scores[k] for k in np.arange(len(scores))]) for j in pred_labels]
    pred_label = pred_labels[np.argmax(pred_scores)]
    if min_dist <= 0:
        return pred_label
    else:
        return -1

def attack_targeted(model, train_dataset, x0, y0, target, alpha = 0.1, beta = 0.001, iterations = 1000):
    # STEP I: find initial direction (theta, g_theta)
    num_samples = 100
    best_theta, g_theta = None, float('inf')
    query_count = 0
    print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    samples = np.random.permutation(train_dataset)[:num_samples]
    for xi in samples:
        if predict(model, xi) == target:
            theta = xi - x0
            initial_lbd = np.linalg.norm(theta)
            theta = theta/np.linalg.norm(theta)
            lbd, count = fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    # STEP II: seach for optimal
    timestart = time.time()

    g1 = 1.0
    theta, g2 = best_theta.copy(), g_theta
    opt_count = 0

    for i in range(iterations):
        print(i)
        gradient = np.zeros(theta.shape)
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = np.random.normal(theta.shape).astype(np.float32)
            u = u/np.linalg.norm(u)
            ttt = theta+beta * u
            ttt = ttt/np.linalg.norm(ttt)
            g1, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, np.linalg.norm(g2*theta), opt_count))

        min_theta = theta
        min_g2 = g2
    
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/np.linalg.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta/np.linalg.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.copy(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = predict(model, x0 + g_theta*best_theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
    return x0 + g_theta*best_theta

def fine_grained_binary_search_local_targeted(model, x0, y0, t, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
   
    if predict(model,x0+lbd*theta) != t:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while predict(model,x0+lbd_hi*theta) != t:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 100: 
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while predict(model,x0+lbd_lo*theta) == t:
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict(model,x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search_targeted(model, x0, y0, t, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd

    while predict(model,x0 + lbd*theta) != t:
        lbd *= 1.05
        nquery += 1
        if lbd > 100: 
            return float('inf'), nquery

    num_intervals = 100

    lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        nquery += 1
        if predict(model,x0 + lbd*theta) == t:
            lbd_hi = lbd
            lbd_hi_index = i
            break

    lbd_lo = lambdas[lbd_hi_index - 1]

    while (lbd_hi - lbd_lo) > 1e-7:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict(model,x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi, nquery

def attack_mnist(model, alpha=0.2, beta=0.001, isTarget= False, num_attacks= 10):
    imgs = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32)/255.
    labs = np.load('data/mnist_labels.npy')[60000:]
    nb_labs = np.max(labs)
    
    print("\n\n Running {} attack on {} random  MNIST test images for alpha= {} beta= {}\n\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta))
    
    total_distortion = []
    samples = []
    for i in range(nb_labs+1):
        samples.append(np.random.permutation(np.arange(len(labs))[labs==i])[0])
    
    # samples = [6312, 6891, 4243, 8377, 7962, 6635, 4970, 7809, 5867, 9559, 3579, 8269, 2282, 4618, 2290, 1554, 4105, 9862, 2408, 5082, 1619, 1209, 5410, 7736, 9172, 1650, 5181, 3351, 9053, 7816, 7254, 8542, 4268, 1021, 8990, 231, 1529, 6535, 19, 8087, 5459, 3997, 5329, 1032, 3131, 9299, 3910, 2335, 8897, 7340, 1495, 5244,8323, 8017, 1787, 4939, 9032, 4770, 2045, 8970, 5452, 8853, 3330, 9883, 8966, 9628, 4713, 7291, 9770, 6307, 5195, 9432, 3967, 4757, 3013, 3103, 3060, 541, 4261, 7808, 1132, 1472, 2134, 634, 1315, 8858, 6411, 8595, 4516, 8550, 3859, 3526]
    #true_labels = [3, 1, 6, 6, 9, 2, 7, 5, 5, 3, 3, 4, 5, 6, 7, 9, 1, 6, 3, 4, 0, 6, 5, 9, 7, 0, 3, 1, 6, 6, 9, 6, 4, 7, 6, 3, 4, 3, 4, 3, 0, 7, 3, 5, 3, 9, 3, 1, 9, 1, 3, 0, 2, 9, 9, 2, 2, 3, 3, 3, 0, 5, 2, 5, 2, 7, 2, 2, 5, 7, 4, 9, 9, 0, 0, 7, 9, 4, 5, 5, 2, 3, 5, 9, 3, 0, 9, 0, 1, 2, 9, 9]
    for idx in samples:
        #idx = random.randint(100, len(test_dataset)-1)
        image, label = imgs[idx], labs[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        print("Original label: ", label)
        lab = predict(model, image)
        print("Predicted label: ", lab)
        if lab != label:
            print('CHANGE IMAGES#{}: prediction of original image is not the same with true label'.format(i))
            continue
        #target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 10)))
        advs = [image]
        for i in range(nb_labs):
            target = (label + i) % (nb_labs+1)
            adv = attack_targeted(model, imgs[labs==target], image, label, target, alpha = alpha, beta = beta, iterations = 1000)
            print(i, "Predicted label for adversarial example: ", predict(model, adversarial))
            advs.append(np.clip(adv, 0, 1))
            total_distortion.append(np.linalg.norm(adv.reshape(-1) - image.reshape(-1)))
        np.save('advs/opt_attacks_{}_show.npy'.format(idx), advs)

    print("Average distortion on random {} images is {}".format(len(total_distortion), np.mean(total_distortion)))




if __name__ == '__main__':
    timestart = time.time()
    random.seed(0)
        
    def custom_loss():
        def loss(y_true, y_pred):
            if config['loss_func'] == 'bce':
                _loss = keras.losses.BinaryCrossentropy()
                return _loss(y_true, tf.nn.sigmoid(y_pred))
            elif config['loss_func'] == 'xent':
                _loss = keras.losses.SparseCategoricalCrossentropy()
                return _loss(y_true, tf.nn.softmax(y_pred))
        return loss
    model = keras.models.load_model(config['model_dir']+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)


    attack_mnist([model], alpha=2, beta=.005, isTarget=True)
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
