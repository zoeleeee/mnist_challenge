import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
#from models import IMAGENET, MNIST, CIFAR10, load_imagenet_data, load_mnist_data, load_cifar10_data, load_model, show_image
import sys
import json
from utils import *
rep_labels = np.load('2_label_permutation.npy')
rep_labels[rep_labels==0] = -1
conf = sys.argv[-1]
nb_models = int(sys.argv[-2])
_type = sys.argv[-3]
input_bytes = eval(sys.argv[-4])
with open(conf) as config_file:
    config = json.load(config_file)
st_lab = config['start_label']
nb_channal = int(config['permutation'].split('_')[1].split('.')[1])
nb_label = config['num_labels']
def predict(models, img, t=0):
    nb_channel = int(config['permutation'].split('_')[1].split('.')[1])
    img = torch.clamp(torch.tensor(img), 0, 1)*255
#    print(torch.max(img), torch.min(img))

    if _type == 'slide':
        imgs = []
        #print(img.shape)
        for i in range(len(models)):
            tmp = two_pixel_perm_sliding_img(nb_channel, np.array([img.numpy()]), i*nb_label).transpose((0,3,1,2))
            imgs.append(torch.tensor(tmp).cuda())
        #img = torch.tensor(two_pixel_perm_sliding_img(nb_channel, np.array([img.numpy()])).transpose((0,3,1,2))).cuda()
        scores = torch.cat(tuple([torch.sigmoid(model(imgs[i])) for i,model in enumerate(models)]), dim=1)
        # scores = torch.cat(tuple([torch.sigmoid(model(img)) for i,model in enumerate(models)]), dim=1)
    elif _type == 'normal':
        img = torch.tensor(extend_data(config['permutation'], np.array([img.numpy()])).transpose((0,3,1,2))).cuda()
        scores = torch.cat(tuple([torch.sigmoid(model(img)) for i,model in enumerate(models)]), dim=1)
    elif _type == 'diff':
        imgs = []
        for i in range(len(models)):
            tmp = diff_perm_per_classifier_img(i*nb_label, nb_channel, np.array([img.numpy()])).transpose((0,3,1,2))
            imgs.append(torch.tensor(tmp).cuda())
        scores = torch.cat(tuple([torch.sigmoid(model(imgs[i])) for i,model in enumerate(models)]), dim=1)
    elif _type == 'window':
        imgs = []
        for i in range(len(models)):
            tmp = window_perm_sliding_img_AES(nb_channel, np.array([img.numpy()]), i*nb_label, input_bytes).transpose((0,3,1,2))
            imgs.append(torch.tensor(tmp).cuda())
        scores = torch.cat(tuple([torch.sigmoid(model(imgs[i])) for i,model in enumerate(models)]), dim=1)
    elif _type == 'slide4':
        imgs = []
        for i in range(len(models)):
            tmp = four_pixel_perm_sliding_img_AES(nb_channel, np.array([img.numpy()]), i*nb_label, input_bytes).transpose((0,3,1,2))
            imgs.append(torch.tensor(tmp).cuda())
        scores = torch.cat(tuple([torch.sigmoid(model(imgs[i])) for i,model in enumerate(models)]), dim=1)
  #  print(scores)
 #   print(scores.size())
    nat_labels = torch.zeros(scores.shape).type(torch.FloatTensor)
#<<<<<<< HEAD
#    nat_labels[scores>=0.5] = 1.
#=======
    nat_labels[scores>=0.9] = 1.
    nat_labels[scores<=0.1] = -1.
#>>>>>>> refs/remotes/origin/master
    #print(nat_labels, rep[3])
    rep = torch.tensor(rep_labels[:scores.size()[1]].T)
    tmp = nat_labels.repeat(rep.size()[0], 1)
    #print(nat_labels, rep[3])
    dists = (tmp-rep).abs().sum(dim=-1)
   # print(dists)
    min_dist = dists.min()
    if min_dist > t:
        return -1
    pred_labels = torch.arange(len(dists))[dists==min_dist]
    pred_scores = torch.tensor([torch.sum(torch.tensor([scores[0][k] if rep[j][k]==1 else 1-scores[0][k] for k in np.arange(scores.size()[1])])) for j in pred_labels])
    pred_label = pred_labels[torch.argmax(pred_scores)]
    
    return pred_label

def attack_targeted(model, train_dataset, x0, y0, target, alpha = 0.1, beta = 0.001, iterations = 1000):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
    """

    if (predict(model,x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0

    # STEP I: find initial direction (theta, g_theta)

    num_samples = 100
    best_theta, g_theta = None, float('inf')
    query_count = 0

    print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, xi in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if predict(model,xi) == target:
            theta = torch.tensor(xi) - x0
            initial_lbd = torch.norm(theta)
            #print(theta, torch.norm(theta))
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)
    if best_theta is None:
        print('===========> Give UP fail to find best distortion in initial %d samples' % (num_samples))
        return 
    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    # return x0+g_theta*best_theta

    # STEP II: seach for optimal
    timestart = time.time()

    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    opt_count = 0

    for i in range(iterations):
        gradient = torch.zeros(theta.size())
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))

        min_theta = theta
        min_g2 = g2
    
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
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
                new_theta = new_theta/torch.norm(new_theta)
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
            best_theta, g_theta = theta.clone(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = predict(model,x0 + g_theta*best_theta)
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



def attack_untargeted(model, train_dataset, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """

    if (predict(model,x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0

    num_samples = 1000
    best_theta, g_theta = None, float('inf')
    query_count = 0

    print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, (xi, yi) in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if predict(model,xi) != y0:
            theta = xi - x0
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))

    
    
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    torch.manual_seed(0)
    opt_count = 0
    stopping = 0.01
    prev_obj = 100000
    for i in range(iterations):
        gradient = torch.zeros(theta.size())
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
            if g2 > prev_obj-stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2
    
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
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
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
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
            best_theta, g_theta = theta.clone(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = predict(model,x0 + g_theta*best_theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
    return x0 + g_theta*best_theta

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
     
    if predict(model,x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while predict(model,x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while predict(model,x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict(model,x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best: 
        if predict(model,x0+current_best*theta) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd
    
    ## original version
    #lbd = initial_lbd
    #while predict(model,x0 + lbd*theta) == y0:
    #    lbd *= 2
    #    nquery += 1
    #    if lbd > 100:
    #        return float('inf'), nquery
    
    #num_intervals = 100

    # lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    # lbd_hi = lbd
    # lbd_hi_index = 0
    # for i, lbd in enumerate(lambdas):
    #     nquery += 1
    #     if predict(model,x0 + lbd*theta) != y0:
    #         lbd_hi = lbd
    #         lbd_hi_index = i
    #         break

    # lbd_lo = lambdas[lbd_hi_index - 1]
    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict(model,x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def attack_mnist(nets, alpha=0.2, beta=0.001, isTarget= False, num_attacks= 100):
    aux_imgs = np.load('data/mnist_data.npy')[:60000].transpose((0,2,3,1)).astype(np.float32)/255.
    imgs = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32)/255.
    labs = np.load('data/mnist_labels.npy')[60000:]
    nb_labs = np.max(labs)
    imgs = torch.tensor(imgs)

    model = []
    for net in nets:
        if torch.cuda.is_available():
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=[0])
        net.eval()
        if torch.cuda.is_available():
            model.append(net.module)
        else:
            model.append(net)
    #load_model(net, 'models/mnist_cpu.pt')
    

    print("\n\n Running {} attack on {} random  MNIST test images for alpha= {} beta= {}\n\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta))
    total_distortion = []
    # samples = []
    # for i in range(nb_labs+1):
        # samples.append(np.random.permutation(np.arange(len(labs))[labs==i])[0])
    labs = torch.tensor(labs)
    samples = [4824, 9591, 775, 6453, 7759, 766, 4037, 9869, 936, 5262, 6312, 6891, 4243, 8377, 7962, 6635, 4970, 7809, 5867, 9559, 3579, 8269, 2282, 4618, 2290, 1554, 4105, 9862, 2408, 5082, 1619, 1209, 5410, 7736, 9172, 1650, 5181, 3351, 9053, 7816, 7254, 8542, 4268, 1021, 8990, 231, 1529, 6535, 19, 8087, 5459, 3997, 5329, 1032, 3131, 9299, 3910, 2335, 8897, 7340, 1495, 5244,8323, 8017, 1787, 4939, 9032, 4770, 2045, 8970, 5452, 8853, 3330, 9883, 8966, 9628, 4713, 7291, 9770, 6307, 5195, 9432, 3967, 4757, 3013, 3103, 3060, 541, 4261, 7808, 1132, 1472, 2134, 634, 1315, 8858, 6411, 8595, 4516, 8550, 3859, 3526]
    #true_labels = [3, 1, 6, 6, 9, 2, 7, 5, 5, 3, 3, 4, 5, 6, 7, 9, 1, 6, 3, 4, 0, 6, 5, 9, 7, 0, 3, 1, 6, 6, 9, 6, 4, 7, 6, 3, 4, 3, 4, 3, 0, 7, 3, 5, 3, 9, 3, 1, 9, 1, 3, 0, 2, 9, 9, 2, 2, 3, 3, 3, 0, 5, 2, 5, 2, 7, 2, 2, 5, 7, 4, 9, 9, 0, 0, 7, 9, 4, 5, 5, 2, 3, 5, 9, 3, 0, 9, 0, 1, 2, 9, 9]
    for idx in samples:
        #idx = random.randint(100, len(test_dataset)-1)
        image, label = torch.tensor(imgs[idx]), labs[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        show_image(image.numpy())
        print("Original label: ", label)
        lab = predict(model, image)
        print("Predicted label: ", lab)
        if lab != label:
            print('CHANGE IMAGES#{}: prediction of original image is not the same with true label'.format(idx))
            continue
        advs = [image.numpy()]
        for i in range(1,2):#nb_labs+1):
            target = (label + i) % (nb_labs+1)
            adv = attack_targeted(model, aux_imgs, image, label, target, alpha = alpha, beta = beta, iterations = 1)
            if adv is None:
                continue
            tmp = (adv.numpy()*255.).astype(np.int).astype(np.float32)/255.
            show_image((adv.numpy()*255.).astype(np.int).astype(np.float32)/255.)
            print(i, "Predicted label for adversarial example: ", predict(model, adv), np.linalg.norm(tmp-image.numpy()), np.linalg.norm(tmp.reshape(-1)-image.numpy().reshape(-1), axis=-1, ord=np.inf), np.sum(tmp!=image.numpy()))
            advs.append(torch.clamp(adv, 0, 1).numpy())
            total_distortion.append(torch.norm(adv - image))
        np.save('advs/opt_attacks_{}_show.npy'.format(idx), advs)

    print("Average distortion on random {} images is {}".format(len(total_distortion), np.mean(total_distortion)))
    print('image distortion beyond 1.0 number:', np.sum(np.array(total_distortion) > 1.))

def attack_cifar10(alpha= 0.2, beta= 0.001, isTarget= False, num_attacks= 100):
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    dataset = train_dataset
    print("Length of test_set: ", len(test_dataset))
    net = CIFAR10()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    load_model(net, 'models/cifar10_gpu.pt')
    #load_model(net, 'models/cifar10_cpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    def single_attack(image, label, target = None):
        print("Original label: ", label)
        print("Predicted label: ", predict(model,image))
        if target == None:
            adversarial = attack_untargeted(model, dataset, image, label, alpha = alpha, beta = beta, iterations = 1000)
        else:
            print("Targeted attack: %d" % target)
            adversarial = attack_targeted(model, dataset, image, label, target, alpha = alpha, beta = beta, iterations = 1000)
        print("Predicted label for adversarial example: ", predict(model,adversarial))
        return torch.norm(adversarial - image)

    print("\n\nRunning {} attack on {} random CIFAR10 test images for alpha= {} beta= {}\n\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta))
    total_distortion = 0.0

    samples = [6311, 6890, 663, 4242, 8376, 7961, 6634, 4969, 7808, 5866, 9558, 3578, 8268, 2281, 2289, 1553, 4104, 8725, 9861, 2407, 5081, 1618, 1208, 5409, 7735, 9171, 1649, 5796, 7113, 5180, 3350,9052, 7253, 8541, 4267, 1020, 8989, 230, 1528, 6534, 18, 8086, 3996, 1031, 3130, 9298, 3632, 3909, 2334, 8896, 7339, 1494, 5243, 8322, 8016, 1786, 9031, 4769, 8969, 5451, 8852, 3329, 9882, 8965, 9627, 4712, 7290, 9769, 6306, 5194, 3966, 4756, 3012, 3102, 540, 4260, 7807, 1471, 2133, 2450, 633, 1314, 8857, 6410, 8594, 4515, 8549, 3858, 3525, 6411, 4360, 7753, 7413, 684,3343, 6785, 7079, 2263] 
    #true_labels = [3, 5, 6, 8, 7, 3, 4, 1, 8, 4, 0, 7, 5, 5, 1, 4, 0, 8, 6, 9, 5, 7, 3, 1, 4, 2, 5, 5, 9, 9, 8, 0, 4, 8, 7, 1, 4, 5, 2, 7, 8, 4, 6, 3, 3, 1, 1, 5, 1, 8, 6, 7, 1, 4, 4, 1, 0, 8, 8, 6, 7, 3, 1, 4, 4, 4, 6, 8, 0, 7, 4, 6, 1, 0, 1, 8, 3, 8, 3, 1, 8, 9, 0, 1, 3, 0, 1, 8, 2, 8, 6, 9, 1, 9, 3, 6, 7, 6]
    #samples = [7753, 1314, 633]
    #samples = [6311]
    for idx in samples:
        #idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        #target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 10)))
        target = None if not isTarget else (1+label) % 10
        total_distortion += single_attack(image, label, target)
    print("Average distortion on random {} images is {}".format(num_attacks, total_distortion/num_attacks))

def attack_imagenet(arch='resnet50', alpha=0.2, beta= 0.001, isTarget=False, num_attacks = 100):
    train_loader, test_loader, train_dataset, test_dataset = load_imagenet_data()
    dataset = test_dataset
    print("Length of test_set: ", len(test_dataset))

    model = IMAGENET(arch)

    def attack_single(image, label, target = None):
        print("Original label: ", label)
        print("Predicted label: ", predict(model,image))
        if target == None:
            adversarial = attack_untargeted(model, dataset, image, label, alpha = alpha, beta = beta, iterations = 1500)
        else:
            print("Targeted attack: %d" % target)
            adversarial = attack_targeted(model, dataset, image, label, target, alpha = alpha, beta = beta, iterations = 1500)
        print("Predicted label for adversarial example: ", predict(model,adversarial))
        return torch.norm(adversarial - image)

    print("\nRunning {} attack on {} random IMAGENET test images for alpha= {} beta= {} using {}\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta, arch))
    total_distortion = 0.0

    samples = [25248, 27563, 2654, 16969, 31846, 26538, 19878, 14316, 33076, 9128, 9159, 49533, 34903, 46215, 963220326, 6473, 483344826,216406600, 23187, 40036, 41971, 13401, 36211, 31262, 4082, 35960, 6113, 47167, 46548, 75, 40102, 32348, 21313, 46114, 4128,37193, 14530, 9339, 5978, 20976, 33289]
    for idx in samples:
        #idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n======== Image %d =========" % idx)
        target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 1000)))
        total_distortion += attack_single(image, label, target)
    
    print("Average distortion on random {} images is {}".format(num_attacks, total_distortion/num_attacks))

if __name__ == '__main__':
    timestart = time.time()
    random.seed(0)

    
    nets = []
    import imp
    MainModel = imp.load_source('MainModel', 'models/256.py')
    for i in range(nb_models):
        with open(conf) as config_file:
            config = json.load(config_file)
        
        #idxs = np.arange(len(labels))
        net = torch.load(config['model_dir']+'_window16.pt')
        conf = conf[:conf.find(conf.split('_')[-1])]+str(config['num_labels']*(i+1))+'.json'
        nets.append(net)
    
    
    

    attack_mnist(nets, alpha=2, beta=0.005, isTarget= False)
    # attack_cifar10(alpha=5, beta=0.001, isTarget= False)
    #attack_imagenet(arch='resnet50', alpha=10, beta=0.005, isTarget= False)
    #attack_imagenet(arch='vgg19', alpha=0.05, beta=0.001, isTarget= False, num_attacks= 10)

    #attack_mnist(alpha=2, beta=0.005, isTarget= True)
    #attack_cifar10(alpha=5, beta=0.001, isTarget= True)
    #attack_imagenet(arch='resnet50', alpha=10, beta=0.005, isTarget= True)
    #attack_imagenet(arch='vgg19', alpha=0.05, beta=0.001, isTarget= True, num_attacks= 10)

    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
