import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from datetime  import timedelta
import cv2

from time import time
from models import meta_model, meta_model_local, meta_model_sharp, meta_model_local_sharp, double_resnet, shooting_model, metamorphoses
from train import train_opt
from prepare_data import C_Dataset
import nibabel as nib
from skimage.exposure import match_histograms
from PIL import Image
from utils import get_contours
from utils import deform_image, dice
import random



if __name__ == "__main__":
    def get_free_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)


    free_gpu_id = get_free_gpu()
    device = 'cuda:' + str(free_gpu_id) if torch.cuda.is_available() else 'cpu'
    print(device)

    use_segmentation = True

    n_epoch = 100
    l = 15
    L2_weight = .5
    lamda = 3e-7
    v_weight = lamda / l
    z_weight = lamda/ l
    mu = 0.05
    batch_size = 1
    kernel_size = 31
    sigma = 6.
    debug = False

    # test_set = C_Dataset(device, "test_split.txt", mode="test")
    # test_set = Brats2021_Dataset(device, mode="test")
    dir = '/home/matthis/datasets/MICCAI_BraTS_2018_Data_Training/HGG/'
    image = 'Brats18_CBICA_ATB_1'
    im = nib.load(dir + image + "/" + image + "_t1.nii.gz").get_fdata()
    #seg = nib.load(dir + image + "/" + image + "_seg.nii.gz").get_fdata()
    #seg=seg[:,:,80].transpose()
    #seg[seg>0] = 1
    s = cv2.imread("/home/matthis/Images/mask-1.png", cv2.IMREAD_GRAYSCALE)
    seg = np.array(cv2.resize(s, (240,240) ))
    seg = (seg > 0.5) * 1.
    im[im<0] = 0
    im = (im - im.min()) / (im.max() - im.min())
    im = im[:,:,80].transpose()
    target_img = np.transpose(np.load("/home/matthis/datasets/sri24_t1_preprocessed.npy").squeeze())


    im[im!=0] = match_histograms(im[im!=0], target_img[target_img!=0])
    target_img = target_img[np.newaxis, np.newaxis, ...].copy()
    target_img = torch.from_numpy(target_img).float().to(device)
    source_img = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0)
    seg = torch.from_numpy(seg[np.newaxis, np.newaxis, ...].copy()).float()
    seg = torch.ones(source_img.shape)
    source = [source_img, seg]

    #print("Number of test images:", len(test_set))
    #target_img = test_set.target.to(device)

    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)
    be_sharp = True
    z0 = torch.zeros(target_img.shape)

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_epoch=", n_epoch)
    print("mu=", mu)
    print("sigma=", sigma)
    t = time()

    L2_norm_list = []
    def_list = []
    num_folds = []
    time_list = []
    landa_list = [3e-6]
    mu_list = [0.04]
    for mu in mu_list:
        for lamda in landa_list:
            v_weight = lamda / l
            z_weight = lamda / l
            print("mu=", mu)
            print("lambda=", lamda)
            t=time()
            model = metamorphoses(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
            optimizer = torch.optim.Adam(list(model.parameters()), lr=1e0, weight_decay=1e-8)
            L2, folds = train_opt(model, source, target_img, optimizer, device, n_iter=100, local_reg=use_segmentation, double_resnets=False, debug=debug, plot_iter=500, L2_weight=L2_weight, v_weight=v_weight, z_weight=z_weight)
            L2_norm_list.append(L2.detach().cpu().item())
            num_folds.append(folds.detach().cpu().item())
            time_list.append(time()-t)
            print("L2 loss:", L2.detach().cpu().item(),  "Fold number:", folds.detach().cpu().item(), "Time:", str(timedelta(seconds=time_list[-1])))

    print("Validation L2 loss: %f" % (sum(L2_norm_list) / len(test_loader)),
          "std: %f" % (np.array(L2_norm_list).std()))
    print("Validation L2 deformation only: %f" % (sum(def_list) / len(test_loader)),
          "std: %f" % (np.array(def_list).std()))
    print("Average fold number:", sum(num_folds) / len(num_folds), "std: %f" % (np.array(num_folds).std()))
    print("Average time :", str(timedelta(seconds=sum(time_list) / len(num_folds))), "std: %f" % (np.array(time_list).std()))




