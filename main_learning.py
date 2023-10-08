import torch
import kornia.filters as flt
import numpy as np
import cv2
import os
import nibabel as nib
from time import time
from models import meta_model, meta_model_local, meta_model_sharp, meta_model_local_sharp, shooting_model
from train import train_learning
from prepare_data import load_brats_2021, load_brats_2020, load_brats_2021_linear, C_Dataset, Brats2021_Dataset
from utils import deform_image, dice, check_diffeo
from model import UNet
import kornia as K
import matplotlib.pyplot as plt
import os
import numpy as np

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
    lamda = 3e-6
    v_weight = lamda/l
    z_weight = 0/l
    mu = 0.01
    batch_size = 5
    kernel_size = 31
    sigma = 6.
    debug = False

    train_set = C_Dataset(device, "test_split.txt", mode="train")
    test_set = C_Dataset(device, "test_split.txt", mode="test")
    #train_set = Brats2021_Dataset(device, mode="train")
    #test_set = Brats2021_Dataset(device, mode="test")
    print("Number of training images:", len(train_set))
    print("Number of test images:", len(test_set))
    target_img = train_set.target

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)
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
    """if use_segmentation:
        if be_sharp:
            model = meta_model_local_sharp(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
        else:
            model = meta_model_local(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
    else:
        if be_sharp:
            model = meta_model_sharp(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
        else:
            model = meta_model(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)"""

    #model = torch.load("../results/meta_model_1123_0955/model.pt", map_location=device)

    #optimizer = torch.optim.Adam(list(model.parameters()) + [z0], lr=1e-4,
    #                             weight_decay=1e-8)
    #model = shooting_model(l, target_img.shape, device, kernel_size, sigma, mu).to(device)
    model = meta_model_local_sharp(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=5e-4, weight_decay=1e-8)
    model = train_learning(model, train_loader, test_loader, target_img, optimizer, device, batch_size=batch_size, n_epoch=n_epoch, local_reg=use_segmentation, debug=debug, plot_epoch=101, L2_weight=L2_weight, v_weight=v_weight, z_weight=z_weight)
    """template_seg = torch.from_numpy(np.transpose(nib.load("/home/matthis/Nextcloud/templates/sri_seg/seg.mgz").get_fdata()[:, ::-1, 80].copy())).unsqueeze(0).unsqueeze(0).to(device)
    template_seg = 1. * (template_seg == 4) + 1. * (template_seg == 43)
    seg_path = '/home/matthis/datasets/brats_seg_test/'

    dice_list = []
    num_folds = []
    L2_norm_list = []

    for i, batch in enumerate(test_loader):
        source_mask = np.transpose(nib.load(seg_path + test_set.test_list_names[i] + "/seg.nii").get_fdata()[:, :, 80])
        source_mask = torch.from_numpy(source_mask).unsqueeze(0).unsqueeze(0).to(device).float()
        source = batch
        source_img = source[0].to(device)
        source_seg = source[1].to(device)
        source_deformed, fields, residuals, grad = model(source_img, source_seg)

        deformed_mask = deform_image(source_mask, model.phi[-1])
        dice_score = dice(deformed_mask, template_seg)
        dice_list.append(dice_score.cpu().item())
        num_folds.append(check_diffeo(model.phi[-1].permute(0, 3, 1, 2)).sum().detach().cpu())
        if num_folds[-1] > 0:
            print("the deformation number %d is not diffeomorphic, number of folds %f" % (i, num_folds[-1].item()))
        L2_norm = ((source_deformed.detach().cpu() - target_img.unsqueeze(0)) ** 2).sum()
        L2_norm_list.append(L2_norm.detach().cpu())
        print("Deformation Dice score:", dice_score.cpu().item())
    print("Dice average:", np.array(dice_list).mean(), "Dice std:", np.array(dice_list).std())
    print("Validation L2 loss: %f" % (sum(L2_norm_list) / len(test_loader)), "std: %f" % (np.array(L2_norm_list).std()))
    print("Average fold number:", sum(num_folds) / len(num_folds), "std: %f" % (np.array(num_folds).std()))"""




