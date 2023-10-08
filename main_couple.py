import torch
from time import time
from models import meta_couple
from train_couple import train
from prepare_data import  C_Dataset_Couple
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

    n_epoch = 300
    l = 10
    L2_weight = .5
    v_weight = 3e-6/l
    z_weight = 3e-8/l
    batch_size = 5
    kernel_size = 31
    sigma = 3.
    debug = True

    train_set = C_Dataset_Couple(device, "test_couple_split.txt", mode="train")
    test_set = C_Dataset_Couple(device, "test_couple_split.txt", mode="test")
    print("Number of training images:", len(train_set))
    print("Number of test images:", len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)
    z0 = torch.zeros(train_set.shape).unsqueeze(0)

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_epoch=", n_epoch)
    print("sigma=", sigma)
    t = time()

    model = meta_couple(l, train_set.shape, device, kernel_size, sigma, z0).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
                                 weight_decay=1e-8)
    model = train(model, train_loader, test_loader, optimizer, device, batch_size=batch_size, n_epoch=n_epoch, debug=debug, plot_epoch=101, L2_weight=L2_weight, v_weight=v_weight, z_weight=z_weight)
