import matplotlib.pyplot as plt

from utils import get_vnorm, dice, save_losses, deform_image, check_diffeo
from tqdm import tqdm
import time
import torch
import os
import numpy as np
import kornia as K




def train(model, train_loader, test_loader, optimizer, device, batch_size=4, n_epoch=50, debug=False, plot_epoch=5, L2_weight=.5, v_weight=3e-6, z_weight=3e-6):
    l = model.l
    L2_loss = []
    L2_val = []
    result_path = "../results/" + "meta_couple_" + time.strftime("%m%d_%H%M", time.localtime())
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for e in range(n_epoch):
        if e == 15 and debug:
            break
        model.train()
        total_loss_avg = 0
        L2_norm_avg = 0
        residuals_norm_avg = 0
        v_norm_avg = 0
        n_iter = len(train_loader)
        for i, batch in enumerate(tqdm(train_loader)):
            if i == 5 and debug:
                break
            source = batch[0].to(device)
            target = batch[1].to(device)
            source_deformed, fields, residuals, grad = model(source, target)

            v_norm = get_vnorm(residuals, fields, grad) / batch_size
            L2_norm = ((source_deformed - target) ** 2).sum() / batch_size
            total_loss = L2_weight * L2_norm + v_weight * v_norm
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_avg += total_loss / n_iter
            L2_norm_avg += L2_norm / n_iter
            v_norm_avg += v_norm / n_iter

        L2_loss.append(L2_norm_avg.detach().cpu())

        print("Training: epoch %d, total loss: %f, L2 norm: %f, v_norm: %f" % (
            e + 1, total_loss_avg, L2_norm_avg, v_norm_avg))

        if e == 40:
            for g in optimizer.param_groups:
                g['lr'] = 5e-4
        if e == 80:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if e == 120:
            for g in optimizer.param_groups:
                g['lr'] = 5e-5

        ### Validation
        if (e + 1) % 10 == 0:
            L2_val = eval(model, test_loader, e, plot_epoch, device, L2_val)
            torch.save(model, result_path + "/model.pt")
            #save_losses(L2_loss, L2_val, e, result_path)

    return model


def eval(model, test_loader, e, plot_iter, device, L2_val, nb_img_plot=0):
    with torch.no_grad():
        model.eval()
        L2_norm_mean = []
        def_dist = []
        dice_C = []
        num_folds = []
        for i, batch in enumerate(tqdm(test_loader)):
            source = batch[0].to(device)
            target = batch[1].to(device)
            source_def = batch[2].to(device)
            source_deformed, fields, residuals, _ = model(source, target)
            num_folds.append(check_diffeo(model.phi[-1].permute(0, 3, 1, 2)).sum().detach().cpu())
            if num_folds[-1] > 0:
                print("the deformation number %d is not diffeomorphic, number of folds %f" % (i, num_folds[-1].item()))
            id_grid = K.utils.grid.create_meshgrid(fields[0].shape[1], fields[0].shape[2], False, "cpu")
            inversed_comp = deform_image(model.phi[-1].permute(0, 3, 1, 2), source_def,
                                         interpolation="nearest").permute(0, 2, 3, 1)
            def_dist.append(torch.sqrt(torch.sum((id_grid - inversed_comp.detach().cpu()) ** 2)))
            L2_norm = ((source_deformed - target.unsqueeze(0)) ** 2).sum().detach().cpu()
            L2_norm_mean.append(L2_norm)
            dice_C.append(dice(source_deformed, target.unsqueeze(0)).detach().cpu())

            if (e + 1) % plot_iter == 0 and i < nb_img_plot:
                fig, ax = plt.subplots(1, 3, figsize=(7.5, 5), constrained_layout=True)
                ax[0].imshow(source.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                ax[0].set_title("Source")
                ax[0].axis('off')
                ax[1].imshow(target.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                ax[1].set_title("Target")
                ax[1].axis('off')
                ax[2].imshow((source_deformed).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                ax[2].set_title("Deformed")
                ax[2].axis('off')
                plt.show()

        print("Validation L2 loss: %f" % (sum(L2_norm_mean) / len(test_loader)), "std: %f" % (np.array(L2_norm_mean).std()))
        print("Deformation difference:", sum(def_dist).item() / len(def_dist), "std: %f" % (np.array(def_dist).std()))
        print("Dice score:", sum(dice_C) / len(dice_C), "std: %f" % (np.array(dice_C).std()))
        print("Average fold number:", sum(num_folds) / len(num_folds), "std: %f" % (np.array(num_folds).std()))
        L2_val.append(sum(L2_norm_mean) / len(test_loader))

    return (L2_val)

