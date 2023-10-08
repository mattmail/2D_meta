import matplotlib.pyplot as plt

from utils import check_diffeo, get_vnorm, get_znorm, plot_results, deform_image, grayscale_to_rgb, overlay_contours, get_contours
from tqdm import tqdm
import time
import torch
import os
import numpy as np
import kornia as K


def train_opt(model, source, target, optimizer, device, n_iter=1000, local_reg=False, double_resnets=False, debug=False, plot_iter=500, L2_weight=.5, v_weight=3e-6, z_weight=3e-6):
    l = model.l
    mu = model.mu
    t=time.time()
    previous_loss = 0
    if local_reg:
        source_img = source[0].to(device)
        source_seg = source[1].to(device)
        #source_def = source[2].to(device)
        #source_seg = torch.ones(source_img.shape).to(device)
    else:
        source_img = source
    i=0
    continue_iter=True
    while continue_iter:
        if i == 5 and debug:
            break
        if local_reg:
            if double_resnets:
                source_deformed, fields, residuals, grad = model(source_img, source_seg)
            else:
                source_deformed, fields, residuals, grad = model(source_img, target,  source_seg)
        else:
            source_deformed, fields, residuals, grad = model(source_img)
        if not double_resnets:
            v_norm = get_vnorm(residuals, fields, grad)
        else:
            v_norm = sum([(field ** 2).sum() for field in fields])
        residuals_norm = get_znorm(residuals)
        L2_norm = ((source_deformed - target) ** 2).sum()
        total_loss = L2_weight * L2_norm + v_weight * (v_norm) + mu * z_weight * residuals_norm
        if i%10 == 0:
            print("\rIteration %d: Total loss: %f   L2 norm: %f   V_reg: %f   Z_reg: %f " % (i, total_loss, L2_norm, v_norm, residuals_norm), end='')


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        """if i == 500:
            #for param in list(model.parameters()):
            #    param.requires_grad = True
            for g in optimizer.param_groups:
                g['lr'] = 5e-4
        if i == 1000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if i == 1500:
            for g in optimizer.param_groups:
                g['lr'] = 5e-4"""
        """if (i + 1) % plot_iter == 0:
            plot_results(source_img, target, source_deformed, fields, residuals, model.l, model.mu, i)"""
        if torch.abs(previous_loss - total_loss) < 1e-5:
            continue_iter = False
        else:
            previous_loss = total_loss
            i+=1
    #id_grid = K.utils.grid.create_meshgrid(fields[0].shape[1], fields[0].shape[2], False, "cpu")
    """inversed_comp = deform_image(model.phi_list[0].permute(0, 3, 1, 2), source_def, interpolation="nearest").permute(0,
                                                                                                                      2,
                                                                                                                      3,
                                                                                                                      1)"""
    #def_dist = torch.sqrt(torch.sum((id_grid - inversed_comp.detach().cpu()) ** 2))
    folds = check_diffeo(model.phi_list[0].permute(0, 3, 1, 2)).sum().detach().cpu()
    plot_results(source_img, target, source_deformed, fields, residuals, model.l, model.mu, i, v_weight)
    plt.imshow(source_deformed.detach().squeeze().cpu(), vmin=0, vmax=1, cmap="gray")
    plt.axis('off')
    plt.savefig("/home/matthis/Images/def_total_nomask_%f_%f.png" %(mu, v_weight*l), bbox_inches='tight', pad_inches=0)
    plt.show()
    plot_image = grayscale_to_rgb(source_img.detach().squeeze().cpu().numpy())
    contours = get_contours((source_seg.squeeze().detach().cpu().numpy() > 0.5) * 1.)
    plot_image = overlay_contours(plot_image, contours)
    plt.imshow(plot_image)
    #plt.imshow(source_img.detach().squeeze().cpu(), vmin=0, vmax=1, cmap="gray")
    plt.axis('off')
    plt.savefig("/home/matthis/Images/source.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    #plt.show()

    return L2_norm, folds


def train_learning(model, train_loader, test_loader, target_img, optimizer, device, batch_size=4, n_epoch=50, local_reg=False, debug=False, plot_epoch=5, L2_weight=.5, v_weight=3e-6, z_weight=3e-6):
    target = torch.cat([target_img for _ in range(batch_size)], dim=0).to(device)
    l = model.l
    mu = model.mu
    L2_loss = []
    L2_val = []
    if local_reg:
        result_path = "../results/" + "meta_model_" + time.strftime("%m%d_%H%M", time.localtime())
    else:
        result_path = "../results/" + "meta_model_local_" + time.strftime("%m%d_%H%M", time.localtime())
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for e in range(n_epoch):
        if e == 5 and debug:
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
            if local_reg:
                source_img = batch[0].to(device)
                source_seg = batch[1].to(device)
                source_deformed, fields, residuals, grad = model(source_img, target, source_seg)
            else:
                source_img = train_list[indexes[i * batch_size:(i + 1) * batch_size]].to(device)
                source_deformed, fields, residuals, grad = model(source_img)

            v_norm = get_vnorm(residuals, fields, grad) / batch_size
            residuals_norm = get_znorm(residuals) / batch_size
            L2_norm = ((source_deformed - target) ** 2).sum() / batch_size
            #mask_norm = model.seg.sum()/((source_deformed > 0)*1.).sum() - source_seg.sum()/((source_img > 0)*1.).sum()
            total_loss = L2_weight * L2_norm + v_weight * v_norm + mu * z_weight * residuals_norm #+ 1.*mask_norm
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_avg += total_loss / n_iter
            L2_norm_avg += L2_norm / n_iter
            residuals_norm_avg += residuals_norm / n_iter
            v_norm_avg += v_norm / n_iter

        L2_loss.append(L2_norm_avg.detach().cpu())

        print("Training: epoch %d, total loss: %f, L2 norm: %f, v_norm: %f, residuals: %f" % (
            e + 1, total_loss_avg, L2_norm_avg, v_norm_avg, residuals_norm_avg))

        if e == 30:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if e == 60:
            for g in optimizer.param_groups:
                g['lr'] = 5e-5
        if e == 80:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5

        ### Validation
        if (e+1) % 1 == 0:
            L2_val = eval(model, test_loader, target_img, local_reg, e, plot_epoch, device, L2_val)
            #torch.save(model, result_path + "/model.pt")
            #save_losses(L2_loss, L2_val, e, result_path)


            """model_path = "/home/matthis/Nextcloud/ventricleSeg/results/baseline_0319_2007/pkl/Checkpoint_best.pkl"
            checkpoint = torch.load(model_path)
            seg_model = UNet(checkpoint['config']).to(device)
            seg_model.load_state_dict(checkpoint['model_state_dict'])
            MNI_img = nib.load("/usr/local/fsl/data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz").get_fdata()
            mask = MNI_img != 0
            MNI_img = MNI_img / 255.
            MNI_img_scaled = MNI_img.copy()
            MNI_img_scaled = (MNI_img - MNI_img[mask].mean()) / MNI_img[mask].std()
            MNI_img_scaled = np.pad(MNI_img_scaled[:, :, 93], ((18, 18), (0, 0)), "constant")
            MNI_img_scaled = np.rot90(cv2.resize(MNI_img_scaled, (208, 208)))
            MNI_img_scaled = torch.from_numpy(MNI_img_scaled.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(
                device)
            target_seg = seg_model(MNI_img_scaled)
            target_seg = 1. * (target_seg > 0.5)

            dice_list = []
            L2_list = []
            for i in range(len(test_list)):
                source_mask = np.transpose(nib.load(seg_path + test_list_names[i] + "/seg.nii").get_fdata()[:, :, 80])
                source_mask = torch.from_numpy(source_mask).unsqueeze(0).unsqueeze(0).to(device).float()
                source = test_list[i]
                source_img = source[0].unsqueeze(0).to(device)
                source_seg = source[1].unsqueeze(0).to(device)
                source_deformed, fields, residuals, grad = model(source_img, source_seg)
    
                deformed_mask = deform_image(source_mask, model.phi[-1])
                dice_score = dice(deformed_mask, template_seg)
                source = test_list[i]
                source_img = source[0].unsqueeze(0).to(device)
                source_seg = source[1].unsqueeze(0).to(device)
                source_deformed, fields, residuals, grad = model(source_img, source_seg)
                _, _, h, w = source_img.shape
                id_grid = K.utils.grid.create_meshgrid(h, w, False, device)
                deformation = id_grid.permute(0, 3, 1, 2)
                for j in range(l):
                    deformation = deform_image(deformation, id_grid - fields[j] / l)
                deformation_only = deform_image(source_img, deformation.permute(0, 2, 3, 1))
                L2_list.append(((deformation_only - target) ** 2).sum().detach().cpu())
                s_mask = deformation_only != 0
                deformation_only = (deformation_only - deformation_only[s_mask].mean()) / (
                            deformation_only[s_mask].std() + 1e-30)
                pred_seg = seg_model(deformation_only)
                pred_seg = 1. * (pred_seg > 0.5)
                dice_score = dice(pred_seg, target_seg)
                dice_list.append(dice_score.cpu().item())
                print("Deformation Dice score:", dice_score.cpu().item())
                print("L2 norm:", L2_list[-1].item())
            print("Dice average:", np.array(dice_list).mean())
            print("Dice std:", np.array(dice_list).std())
            print("L2 average:", np.array(L2_list).mean())
            print("L2 std:", np.array(L2_list).std())"""
    return model

def eval(model, test_loader, target, local_reg, e, plot_iter, device, L2_val, nb_img_plot=4):
    with torch.no_grad():
        model.eval()
        L2_norm_list = []
        def_dist = []
        #dice_C = []
        num_folds = []
        #prob_map_seg = torch.zeros(target.shape)
        #probs_count = 0
        for i, batch in enumerate(tqdm(test_loader)):
            if local_reg:
                source_img = batch[0].to(device)
                source_seg = batch[1].to(device)
                source_def = batch[2].to(device)
                source_deformed, fields, residuals, _ = model(source_img, target.to(device), source_seg)
                num_folds.append(check_diffeo(model.phi_list[0].permute(0, 3, 1, 2)).sum().detach().cpu())
                if num_folds[-1] > 0:
                    print("the deformation number %d is not diffeomorphic, number of folds %f" %(i,num_folds[-1].item()))
                id_grid = K.utils.grid.create_meshgrid(fields[0].shape[1], fields[0].shape[2], False, "cpu")
                inversed_comp = deform_image(model.phi_list[0].permute(0, 3, 1, 2), source_def, interpolation="nearest").permute(0, 2, 3, 1)
                def_dist.append(torch.sqrt(torch.sum((id_grid - inversed_comp.detach().cpu()) ** 2)))
                #dice_C.append(dice(source_deformed.detach().cpu(), target.unsqueeze(0)))
            else:
                source_img = test_list[i].to(device).unsqueeze(0)
                source_deformed, fields, residuals, grad = model(source_img)
            L2_norm = ((source_deformed.detach().cpu() - target.unsqueeze(0)) ** 2).sum()
            L2_norm_list.append(L2_norm.detach().cpu())

            """if source_seg.sum() > 0:
                prob_map_seg += model.seg.detach().cpu()
                probs_count +=1"""
            if (e + 1) % plot_iter == 0 and i < nb_img_plot:
                plot_results(source_img, target, source_deformed, fields, residuals, model.l, model.mu, e, mode="learning")
                """fig, ax = plt.subplots(2)
                ax[0].imshow(source_seg.squeeze().cpu(), cmap="gray")
                ax[1].imshow(model.seg.detach().squeeze().cpu(), cmap="gray")
                plt.show()"""

        print("Validation L2 loss: %f" %(sum(L2_norm_list) /len(test_loader)), "std: %f" %(np.array(L2_norm_list).std()))
        print("Deformation difference:", sum(def_dist).item() / len(def_dist), "std: %f" %(np.array(def_dist).std()))
        #print("Dice score:", sum(dice_C) / len(dice_C), "std: %f" %(np.array(dice_C).std()))
        print("Average fold number:", sum(num_folds) / len(num_folds), "std: %f" %(np.array(num_folds).std()))
        L2_val.append(sum(L2_norm_list) / len(test_loader))
        """fig, ax = plt.subplots()
        im = ax.imshow(model.z0.squeeze().detach().cpu())
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()"""
    return (L2_val)

