import torch
import os
import numpy as np
import kornia as K
import matplotlib.pyplot as plt
import nibabel as nib

from prepare_data import C_Dataset, Brats2021_Dataset
from utils import deform_image, check_diffeo, dice
from train import eval
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib as mpl
import imageio

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)
free_gpu_id = get_free_gpu()
device = 'cuda:' + str(free_gpu_id) if torch.cuda.is_available() else 'cpu'
print(device)

def inverse_meta(target, fields, residuals, seg, mu):
    im_shape = target.shape
    id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
    l = len(fields)
    phi_inv = [id_grid]
    residuals_deform = [residuals[-1] * seg]
    pos_def = id_grid.clone()
    images=[]
    for i in range(l):
        deformation = id_grid + fields[l-i-1] / l
        pos_def = deform_image(pos_def.permute(0,3,1,2), id_grid - fields[l - i - 1] / l).permute(0,2,3,1)
        phi_inv.append(deform_image(phi_inv[-1].permute(0,3,1,2), deformation).permute(0,2,3,1))
        images.append(deform_image(target - sum(residuals_deform)/l*mu**2 *seg,phi_inv[-1]))
        residuals_deform.append(deform_image(residuals[l-i-1], pos_def))
    return images

#torch.manual_seed(24)
#seed 4 best for now or seed 7, 10 is the best, 15
result_path = "/home/matthis/Nextcloud/meta_dl/results/"
#model_path = "meta_model_0520_0950/model.pt"
#model_path = "meta_model_0523_1705/model.pt"
model_path = "meta_model_0524_0932/model.pt"
model = torch.load(result_path + model_path)

#test_set = C_Dataset(device, "test_split.txt", mode="test")
test_set = Brats2021_Dataset(device, mode="test")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=10)
target = test_set.target

#eval(model, test_loader, target, True, 0, 1, device, [], nb_img_plot=4)
loss = 1000
# 0, 26 the best
dice_list = []
L2_list = []
template_seg = torch.from_numpy(np.transpose(nib.load("/home/matthis/Nextcloud/templates/sri_seg/seg.mgz").get_fdata()[:, ::-1, 80].copy())).unsqueeze(0).unsqueeze(0)
template_seg = 1. * (template_seg == 4) + 1. * (template_seg == 43)
seg_path = '/home/matthis/datasets/brats_seg_test/'
for i in tqdm(range(len(test_set))):
    if i == 26:
        image, seg = test_set[i]
        source_mask = np.transpose(nib.load(seg_path + test_set.test_list_names[i] + "/seg.nii").get_fdata()[:, :, 80])
        source_mask = torch.from_numpy(source_mask).unsqueeze(0).unsqueeze(0).float()
        image = image.unsqueeze(0)
        seg = seg.unsqueeze(0)
        model.train()
        phi_list = []
        deform_list = []
        for j in range(20):
            source_deformed, fields, residuals, _ = model(image.to(device), seg.to(device))
            phi_list.append(model.phi[-1].detach().cpu())
            deform_list.append(source_deformed.detach().cpu())
            is_diff = check_diffeo(phi_list[-1].permute(0, 3, 1, 2))
            if is_diff.sum() > 0:
                print("Sample number %d is not diffeomorphic, number of folds: %d" %(i, is_diff.sum()))
        deform_list = torch.stack(deform_list)
        phi_list = torch.stack(phi_list)
        uc_def = torch.std(deform_list, dim=0)
        uc_phi = torch.std(phi_list, dim=0)
        avg_def = torch.mean(deform_list, dim=0)
        avg_phi = torch.mean(phi_list, dim=0)
        deformed_mask = deform_image(source_mask, avg_phi)
        dice_score = dice(deformed_mask, template_seg)
        dice_list.append(dice_score.cpu().item())
        L2_list.append(((avg_def.detach().cpu() - target.unsqueeze(0)) ** 2).sum())
        is_diff = check_diffeo(avg_phi.permute(0, 3, 1, 2))
        if is_diff.sum() > 0:
            print("Average sample is not diffeomorphic, number of folds: %d" %(is_diff.sum()))
        deform_only = deform_image(image, avg_phi).squeeze()
        print(uc_def.max())
        uc_phi = uc_phi.sum(dim=3)
        fig, ax = plt.subplots(2,3)
        ax[0,1].imshow(avg_def.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[0, 1].set_title("mean transformation")
        ax[0,2].imshow(uc_def.squeeze(), cmap="gray")
        ax[0, 2].set_title("uc meta.")
        ax[1,1].imshow(deform_only.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[1, 1].set_title("mean def.")
        ax[1,2].imshow(uc_phi.squeeze(), cmap="gray")
        ax[1,2].set_title("uc deformation")
        ax[0,0].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[0,0].set_title("source")
        ax[1,0].imshow(target.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[1,0].set_title("target")
        ax[1, 1].axis("off")
        ax[0, 1].axis("off")
        ax[1, 0].axis("off")
        ax[0, 0].axis("off")
        ax[0, 2].axis("off")
        ax[1, 2].axis("off")
        #plt.title("image %d" %i)
        plt.subplots_adjust(wspace=0, hspace=0.01)
        plt.margins(0,0)
        plt.savefig("../results/figs/uncertainty.png")
        plt.show()
print("Dice average:", np.array(dice_list).mean(), "Dice std:", np.array(dice_list).std())
print("Validation L2 loss: %f" % (sum(L2_list) / len(test_loader)), "std: %f" % (np.array(L2_list).std()))
images = inverse_meta(source_deformed, fields, residuals, model.seg, model.mu)
fig, ax = plt.subplots(4,4)
ax = ax.ravel()
ax[0].imshow(source_deformed.squeeze().detach().cpu(), cmap="gray", vmax=1., vmin=0.)
ax[0].axis("off")
for i in range(15):
    ax[i+1].imshow(images[i].squeeze().detach().cpu(), cmap="gray", vmax=1., vmin=0.)
    ax[i + 1].axis("off")
plt.subplots_adjust(wspace=0, hspace=0.05)
plt.margins(0, 0)
plt.show()
plt.figure()
plt.imshow(source_deformed.squeeze().detach().cpu(), cmap="gray", vmax=1., vmin=0.)
plt.axis("off")
plt.savefig("../results/figs/deformed.png", bbxox_inches="tight")
plt.show()
for i in range(15):
    plt.figure()
    plt.imshow(images[i].squeeze().detach().cpu(), cmap="gray", vmax=1., vmin=0.)
    plt.axis("off")
    plt.savefig("../results/figs/image_%d.png" %i)
    plt.show()

fig, ax = plt.subplots(1,4, gridspec_kw={"width_ratios":[1,1, 1, 0.05]})
ax[0].imshow(image.squeeze().detach().cpu(), cmap="gray", vmax=1., vmin=0.)
ax[0].set_title("Source")
ax[1].imshow(images[-1].squeeze().detach().cpu(), cmap="gray", vmax=1., vmin=0.)
ax[1].set_title("Backward Image")
superposition = torch.abs(image.squeeze().detach().cpu() - images[-1].squeeze().detach().cpu())
im = ax[2].imshow(superposition, cmap="viridis")
ip = InsetPosition(ax[2], [1.05,0,0.05,1])
ax[3].set_axes_locator(ip)
fig.colorbar(im, cax=ax[3], ax = [ax[0], ax[1], ax[2]])
ax[2].set_title("Difference")
ax[1].axis("off")
ax[0].axis("off")
ax[2].axis("off")
plt.subplots_adjust(wspace=0, hspace=0.05)
plt.margins(0,0)
#plt.tight_layout()
plt.savefig("../results/figs/inverse_def.png")
plt.show()

video_image = []
n_frame = 3
image = source_deformed
image[image > 1.] = 1.
image[image < 0.] = 0.
for j in range(n_frame):
    video_image.append((image.detach().squeeze().cpu() * 255.).type(torch.uint8))
for i in range(len(images)):
    image = images[i]
    image[image > 1.] = 1.
    image[image < 0.] = 0.
    for j in range(n_frame):
        video_image.append((image.detach().squeeze().cpu()*255.).type(torch.uint8))

imageio.mimsave('../results/figs/meta.gif', video_image)




