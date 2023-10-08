import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import kornia as K
from kornia.filters import filter2d
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import convolve2d

def deform_image(image, deformation, interpolation="bilinear"):
    _, _, H, W = image.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(image.device)
    deformation = deformation * mult - 1

    image = F.grid_sample(image, deformation, interpolation, padding_mode="border", align_corners=True)

    return image


def make_grid(size, step):
    grid = torch.zeros(size)
    grid[:,:, ::step] = 1.
    grid[:,:,:, ::step] = 1.
    return grid

def integrate(v, device="cpu"):
    l = len(v)
    id_grid = K.utils.grid.create_meshgrid(v[0].shape[1], v[0].shape[2], False, device)
    grid_def = id_grid - v[0]/l
    _, H, W, _ = grid_def.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)

    for t in range(1, l):
        slave_grid = grid_def.detach()
        interp_vectField = F.grid_sample(v[t].permute(0,3,1,2), slave_grid * mult - 1, "nearest", padding_mode="border", align_corners=True).permute(0,2,3,1)

        grid_def -= interp_vectField/l

    return grid_def

def integrate_stationnary(v, l=15, device="cpu"):
    id_grid = K.utils.grid.create_meshgrid(v.shape[1], v.shape[2], False, device)
    grid_def = id_grid - v/l
    _, H, W, _ = grid_def.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)

    for t in range(1, l):
        slave_grid = grid_def.detach()
        interp_vectField = F.grid_sample(v.permute(0,3,1,2), slave_grid * mult - 1, "nearest", padding_mode="border", align_corners=True).permute(0,2,3,1)

        grid_def -= interp_vectField/l

    return grid_def

def get_vnorm(residuals, fields, grad):
    return torch.stack([(residuals[j] * grad[j].squeeze(1) * (-fields[j].permute(0, 3, 1, 2))).sum() for j in
                              range(len(fields))]).sum()

def get_znorm(residuals):
    return (torch.stack(residuals) ** 2).sum()

def plot_results(source, target, source_deformed, fields, residuals, l, mu, i, v_weight, mode="opt"):
    _,_,h,w = source.shape
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
    x,y = 60, 85
    ax[0, 0].imshow(source.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
    ax[0, 0].set_title("Source")
    ax[0, 0].axis('off')
    ax[0, 1].imshow(target.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
    ax[0, 2].scatter(y,x, s=10)
    ax[0, 1].set_title("Target")
    ax[0, 1].axis('off')
    ax[1, 0].imshow((source_deformed).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
    ax[1, 0].set_title("Deformed")
    ax[1, 0].axis('off')
    superposition = torch.stack([target.squeeze().detach().cpu(),
                                 source_deformed.squeeze().detach().cpu(),
                                 torch.zeros(source.squeeze().shape)], dim=2)
    ax[1, 1].imshow(superposition, vmin=0, vmax=1)
    ax[1, 1].set_title("Superposition")
    ax[1, 1].axis('off')
    id_grid = K.utils.grid.create_meshgrid(h, w, False, source.device)
    residuals_deform = residuals[l]
    for r in range(1, l):
        res_tmp = residuals[r]
        for f in range(r, l):
            res_tmp = deform_image(res_tmp, id_grid - fields[f] / l)
        residuals_deform += res_tmp
    residuals_deform = residuals_deform * mu ** 2 / l

    im = ax[1, 2].imshow(residuals_deform.squeeze().detach().cpu().numpy())
    cbar = ax[1, 2].figure.colorbar(im, ax=ax[1, 2])
    ax[1, 2].set_title("Residuals heatmap")
    ax[1, 2].axis('off')
    ax[0, 2].set_title('Shape deformation only')
    deformation = id_grid.permute(0,3,1,2)
    for j in range(l):
        deformation = deform_image(deformation, id_grid - fields[j]/ l)
    deformation_only = deform_image(source, deformation.permute(0,2,3,1))
    dx,dy = deformation.permute(0,2,3,1)[0, x, y]
    ax[0,0].scatter(dx.detach().cpu(),dy.detach().cpu(), s=10)
    """deformation = id_grid - sum(fields) / l
    deformation_only = deform_image(source, deformation)"""


    ax[0, 2].imshow(deformation_only.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[0, 2].axis('off')

    if mode == "learning":
        fig.suptitle('Metamorphoses, epoch: %d' % (i + 1))
    else:
        fig.suptitle('Metamorphoses, iteration: %d' % (i + 1))

    plt.axis('off')
    plt.show()
    fig, ax = plt.subplots()
    c = ax.imshow(torch.abs(deformation_only).detach().cpu().squeeze(), cmap='gray')
    ax.axis("off")
    # fig.colorbar(c, ax=ax)
    plt.savefig("deformation_only.png", bbox_inches='tight')
    plt.show()
    plt.figure()
    is_diffeo = check_diffeo(deformation)
    if is_diffeo.sum() > 0:
        visual = torch.cat([deformation_only, deformation_only, deformation_only], dim=1).squeeze()
        is_diffeo = is_diffeo.squeeze()
        visual[0, is_diffeo] = 1.
        visual[1:3, is_diffeo] = 0.
        plt.imshow(visual.detach().cpu().permute(1, 2, 0))
        plt.title('Non-diffeomorphic points')
        plt.show()
        print("Deformation is not diffeomorphic: %d folds" % (is_diffeo.sum()))
    else:
        print("Deformation is diffeomorphic: %d folds" % (is_diffeo.sum()))
    plt.imshow(deformation_only.detach().squeeze().cpu(), vmin=0, vmax=1, cmap="gray")
    plt.axis('off')
    plt.savefig("/home/matthis/Images/def_only_nomask_%f_%f.png" % (mu, v_weight*l), bbox_inches='tight', pad_inches=0)
    plt.show()

def save_losses(L2_loss, L2_val, e, result_path):
    plt.figure()
    x_test = np.linspace(1, e + 1, int((e)/1) + 1)
    x_train = np.linspace(1, e + 1, e + 1)
    plt.plot(x_train, L2_loss, color='blue', label="Training")
    plt.plot(x_test, L2_val, color='red', label="Validation")
    plt.title('L2 norm during training and validation ')
    plt.xlabel('epoch')
    plt.ylabel('L2 norm')
    plt.legend()
    plt.savefig(result_path + '/loss.png')
    plt.clf()

def check_diffeo(field):
    Jac = K.filters.SpatialGradient()(field)
    det = Jac[:, 0, 0, :, :] * Jac[:, 1, 1, :, :] - Jac[:, 1, 0, :, :] * Jac[:, 0, 1, :, :]
    return det <= 0

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def dice(pred, gt):
    eps = 1e-10
    tp = torch.sum(torch.mul(pred, gt))
    fp = torch.sum(torch.mul(pred, 1 - gt))
    fn = torch.sum(torch.mul(1 - pred, gt))
    dice_eps = (2. * tp + eps) / (2. * tp + fp + fn + eps)
    return dice_eps

def field_divergence(field,dx_convention = 'pixel'):
    r"""
    make the divergence of a field, for each pixel $p$ in I
    $$div(I(p)) = \sum_{i=1}^C \frac{\partial I(p)_i}{\partial x_i}$$
    :param field: (B,H,W,2) tensor
    :return:

    """
    _,H,W,_ = field.shape
    x_sobel = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])/8
    field_as_im = field.permute(0,3,1,2)
    field_x_dx = filter2d(field_as_im[:,0,:,:].unsqueeze(1),
                          x_sobel.unsqueeze(0))
    field_y_dy = filter2d(field_as_im[:,1,:,:].unsqueeze(1),
                          x_sobel.T.unsqueeze(0))

    if dx_convention == '2square':
        _,H,W,_ = field.shape
        return field_x_dx*(H-1)/2 + field_y_dy*(W-1)/2
    else:
        return field_x_dx + field_y_dy

def analyse_div(residuals, fields, type="eulerian"):
    f = [residuals[i+1] - residuals[i] for i in range(len(residuals)-1)]
    div = []
    L2 = []
    corr = []
    if type == "eulerian":
        for i in range(len(fields)):
            div.append(-field_divergence((fields[i].permute(0,3,1,2) * residuals[i]).permute(0,2,3,1))/20.)
            L2.append(torch.sqrt(((div[i] - f[i])**2).sum()).detach().cpu().numpy()/residuals[0].shape[2]/residuals[0].shape[3])
            corr.append((pearsonr(div[i].detach().flatten().cpu().numpy(), f[i].detach().flatten().cpu().numpy())[0]))
    return L2, corr

def get_contours(label):
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    contours = convolve2d(label, laplacian, mode="same")
    contours = 1. * (contours > 0.5)
    return contours

def grayscale_to_rgb(grayscale):
    return np.stack([grayscale, grayscale, grayscale], axis=-1)

def overlay_contours(image, contours, color="red"):
    if color == "red":
        image[contours == 1.] = np.array([1., 0., 0.])
    if color == "green":
        image[contours == 1.] = np.array([0., 1., 0.])
    if color == "blue":
        image[contours == 1.] = np.array([0., 0., 1.])
    else:
        assert "color %s is not valid" %color
    return image

