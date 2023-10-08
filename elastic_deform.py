import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import torch
from utils import deform_image, field_divergence, check_diffeo
import kornia as K
from scipy import ndimage
import os
import torch.nn.functional as F

def integrate_staionnary(v):
    l=20
    device='cpu'
    id_grid = K.utils.grid.create_meshgrid(v.shape[1], v.shape[2], False, device)
    grid_def = id_grid - v/l
    _, H, W, _ = grid_def.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)

    for t in range(1, l):
        slave_grid = grid_def.detach()
        interp_vectField = F.grid_sample(v.permute(0,3,1,2), slave_grid * mult - 1, "bicubic", padding_mode="border", align_corners=True).permute(0,2,3,1)
        grid_def -= interp_vectField/l

    return grid_def

def integrate(v, device="cpu"):
    l = len(v)
    id_grid = K.utils.grid.create_meshgrid(v[0].shape[1], v[0].shape[2], False, device)
    grid_def = id_grid - v[0]/l
    _, H, W, _ = grid_def.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)

    for t in range(1, l):
        slave_grid = grid_def.detach()
        interp_vectField = F.grid_sample(v[t].permute(0,3,1,2), slave_grid * mult - 1, "bilinear", padding_mode="border", align_corners=True).permute(0,2,3,1)

        grid_def -= interp_vectField/l

    return grid_def

def integrate_momentum(momentum, image):
    l=50
    device='cpu'
    id_grid = K.utils.grid.create_meshgrid(momentum.shape[0], momentum.shape[1], False, device)
    kernel = K.filters.GaussianBlur2d((31, 31),
                                      (6., 6.),
                                      border_type='constant')
    source = image.clone()
    z = momentum
    field = []
    phi = [id_grid]
    for i in range(l):
        grad_image = K.filters.SpatialGradient(mode='sobel')(image)
        v = -kernel(z * grad_image.squeeze(1))
        v = v.permute(0, 2, 3, 1)
        field.append(v)
        deformation = id_grid - v / l
        div = field_divergence((v.permute(0,3,1,2) * z).permute(0,2,3,1)) * 1/l
        z = z - div
        phi.append(deform_image(phi[-1].permute(0, 3, 1, 2), deformation, interpolation="bilinear").permute(0, 2, 3, 1))
        image = deform_image(source, phi[i + 1])
    return phi[-1], field


def elastic_transform(image, seg, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    #if random_state is None:
    #    random_state = np.random.RandomState(None)

    shape = image.shape
    source_torch = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    grid = K.utils.grid.create_meshgrid(shape[0], shape[1], False)
    #v_torch = torch.stack((torch.tensor(dx), torch.tensor(dy)), dim=2).unsqueeze(0).float()
    momentum = torch.tensor(np.random.rand(*shape) * 2 - 1).float()* alpha
    #v_torch = kernel(-momentum * grad_image.squeeze(1)).permute(0,2,3,1)
    ind_torch, v = integrate_momentum(momentum, source_torch)
    v_inv = [-v[len(v)-1-i] for i in range(len(v))]
    inv_grid = integrate(v_inv)
    deformed = deform_image(source_torch, ind_torch, interpolation="nearest")

    #deformed = (deformed > 0.5)*1.
    original = deform_image(deformed, inv_grid, interpolation="nearest")
    #original = (original > 0.5)*1.
    if seg is not None:
        seg_def = deform_image(torch.tensor(seg).unsqueeze(0).unsqueeze(0).float(), ind_torch, interpolation="nearest")
    """fig, ax = plt.subplots(3)
    ax[0].imshow(source_torch.squeeze(), cmap='gray')
    ax[1].imshow(deformed.squeeze(), cmap='gray')
    ax[2].imshow(original.squeeze(), cmap='gray')
    #ax[2].set_title(str(torch.sum(torch.abs(original - source_torch))))
    #ax[3].imshow(torch.abs(original - source_torch).squeeze())
    plt.show()"""
    """plt.figure()
    plt.imshow(original.squeeze(), cmap="gray")
    plt.axis('off')
    plt.show()"""
    is_diff = check_diffeo(ind_torch.permute(0,3,1,2))
    goodness = (is_diff).sum() == 0
    #goodness = torch.sum(torch.abs(original - source_torch))
    if seg is not None:
        return np.array(deformed.squeeze()), np.array(ind_torch.squeeze()), np.array(seg_def.squeeze()), goodness
    else:
        return np.array(deformed.squeeze()), np.array(ind_torch.squeeze())

def change_topology(source):
    image = source.copy()
    angle = np.random.randint(-40, 40)
    rectangle = np.zeros((image.shape))
    h, w = rectangle.shape
    width = np.random.randint(5, 15)
    rectangle[h // 2 - width:h // 2 + width, 0:w // 2] = 1.
    rectangle = ndimage.rotate(rectangle, angle, reshape=False, order=0)
    rectangle = (rectangle > .5) * 1
    segmentation = ((rectangle + image) == 2) * 1

    image[rectangle == 1] = 0.
    return image, segmentation



source = cv2.resize(cv2.imread("../images/reg_test_m0.png", 0), (200,200))/255.
np.random.seed(0)
for i in range(2000):
    if not os.path.exists('../images_seg/image_%d' % i):
        os.mkdir('../images_seg/image_%d' % i)


    im, seg = change_topology(source)
    deformed, field, seg, goodness = elastic_transform(im, seg, 1000., 8.)
    while not goodness:
        deformed, field, seg, goodness = elastic_transform(im, seg, 1000., 8.)
    print("Image %d saved" % i)
    np.save("../images_seg/image_%d/image.npy" % i, deformed)
    np.save("../images_seg/image_%d/seg.npy" % i, seg)
    np.save("../images_seg/image_%d/deformation.npy" % i, field)
    """fig, ax = plt.subplots(3)
    ax[0].imshow(source, cmap='gray')
    ax[1].imshow(deformed, cmap='gray')
    ax[2].imshow(seg, cmap='gray')
    plt.show()"""

"""source = cv2.resize(cv2.imread("../images/reg_test_m0.png", 0), (200,200))/255.
for i in range(2000):
    if not os.path.exists('../images_sig6/image_%d' % i):
        os.mkdir('../images_sig6/image_%d' % i)
    
    
    deformed, _ = elastic_transform(source, None, 1000., 6.)
    #target, field = elastic_transform(deformed, None, 1000., 8.)
    print("Image %d saved" % i)
    np.save("../images_couple/image_%d/image.npy" % i, deformed)
    np.save("../images_couple/image_%d/target.npy" % i, target)
    np.save("../images_couple/image_%d/deformation.npy" % i, field)"""
"""fig, ax = plt.subplots(2)
    ax[0].imshow(target, cmap='gray')
    ax[1].imshow(deformed, cmap='gray')
    plt.show()"""



