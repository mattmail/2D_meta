import torch.nn as nn
import torch
import kornia as K
from utils import deform_image, field_divergence
import reproducing_kernels as rk
from scipy.stats import pearsonr
from model import UNet

class res_block(nn.Module):

    def __init__(self, h, n_in=3, n_out=1):
        super().__init__()
        self.n_in = n_in
        if n_in == 3:
            kernel_size = 3
            pad = 1
        else:
            kernel_size = 3
            pad = 1
        self.conv1 = nn.Conv2d(n_in, h, kernel_size, bias=False, padding=pad)
        self.conv2 = nn.Conv2d(h, h, kernel_size, bias=False, padding=pad)
        self.conv3 = nn.Conv2d(h, n_out, kernel_size, bias=False, padding=pad)
        self.dropout = nn.Dropout(p=0.2)
        """div_w = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])/8
        weights = torch.stack([div_w, div_w.T]).unsqueeze(0)
        self.conv1.weight = nn.Parameter(weights, requires_grad=False)"""

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, I, J):
        x = torch.cat([z,I, J], dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x

class res_block_couple(nn.Module):

    def __init__(self, h, n_in=3, n_out=1):
        super().__init__()
        self.n_in = n_in
        if n_in == 3:
            kernel_size = 3
            pad = 1
        else:
            kernel_size = 3
            pad = 1
        self.conv1 = nn.Conv2d(n_in, h, kernel_size, bias=False, padding=pad)
        self.conv2 = nn.Conv2d(h, h, kernel_size, bias=False, padding=pad)
        self.conv3 = nn.Conv2d(h, n_out, kernel_size, bias=False, padding=pad)
        self.dropout = nn.Dropout(p=0.5)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, I, J):
        x = torch.cat([z,I,J], dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x

class meta_model(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            f = self.res_list[i](self.residuals[i], image[i]) * 1/self.l
            self.residuals.append(self.residuals[i] + f)
            deformation = self.id_grid - self.field[i]/self.l
            image.append(deform_image(image[i], deformation) + self.residuals[i+1] * self.mu**2 / self.l)

        return image, self.field, self.residuals, self.grad

class meta_model_local(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, source_seg):
        """image = []
        image.append(source)"""
        image = source
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            f = self.res_list[i](self.residuals[i], image) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)

            deformation = self.id_grid - self.field[i]/self.l
            source_seg = deform_image(source_seg, deformation)
            image = deform_image(image, deformation) + self.residuals[i+1] * mu**2 / self.l * source_seg
        self.seg = source_seg
        return image, self.field, self.residuals, self.grad


class meta_model_sharp(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, target, source_seg):
        image = source.clone()
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        self.phi_list = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            deformation = self.id_grid - self.field[i] / self.l
            f = self.res_list[i](self.residuals[i], image, target) * 1 / self.l
            self.residuals.append(self.residuals[i] - f)
            if i !=0:
                self.phi_list = [deform_image(phi.permute(0,3,1,2), deformation).permute(0,2,3,1) for phi in self.phi_list]
            self.phi_list.append(deformation)
            residuals_sum = self.residuals[-1].clone()
            for k in range(len(self.residuals) - 2):
                residuals_sum += deform_image(self.residuals[k+1], self.phi_list[k+1])
            image = deform_image(source, self.phi_list[0]) + residuals_sum * mu**2 / self.l
        return image, self.field, self.residuals, self.grad

class meta_model_local_sharp(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0, requires_grad=True)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, target, source_seg):
        """image = []
        image.append(source)"""
        image = source.clone()
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        self.phi_list = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            deformation = self.id_grid - self.field[i] / self.l
            f = self.res_list[i](self.residuals[i], image, target) * 1 / self.l
            self.residuals.append(self.residuals[i] - f)
            if i !=0:
                self.phi_list = [deform_image(phi.permute(0,3,1,2), deformation).permute(0,2,3,1) for phi in self.phi_list]
            self.phi_list.append(deformation)
            residuals_sum = self.residuals[-1].clone()
            for k in range(len(self.residuals) - 2):
                residuals_sum += deform_image(self.residuals[k+1], self.phi_list[k+1])

            mask = deform_image(source_seg, self.phi_list[0])
            image = deform_image(source, self.phi_list[0]) + residuals_sum * mu**2 / self.l * mask
        self.seg = mask
        return image, self.field, self.residuals, self.grad

class shooting_model(nn.Module):
    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.conv1 = nn.Conv2d(2, 1, 3, bias=False, padding=1)
        div_w = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]) / 8
        weights = torch.stack([div_w, div_w.T]).unsqueeze(0)

        self.conv1.weight = nn.Parameter(weights, requires_grad=False)
        self.mu = mu
        self.unet = UNet()

    def forward(self, source, target, source_seg):

        z0 = self.unet(source, source_seg, target)
        return self.shooting(source, z0, source_seg)


    def shooting(self, source, z0, source_seg):
        """image = []
        image.append(source)"""
        image = source.clone()
        self.residuals = []
        self.residuals.append(z0)
        self.field = []
        self.grad = []
        self.phi_list = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            f = self.div(self.residuals[i] * self.field[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] - f)
            self.field[i] = self.field[i].permute(0, 2, 3, 1)
            deformation = self.id_grid - self.field[i] / self.l
            if i !=0:
                self.phi_list = [deform_image(phi.permute(0,3,1,2), deformation).permute(0,2,3,1) for phi in self.phi_list]
            self.phi_list.append(deformation)
            residuals_sum = self.residuals[-1].clone()
            for k in range(len(self.residuals) - 2):
                residuals_sum += deform_image(self.residuals[k+1], self.phi_list[k+1])

            mask = deform_image(source_seg, self.phi_list[0])
            image = deform_image(source, self.phi_list[0]) + residuals_sum * mu**2 / self.l * mask
        self.seg = mask
        return image, self.field, self.residuals, self.grad

    def div(self, x):
        return self.conv1(x)

class metamorphoses(nn.Module):
    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.conv1 = nn.Conv2d(2, 1, 3, bias=False, padding=1)
        div_w = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]) / 8
        weights = torch.stack([div_w, div_w.T]).unsqueeze(0)

        self.conv1.weight = nn.Parameter(weights, requires_grad=False)
        self.mu = mu
        self.z0 = nn.Parameter(z0, requires_grad=True)

    def forward(self, source, target, source_seg):

        return self.shooting(source, self.z0, source_seg)


    def shooting(self, source, z0, source_seg):
        """image = []
        image.append(source)"""
        image = source.clone()
        self.residuals = []
        self.residuals.append(z0)
        self.field = []
        self.grad = []
        self.phi_list = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            f = self.div(self.residuals[i] * self.field[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] - f)
            self.field[i] = self.field[i].permute(0, 2, 3, 1)
            deformation = self.id_grid - self.field[i] / self.l
            if i !=0:
                self.phi_list = [deform_image(phi.permute(0,3,1,2), deformation).permute(0,2,3,1) for phi in self.phi_list]
            self.phi_list.append(deformation)
            residuals_sum = self.residuals[-1].clone()
            for k in range(len(self.residuals) - 2):
                residuals_sum += deform_image(self.residuals[k+1], self.phi_list[k+1])

            mask = deform_image(source_seg, self.phi_list[0])
            image = deform_image(source, self.phi_list[0]) + residuals_sum * mu**2 / self.l * mask
        self.seg = mask
        return image, self.field, self.residuals, self.grad

    def div(self, x):
        return self.conv1(x)


class meta_couple(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block_couple(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0, requires_grad=True)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[1], im_shape[2], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')

    def forward(self, source, target):
        """image = []
        image.append(source)"""
        image = source.clone()
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        self.phi = [torch.cat([self.id_grid for _ in range(source.shape[0])])]
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            deformation = self.id_grid - self.field[i] / self.l
            f = self.res_list[i](self.residuals[i], image, target) * 1 / self.l
            self.residuals.append(self.residuals[i] - f)
            self.phi.append(deform_image(self.phi[-1].permute(0,3,1,2), deformation).permute(0,2,3,1))
            image = deform_image(source, self.phi[i+1])

        return image, self.field, self.residuals, self.grad


class double_resnet(nn.Module):

    def __init__(self, l, im_shape, device, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.z_list = []
        self.v_list = []
        self.device = device
        for i in range(l):
            self.z_list.append(res_block(h))
            self.v_list.append(res_block(h, 3, 2))
        self.z_list = nn.ModuleList(self.z_list)
        self.v_list = nn.ModuleList(self.v_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.mu = mu
        self.kernel = K.filters.GaussianBlur2d((51, 51),
                                               (10, 10),
                                               border_type='constant')

    def forward(self, source, source_seg):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        residuals_deformed = [self.residuals[0]]
        deformation = self.id_grid.clone()
        mu = self.mu
        self.field = []
        self.grad = [K.filters.SpatialGradient(mode='sobel')(source)]
        for i in range(self.l):
            f = self.z_list[i](self.residuals[i], image[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)
            residuals_deformed = [deform_image(z, deformation) for z in residuals_deformed]
            residuals_deformed.append(self.residuals[-1])
            self.field.append(self.kernel(self.v_list[i](deformation, image[-1])))
            deformation = deformation - self.field[i] / self.l
            mask = deform_image(source_seg, deformation)
            image.append(deform_image(image[0], deformation) + sum(residuals_deformed[1:]) * mu**2 / self.l * mask)
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i+1])
            self.grad.append(grad_image)

        self.seg = mask
        return image, self.field, self.residuals, self.grad
