import torch
import kornia as K
import matplotlib.pyplot as plt

from utils import deform_image, make_grid, integrate_stationnary
from reproducing_kernels import GaussianRKHS2d

def compose_fields(u,v):
    return deform_image(u.permute(0,3,1,2), v).permute(0,2,3,1)

image_shape = (1, 1, 200,200)
v_shape = (1, 2, 200,200)
kernel = K.filters.GaussianBlur2d((31,31),
                                                 (20, 20),
                                                 border_type='constant')

grid = make_grid(image_shape, 10)
id_grid = K.utils.grid.create_meshgrid(image_shape[2], image_shape[3], False, "cpu")
u_big = kernel(torch.rand(v_shape)*500-250).permute(0,2,3,1)
base_tensor = torch.zeros(v_shape)
x_v, y_v = 50,50
base_tensor[:,0,50:100,50:100] = x_v
base_tensor[:,1,50:100,50:100] = 0
base_tensor[:,0,50:100,100:150] =  - x_v
base_tensor[:,1,50:100,100:150] = 0
base_tensor[:,0,100:150,50:100] =  0
base_tensor[:,1,100:150,50:100] =  - y_v
base_tensor[:,0,100:150,100:150] = 0
base_tensor[:,1,100:150,100:150] =  - y_v
u_small = kernel(kernel(kernel(base_tensor))).permute(0,2,3,1)
base_tensor = torch.zeros(v_shape)
x_v, y_v = 450,800
base_tensor[:,0,50:60,50:60] = x_v
base_tensor[:,1,50:60,50:60] = y_v
base_tensor[:,0,150:160,50:60] = x_v
base_tensor[:,1,150:160,50:60] = y_v
base_tensor[:,0,50:60,150:160] = x_v
base_tensor[:,1,50:60,150:160] = y_v
base_tensor[:,0,150:160,150:160] = x_v
base_tensor[:,1,150:160,150:160] = y_v
u_big = kernel(kernel(kernel(kernel(kernel(kernel(base_tensor)))))).permute(0,2,3,1)
diff_pos = integrate_stationnary(-u_big)
diff_neg = integrate_stationnary(u_big)

big_pos = deform_image(grid, id_grid + u_big, interpolation="bicubic")
big_neg = deform_image(grid, compose_fields(id_grid + u_big, id_grid - u_big), interpolation="bicubic")
small_pos = deform_image(grid, id_grid + u_small, interpolation="bicubic")
small_neg = deform_image(grid, compose_fields(id_grid + u_small, id_grid - u_small), interpolation="bicubic")
im_pos = deform_image(grid, diff_pos, interpolation="bicubic")
im_neg = deform_image(grid, compose_fields(diff_pos, diff_neg), interpolation="bicubic")

fig, ax = plt.subplots(1,2)
ax[0].imshow(small_pos.squeeze(), cmap="gray", vmin=0., vmax=1.)
ax[0].axis("off")
ax[1].imshow(small_neg.squeeze(), cmap="gray", vmin=0., vmax=1.)
ax[1].axis("off")
plt.show()

plt.imshow(big_pos.squeeze(), cmap="gray", vmin=0., vmax=1.)
plt.axis("off")
plt.savefig('/home/matthis/Images/big_pos.png',bbox_inches='tight', pad_inches=0.0)
plt.imshow(big_neg.squeeze(), cmap="gray", vmin=0., vmax=1.)
plt.axis("off")
plt.savefig('/home/matthis/Images/big_neg.png',bbox_inches='tight', pad_inches=0.0)
