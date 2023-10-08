import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from prepare_data import C_Dataset
from utils import deform_image, make_grid

device = "cuda" if torch.cuda.is_available() else "cpu"

test_set = C_Dataset(device, "test_split.txt", mode="test")
target_img = test_set.target.to(device)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)

def get_contours(label):
    laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).unsqueeze(0).unsqueeze(0).float()
    contours = F.conv2d(label, laplacian, padding=1)
    contours = 1. * (contours != 0)
    return contours

def grayscale_to_rgb(grayscale):
    return torch.stack([grayscale, grayscale, grayscale], dim=-1)

def overlay_contours(image, contours):
    image[contours == 1.] = torch.tensor([1., 0., 0.])
    return image

model = torch.load("../results/meta_model_1122_1518/model.pt", map_location=device)
#model = torch.load("../results/meta_model_1123_1020/model.pt", map_location=device)

with torch.no_grad():
    model.eval()
    for i, batch in enumerate(tqdm(test_loader)):
        source_img = batch[0].to(device)
        source_seg = batch[1].to(device)
        source_def = batch[2].to(device)
        name = batch[3][0]
        source_deformed, fields, residuals, _ = model(source_img, target_img, source_seg)
        deformed_only = deform_image(source_img, model.phi_list[0])
        mask_template_space = model.seg

        plt.figure()
        plot_image = grayscale_to_rgb(source_img.detach().cpu())
        #contours = get_contours(source_seg.detach().cpu())
        #plot_image = overlay_contours(plot_image, contours)
        plt.imshow(plot_image.squeeze())
        plt.axis('off')
        #plt.title(name)
        plt.savefig("../results/images/source_" + name + ".png", bbox_inches='tight', pad_inches=0)
        plt.show()

        plt.figure()
        plot_image = grayscale_to_rgb(deformed_only.detach().cpu())
        #contours = get_contours((mask_template_space.detach().cpu() > 0.5) * 1.)
        #plot_image = overlay_contours(plot_image, contours)
        plt.imshow(plot_image.squeeze())
        plt.axis('off')
        #plt.title(name)
        plt.savefig("../results/images/deformation_" + name + ".png", bbox_inches='tight',pad_inches=0)
        plt.show()
        plt.figure()

        plot_image = grayscale_to_rgb(source_deformed.detach().cpu())
        #contours = get_contours((mask_template_space.detach().cpu() > 0.5) * 1.)
        #plot_image = overlay_contours(plot_image, contours)
        plt.imshow(plot_image.squeeze())
        plt.axis('off')
        #plt.title(name)
        plt.savefig("../results/images/meta_" + name + ".png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.figure()

        plt.imshow(target_img.squeeze().detach().cpu(), cmap='gray', vmin=0., vmax=1.)
        plt.axis('off')
        #plt.title(name)
        plt.savefig("../results/images/target_" + name + ".png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.figure()
        fig, ax = plt.subplots(1,4)
        grid = make_grid(source_img.shape, 12).to(device)
        ax[0].imshow(grid.squeeze().detach().cpu(), cmap='gray', vmin=0., vmax=1.)
        ax[0].axis('off')
        grid_forward = deform_image(grid, source_def, interpolation="bilinear")
        grid_backward = deform_image(grid, model.phi_list[0])
        grid_identity = deform_image(grid_forward, model.phi_list[0], interpolation="bilinear")
        ax[1].imshow(grid_forward.squeeze().detach().cpu(), cmap='gray', vmin=0., vmax=1.)
        ax[1].axis('off')
        ax[2].imshow(grid_backward.squeeze().detach().cpu(), cmap='gray', vmin=0., vmax=1.)
        ax[2].axis('off')
        ax[3].imshow(grid_identity.squeeze().detach().cpu(), cmap='gray', vmin=0., vmax=1.)
        ax[3].axis('off')
        #plt.title(name)
        #plt.savefig("../results/images/target_" + name + ".png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.figure()
