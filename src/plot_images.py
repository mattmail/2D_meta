import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

dir = '/home/matthis/datasets/MICCAI_BraTS_2018_Data_Training/HGG/'
'Brats18_CBICA_ATB_1'
i = 0
for image in os.listdir(dir):
    if image[:5] == "Brats":
        if i >= 80:
            #im = np.transpose(np.load(dir + image))
            #seg = np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_seg.npy"))
            #seg[seg != 0] = 1
            #seg = 1 - seg
            im = nib.load(dir + image + "/" + image + "_t1.nii.gz").get_fdata()
            im = (im - im.min()) / (im.max() - im.min())
            plt.imshow(im[:,:,80].transpose(), cmap='gray', vmin=0, vmax=1)
            plt.title(image)
            plt.axis('off')
            plt.show()
            if i >= 100:
                break
        i+=1
