import SimpleITK as sitk
import sys
import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt


"""inputImage = sitk.ReadImage("/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_00008_T1.nii.gz", sitk.sitkFloat32)
image = inputImage

maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

corrector = sitk.N4BiasFieldCorrectionImageFilter()

numberFittingLevels = 4

if len(sys.argv) > 6:
    numberFittingLevels = int(sys.argv[6])

if len(sys.argv) > 5:
    corrector.SetMaximumNumberOfIterations([int(sys.argv[5])]
                                           * numberFittingLevels)

corrected_image = corrector.Execute(image, maskImage)
log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
bias_field = inputImage / sitk.Exp( log_bias_field )

sitk.WriteImage(corrected_image, "/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_00008_T1_corrected.nii.gz")

MNI_nib = nib.load("/Users/maillard/Downloads/sri24_spm8/templates/T1_brain.nii")
MNI_img = MNI_nib.get_fdata()

source = nib.load("/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_00008_T1_corrected.nii.gz")
source_img = source.get_fdata()
print(source_img.max(), source_img.min())

source_img[source_img > 0] = match_histograms(source_img[source_img > 0], MNI_img[MNI_img > 0])
source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
MNI_img = (MNI_img - MNI_img.min()) / (MNI_img.max() - MNI_img.min())
#source_img = source_img[:, ::-1, :]

MNI = nib.Nifti1Image(MNI_img, MNI_nib.affine)
nib.save(MNI, "/Users/maillard/Downloads/sri24_spm8/templates/T1_brain_scaled.nii")

source = nib.Nifti1Image(source_img, MNI_nib.affine)
nib.save(source, "/Users/maillard/Downloads/sri24_spm8/templates/brats_image_test_corrected.nii")"""

data_dir = "/home/matthis/datasets/BraTS_2021/"
save_dir = "/home/matthis/datasets/BraTS_2021_2D/"

for file in os.listdir(data_dir):
    seg = nib.load(data_dir + file + "/" + file + "_seg.nii.gz").get_fdata()[:,:,80]
    if not os.path.exists(save_dir + file):
        os.mkdir(save_dir + file)
    np.save(save_dir + file + "/" + file + "_seg.npy", seg)
    os.system("mv " + save_dir + file + ".npy " + save_dir + file + "/" + file + "_t1.npy")

"""csv_file = "/home/matthis/brats_2021_train.csv"
dir = "/home/matthis/datasets/BraTS_2021/"
save_dir = "/home/matthis/brats_seg_test/"

f = open(csv_file, "r")
file_list = list(f.readlines())
print(file_list)

for file in file_list:
    file = file[:-1]
    print(file)
    if not file == 'folder_name':
            if int(file[-3:]) > 100:
                print('run_samseg -i '+ dir + file + "/" + file + "_t1.nii.gz -o " + save_dir + file + " --threads 15")
                os.system('run_samseg -i '+ dir + file + "/" + file + "_t1.nii.gz -o " + save_dir + file + " --threads 15")
"""

