import numpy as np
import torch
import nibabel as nib
import os
import cv2
import random
from torch.utils.data import Dataset

def load_brats_2021_linear(device, use_segmentation,chose_randomly=False):
    test_list = []
    if not chose_randomly:
        with open("test_list.txt", "r") as f:
            test_paths = f.readlines()
            test_paths = [path[:-1] for path in test_paths]
            f.close()
    else:
        test_paths = []
    source_list = []
    for image in os.listdir('../data_miccai_2D_2021/'):
        if image[:5] == "BraTS":
            if image not in test_paths:
                if use_segmentation:
                    source_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
                    source_seg[source_seg == 2.] = 1.
                    source_seg[source_seg == 4.] = 1.
                    source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    source_list.append(torch.cat([torch.from_numpy(
                        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
                else:
                    source_list.append(torch.from_numpy(
                        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0))
            else:
                if use_segmentation:
                    test_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
                    test_seg[test_seg == 2.] = 1.
                    test_seg[test_seg == 4.] = 1.
                    test_seg = torch.from_numpy(test_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    test_list.append(torch.cat([torch.from_numpy(
                        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0).unsqueeze(0), test_seg]))
                else:
                    test_list.append(torch.from_numpy(
                        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0))

    if chose_randomly:
        random.shuffle(source_list)
        test_list = source_list[-40:]
        source_list = source_list[:-40]

    MNI_img = nib.load("/usr/local/fsl/data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz").get_fdata()
    target_img = np.pad(MNI_img[:, :, 93], ((18, 18), (0, 0)), "constant")
    target_img = np.rot90(cv2.resize(target_img, (208, 208)))
    target_img = torch.from_numpy(target_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device) / 255.
    return torch.stack(source_list), torch.stack(test_list), target_img, None

def load_brats_2021(device, use_segmentation,chose_randomly=False):
    test_list = []
    if not chose_randomly:
        with open("/home/matthis/Nextcloud/voxelmorph/data/brats_2021_train.csv", "r") as f:
            test_paths = f.readlines()
            test_paths = [path[:-1] for path in test_paths]
            f.close()
    else:
        test_paths = []
    source_list = []
    data_dir = "/home/matthis/datasets/BraTS_2021_2D/"
    test_list_names = []
    for image in os.listdir(data_dir):
        if image[:5] == "BraTS":
            if image not in test_paths:
                if use_segmentation:
                    source_seg = np.transpose(np.load(data_dir + image + "/" + image + "_seg.npy"))
                    source_seg[source_seg == 2.] = 1.
                    source_seg[source_seg == 4.] = 1.
                    source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    source_list.append(torch.cat([torch.from_numpy(
                        np.transpose(np.load(data_dir + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
                else:
                    source_list.append(torch.from_numpy(
                        np.transpose(np.load(data_dir + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0))
            else:
                test_list_names.append(image)
                if use_segmentation:
                    test_seg = np.transpose(np.load(data_dir + image + "/" + image + "_seg.npy"))
                    test_seg[test_seg == 2.] = 1.
                    test_seg[test_seg == 4.] = 1.
                    test_seg = torch.from_numpy(test_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    test_list.append(torch.cat([torch.from_numpy(
                        np.transpose(np.load(data_dir + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0).unsqueeze(0), test_seg]))
                else:
                    test_list.append(torch.from_numpy(
                        np.transpose(np.load(data_dir + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0))

    if chose_randomly:
        random.shuffle(source_list)
        test_list = source_list[-200:]
        source_list = source_list[:-200]
    print("Number of training images:", len(source_list))
    print("Number of test images:", len(test_list))

    target_img = np.transpose(np.load("/home/matthis/datasets/sri24_t1_preprocessed.npy").squeeze())
    target_img = target_img[np.newaxis,np.newaxis, ...].copy()
    target_img = torch.from_numpy(target_img).float()
    return torch.stack(source_list), torch.stack(test_list), target_img, test_list_names

def load_brats_2020(device, use_segmentation, test_size=40):
    target_img = torch.from_numpy(
        np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(
        torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_list = []
    for image in os.listdir('../brats_2020_2D/cancerous'):
        if image[:5] == "BraTS":
            if use_segmentation:
                source_seg = np.transpose(np.load("../brats_2020_2D/cancerous/" + image + "/" + image + "_seg.npy")).astype(float)
                source_seg[source_seg == 2.] = 1.
                source_seg[source_seg == 4.] = 1.
                source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                source_list.append(torch.cat([torch.from_numpy(
                    np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(
                    torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
            else:
                source_list.append(torch.from_numpy(
                    np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(
                    torch.FloatTensor).unsqueeze(0))

    random.shuffle(source_list)
    test_list = source_list[-test_size:]
    source_list = source_list[:-test_size]

    return torch.stack(source_list), torch.stack(test_list), target_img

def load_C_segmented(device, test_file=None, test_size=40):
    target = torch.tensor(cv2.resize(cv2.imread("../images/reg_test_m0.png", 0), (200, 200)) / 255.).unsqueeze(0).unsqueeze(0).float()
    if test_file is not None:
        with open(test_file, "r") as f:
            test_images = f.readlines()
            test_images = [path[:-1] for path in test_images]
            f.close()
    else:
        test_images = []
    path = "../images_seg/"
    test_list = []
    train_list = []
    train_list_names = []
    test_names = []
    for file in os.listdir(path):
        if file in test_images:
            source_img = torch.tensor(np.load(path+file+"/image.npy")).unsqueeze(0).float()
            source_seg = torch.tensor(np.load(path+file+"/seg.npy")).unsqueeze(0).float()
            source_def = torch.tensor(np.load(path+file+"/deformation.npy")).float()
            test_list.append([source_img, source_seg, source_def])
            test_names.append(file)

        else:
            source_img = torch.tensor(np.load(path + file + "/image.npy")).unsqueeze(0).float()
            source_seg = torch.tensor(np.load(path + file + "/seg.npy")).unsqueeze(0).float()
            source_def = torch.tensor(np.load(path + file + "/deformation.npy")).float()
            train_list.append([source_img, source_seg, source_def])
            train_list_names.append(file)

    if test_file is None:
        indexes = np.arange(len(train_list))
        random.shuffle(indexes)
        test_list = [train_list[i] for i in indexes[-test_size:]]
        train_list = [train_list[i] for i in indexes[:-test_size]]
        test_names = [train_list_names[i] for i in indexes[-test_size:]]
        textfile = open("test_split.txt", "w")
        for element in test_names:
            textfile.write(element + "\n")
        textfile.close()
    return train_list, test_list, target, test_names

def load_C_couple(test_file=None, test_size=40):
    if test_file is not None:
        with open(test_file, "r") as f:
            test_images = f.readlines()
            test_images = [path[:-1] for path in test_images]
            f.close()
    else:
        test_images = []
    path = "../images_couple/"
    test_list = []
    train_list = []
    train_list_names = []
    for file in os.listdir(path):
        if file in test_images:
            source_img = torch.tensor(np.load(path + file+"/image.npy")).unsqueeze(0).float()
            target_img = torch.tensor(np.load(path + file + "/target.npy")).unsqueeze(0).float()
            source_def = torch.tensor(np.load(path + file+"/deformation.npy")).float()
            test_list.append([source_img, target_img, source_def])

        else:
            source_img = torch.tensor(np.load(path + file + "/image.npy")).unsqueeze(0).float()
            target_img = torch.tensor(np.load(path + file + "/target.npy")).unsqueeze(0).float()
            source_def = torch.tensor(np.load(path + file + "/deformation.npy")).float()
            train_list.append([source_img, target_img, source_def])
            train_list_names.append(file)

    if test_file is None:
        indexes = np.arange(len(train_list))
        random.shuffle(indexes)
        test_list = [train_list[i] for i in indexes[-test_size:]]
        train_list = [train_list[i] for i in indexes[:-test_size]]
        test_names = [train_list_names[i] for i in indexes[-test_size:]]
        textfile = open("test_couple_split.txt", "w")
        for element in test_names:
            textfile.write(element + "\n")
        textfile.close()
    return train_list, test_list

class C_Dataset(Dataset):
    def __init__(self, device, test_file=None, test_size=40, mode="train"):
        super(C_Dataset, self).__init__()
        train_list, test_list, target, test_names = load_C_segmented(device, test_file, test_size)
        if mode == "train":
            self.image_list = train_list
        else:
            self.image_list = test_list
        self.target = target
        self.test_names = test_names
        self.mode = mode

    def __getitem__(self, index):
        output = self.image_list[index]
        if self.mode == "train":
            return output[0], output[1], output[2]
        else:
            return output[0], output[1], output[2], self.test_names[index]
    def __len__(self):
        return len(self.image_list)

class C_Dataset_Couple(Dataset):
    def __init__(self, device, test_file=None, test_size=500, mode="train"):
        super(C_Dataset_Couple, self).__init__()
        train_list, test_list = load_C_couple(test_file, test_size)
        if mode == "train":
            self.image_list = train_list
        else:
            self.image_list = test_list
        self.shape = train_list[0][0].shape

    def __getitem__(self, index):
        output = self.image_list[index]
        return output[0], output[1], output[2]
    def __len__(self):
        return len(self.image_list)

class Brats2021_Dataset(Dataset):
    def __init__(self, device, mode="train"):
        super(Brats2021_Dataset, self).__init__()
        train_list, test_list, target, test_list_names = load_brats_2021(device, True, False)
        if mode == "train":
            self.image_list = train_list
        else:
            self.image_list = test_list
            self.test_list_names = test_list_names
        self.target = target
        self.shape = train_list[0][0].shape

    def __getitem__(self, index):
        output = self.image_list[index]
        return output[0], output[1]
    def __len__(self):
        return len(self.image_list)





