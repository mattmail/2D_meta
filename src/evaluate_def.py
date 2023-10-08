from prepare_data import C_Dataset
import kornia as K
import torch
from tqdm import tqdm
import numpy as np

test_set = C_Dataset("cpu", "test_split.txt", mode="test")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)

def_dist = []
#dice_C = []
num_folds = []
#prob_map_seg = torch.zeros(target.shape)
#probs_count = 0
for i, batch in enumerate(tqdm(test_loader)):
    source_def = batch[2]
    id_grid = K.utils.grid.create_meshgrid(source_def.shape[1], source_def.shape[2], False, "cpu")
    def_dist.append(torch.sqrt(torch.sum((id_grid - source_def.detach().cpu()) ** 2)))

print("Deformation difference:", sum(def_dist).item() / len(def_dist), "std: %f" % (np.array(def_dist).std()))

