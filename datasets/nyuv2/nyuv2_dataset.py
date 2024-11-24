import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import os


class NYUv2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.frame_dir = os.path.join(root_dir, 'frame')
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.npy')])
        self.frame_files = sorted([f for f in os.listdir(self.frame_dir) if f.endswith('.png')])
        assert len(self.depth_files) == len(self.frame_files), "Depth and frame counts do not match"

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        frame_path = os.path.join(self.frame_dir, self.frame_files[idx])
        depth = np.load(depth_path)
        depth_tensor = torch.from_numpy(depth).float().permute(2, 0, 1)
        image = Image.open(frame_path).convert('RGB')
        image = to_tensor(image)

        return {'image': image, 'depth': depth_tensor}
