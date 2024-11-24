import os
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
import torch
import random
from torchvision.transforms.functional import to_tensor

class YTDepthDataset(Dataset):
    def __init__(self, file_path, frames_root, disps_root, is_train, size=None, crop=None):
        self.frames_root = frames_root
        self.disps_root = disps_root
        self.size = size
        self.crop = crop
        self.is_train = is_train
        self.prefix_to_folder = {
            'CW': 'city_walking',
            'D': 'driving',
            'H': 'hiking'
        }
        self.samples = self._parse_file(file_path)

        self.flip = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ],
            additional_targets={'frame_raw_prev': 'image', 'frame_raw_next': 'image', 'disp': 'image'}
        )

        self.jitter = A.Compose(
            [
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                    p=1
                ),            
            ],
            additional_targets={'frame_raw_prev': 'image', 'frame_raw_next': 'image'}
        )

    def _parse_file(self, file_path):
        samples = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    category_sequence, frame = parts
                    prefix = category_sequence.split('_')[0]
                    samples.append((prefix, category_sequence, frame))
        return samples

    def _read_frames(self, file_paths):
        frames = []
        for file_path in file_paths:
            frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Unable to load frame at path {file_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames
    
    def _read_disp(self, file_path, max_val_path):
        disp = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise ValueError(f"Unable to load disp at path {file_path}")
        with open(max_val_path, 'r') as f:
            max_val_str = f.readline().strip()  
            max_val = float(max_val_str)
        disp = max_val * disp.astype(np.float32) / 65535.0
        return disp

    def __len__(self):
        return len(self.samples)
   
    def __getitem__(self, idx):
        # prepare path info
        prefix, category_sequence, frame = self.samples[idx]
        folder_name = self.prefix_to_folder[prefix]
        frame_num = int(frame)

        # read frames
        frame_paths = [os.path.join(self.frames_root, folder_name, category_sequence, f'{frame_num + offset:07d}.png') for offset in [0, -1, 1]]
        frame_raw_target, frame_raw_prev, frame_raw_next = self._read_frames(frame_paths)

        # read disp
        disp_path = os.path.join(self.disps_root, folder_name, category_sequence, f'{frame_num:07d}.png')
        max_val_path = os.path.join(self.disps_root, folder_name, category_sequence, f'{frame_num:07d}.txt')
        disp = self._read_disp(disp_path, max_val_path)

        # flip
        frame_raw_target, frame_raw_prev, frame_raw_next, disp = self.flip(image=frame_raw_target, frame_raw_prev=frame_raw_prev, frame_raw_next=frame_raw_next, disp=disp).values()

        frame_aug_target, frame_aug_prev, frame_aug_next = self.jitter(image=frame_raw_target, frame_raw_prev=frame_raw_prev, frame_raw_next=frame_raw_next).values()

        return frame_raw_target, frame_raw_prev, frame_raw_next, frame_aug_target, frame_aug_prev, frame_aug_next, disp#[..., np.newaxis]

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d    
 
import matplotlib.pyplot as plt

def show_images(frames_raw_target, frames_raw_prev, frames_raw_next, frames_aug_target, frames_aug_prev, frames_aug_next, disps):
    batch_size = frames_raw_target.shape[0]  # Get batch size from the shape of the tensor
    num_columns = 3 * 2 + 1  # 3 frames (previous, target, next), raw and augmented versions for each, plus 1 depth map

    fig, axs = plt.subplots(batch_size, num_columns, figsize=(21, 3 * batch_size))

    # Ensure axs is 2D for consistency when batch_size is 1
    if batch_size == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(batch_size):
        # List of frames to display (raw and augmented), along with their titles
        frames_to_display = [
            (frames_raw_prev[i], 'Raw Support Frame t-1'),
            (frames_aug_prev[i], 'Augmented Support Frame t-1'),
            (frames_raw_target[i], 'Raw Target Frame t'),
            (frames_aug_target[i], 'Augmented Target Frame t'),
            (frames_raw_next[i], 'Raw Support Frame t+1'),
            (frames_aug_next[i], 'Augmented Support Frame t+1')
        ]

        # Display raw and augmented frames
        for j, (frame, title) in enumerate(frames_to_display):
            # Use .permute(1, 2, 0) to change tensor shape from (C, H, W) to (H, W, C) for display
            axs[i, j].imshow(frame.permute(1, 2, 0).cpu().numpy())  # Move tensor to CPU, permute dimensions, and convert to NumPy array for display
            axs[i, j].set_title(title)
            axs[i, j].axis('off')

        # Normalize and display the depth map in the last column
        disp_normalized = normalize_image(disps[i]).squeeze().cpu().numpy()  # Remove channel dimension and normalize depth map
        axs[i, -1].imshow(disp_normalized, cmap='gray')
        axs[i, -1].set_title('Target Frame Pseudo Depth')
        axs[i, -1].axis('off')

    plt.tight_layout()
    plt.show()


ratio_crop_resolution = {
    "4:3": ((480, 640), (288, 384)),
    "3:2": ((480, 720), (320, 480)),
    "16:9": ((480, 854), (288, 512)),
    "1:1": ((480, 480), (384, 384)),
    "3:4": ((480, 360), (384, 288)),
    "2:3": ((480, 320), (480, 320)),
    "9:16": ((480, 270), (512, 288)),
    "10:3": ((256, 854), (192, 640))
}


def collate_fn(batch):
    batch = list(zip(*batch))

    frames = batch[:6]
    disps = batch[6]

    aspect_ratio, sizes = random.choice(list(ratio_crop_resolution.items()))
    crop_size, resize_size = sizes
    crop_height, crop_width= crop_size[0], crop_size[1]
    img_height, img_width= resize_size[0], resize_size[1]


    crop_resize = A.Compose([
        A.CenterCrop(height=crop_height, width=crop_width),  # Center crop
        A.Resize(height=img_height, width=img_width),  # Resize
        ])

    output = []

    for frame_idx in range(len(frames)):
        frame_batch = []
        for batch_idx in range(len(frames[0])):
            frame = frames[frame_idx][batch_idx]
            transformed_frame = crop_resize(image=frame)["image"]
            transformed_frame_tensor = to_tensor(transformed_frame)
            frame_batch.append(transformed_frame_tensor)
        output.append(torch.stack(frame_batch))

    disps_transformed = []
    for batch_idx in range(len(disps)):
        disp = disps[batch_idx]
        transformed_disp = crop_resize(image=disp)["image"]
        transformed_disp_tensor = torch.tensor(transformed_disp, dtype=torch.float32).unsqueeze(0)
        disps_transformed.append(transformed_disp_tensor)
    output.append(torch.stack(disps_transformed))      

    return (*output, img_height, img_width)
