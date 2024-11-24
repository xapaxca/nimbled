import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class KITTIDepthAnything(Dataset):
    def __init__(self, data_dir, height=518, width=518):
        self.height = height
        self.width = width
        self.frame_paths, self.save_paths = self._get_frames(data_dir)
        self.original_height = 375
        self.original_width = 1242
        
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame = cv2.imread(self.frame_paths[idx], cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to load image at path {self.frame_paths[idx]}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        frame = to_tensor(frame)
        save_path = self.save_paths[idx]
        return frame, save_path

    def _get_frame_path(self, data_dir, folder, frame_idx, side, save_dir):
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        f_str = "{:010d}{}".format(frame_idx, ".png")
        f_str_raw = "{:010d}".format(frame_idx)

        frame_dir = os.path.join(data_dir, folder, "image_0{}".format(side_map[side]), "data")
        save_dir = os.path.join(save_dir, folder, "image_0{}".format(side_map[side]), "data")

        os.makedirs(save_dir, exist_ok=True)

        frame_path = os.path.join(frame_dir, f_str)
        save_path = os.path.join(save_dir, f_str_raw)

        return frame_path, save_path
    
    def _get_frames(self, data_dir):
        frame_paths = []
        save_paths = []

        save_dir = os.path.join(data_dir, "pseudo_depths_with_max")

        os.makedirs(save_dir, exist_ok=True)

        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "kitti", "splits", "eigen_zhou", "train_files.txt")

        if os.path.isfile(split_path):
            if split_path.endswith('txt'):
                with open(split_path, 'r') as f:
                    lines = f.read().splitlines()
                    frame_paths = []
                    for line in tqdm(lines):
                        folder, frame_idx, side = line.split()
                        frame_idx = int(frame_idx)
                        frame_path, save_path = self._get_frame_path(data_dir, folder, frame_idx, side, save_dir)
                        frame_paths.append(frame_path)
                        save_paths.append(save_path)

        return frame_paths, save_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--compression', type=int, default=1, help='PNG compression level from 0 (no compression) to 9 (maximum compression)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
    model.to(device)
    model.eval()
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")

    dataset = KITTIDepthAnything(data_dir=args.data_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    for batch in tqdm(dataloader):   
        frames, save_paths = batch
        frames = frames.to(device)

        with torch.inference_mode():
            frames = image_processor(images=frames, do_rescale=False, return_tensors="pt").to(device)
            pseudo_disps = model(**frames).predicted_depth.unsqueeze(1)
            pseudo_disps = F.interpolate(pseudo_disps, size=(dataset.original_height, dataset.original_width), mode="bilinear", align_corners=False)
            max_val = pseudo_disps.max().item()
            pseudo_disps = (pseudo_disps / max_val) * 65535
            pseudo_disps = pseudo_disps.cpu()[:, 0].numpy().astype(np.uint16)
        
        for i, save_path in enumerate(save_paths):
            success = cv2.imwrite(save_path + ".png", pseudo_disps[i], [cv2.IMWRITE_PNG_COMPRESSION, args.compression])
            if not success:
                print(f"The disp could not be saved at the following path: {save_path}.png")
            with open(save_path + ".txt", 'w') as f:
                f.write(str(max_val))
