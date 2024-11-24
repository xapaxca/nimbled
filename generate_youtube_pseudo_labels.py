import argparse
import cv2
import os
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from torchvision.transforms.functional import to_tensor


class YTDepthGeneratorDataset(Dataset):
    def __init__(self, data_dir: str, height=518, width=518):
        self.height = height
        self.width = width
        self.data_dir = data_dir
        self.image_paths: List[str] = []
        self.original_height = 480
        self.original_width = 854

        self.prefix_to_folder: Dict[str, str] = {
            'CW': 'city_walking',
            'D': 'driving',
            'H': 'hiking'
        }

        self._read_frames_file()

    def _read_frames_file(self) -> None:
        frames_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "youtube", "splits", "all_frames.txt")
        frames_dir = os.path.join(self.data_dir, 'frames')

        with open(frames_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    prefix, frame = parts
                    folder = self.prefix_to_folder.get(prefix.split("_")[0])
                    if folder:
                        img_path = os.path.join(frames_dir, folder, prefix, f"{frame}.png")
                        self.image_paths.append(img_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple[str, str, str]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise ValueError(f"Unable to load image at path {img_path}")
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = to_tensor(image)

        path_parts = img_path.split(os.sep)
        prefix = path_parts[-2]
        folder = self.prefix_to_folder.get(prefix.split("_")[0])
        frame = os.path.splitext(path_parts[-1])[0]

        return image, (folder, prefix, frame)

def save_depth_map(pseudo_disp, out_disp_dir, out_disp_path, compression, max_val, out_max_path):
    os.makedirs(out_disp_dir, exist_ok=True)
    success = cv2.imwrite(out_disp_path, pseudo_disp, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not success:
        print(f"Failed to write {out_disp_path}")
    with open(out_max_path, 'w') as f:
        f.write(str(max_val))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--compression', type=int, default=1, help='PNG compression level from 0 to 9')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    save_dir = os.path.join(args.data_dir, 'disps')

    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
    model.to(device)
    model.eval()
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")

    dataset = YTDepthGeneratorDataset(data_dir=args.data_dir)
    print(f"Total number of images to process: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    for images, path_info in tqdm(dataloader, desc=f"Processing image batches of size {args.batch_size}"):
        images = images.to(device)

        with torch.inference_mode():
            images = image_processor(images=images, do_rescale=False, return_tensors="pt").to(device)
            pseudo_disps = model(**images).predicted_depth.unsqueeze(1)
            pseudo_disps = F.interpolate(pseudo_disps, size=(dataset.original_height, dataset.original_width), mode="bilinear", align_corners=False)
            max_val = pseudo_disps.max().item()
            pseudo_disps = (pseudo_disps / max_val) * 65535
            pseudo_disps = pseudo_disps.cpu()[:, 0].numpy().astype(np.uint16)

        for i, pseudo_disp in enumerate(pseudo_disps):
            out_disp_dir = os.path.join(save_dir, path_info[0][i], path_info[1][i])
            out_disp_path = os.path.join(out_disp_dir, f"{path_info[2][i]}.png")
            out_max_path = os.path.join(out_disp_dir, f"{path_info[2][i]}.txt")
            save_depth_map(pseudo_disp, out_disp_dir, out_disp_path, args.compression, max_val, out_max_path)

if __name__ == '__main__':
    main()
