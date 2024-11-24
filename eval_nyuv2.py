import numpy as np
import cv2
import os
import argparse
import torch
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.resnet_encoder import ResnetEncoder
from networks.monodepth_decoder import MonoDepthDecoderReLU
from networks.swiftdepth_encoder import SwiftFormer_XS, SwiftFormer_S
from networks.swiftdepth_decoder import DepthDecoderReLU
from networks.litemono_encoder import LiteMono
from networks.litemono_decoder import LiteMonoDecoderReLU
from datasets.nyuv2.nyuv2_dataset import NYUv2
from utils import *


warnings.filterwarnings("ignore", category=UserWarning)


def eval(args):
    # Prepare dataset
    dataset = NYUv2(root_dir=args.data_dir)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers, 
                            pin_memory=True)

    # Prepare trained modelss
    if args.model_name == 'md2_r18':
        depth_encoder = ResnetEncoder(18, False)
        depth_decoder = MonoDepthDecoderReLU(depth_encoder.num_ch_enc)
    elif args.model_name == 'md2_r50':
        depth_encoder = ResnetEncoder(50, False)
        depth_decoder = MonoDepthDecoderReLU(depth_encoder.num_ch_enc)
    elif args.model_name == 'swiftdepth_s':
        depth_encoder = SwiftFormer_XS()
        depth_decoder = DepthDecoderReLU(num_ch_enc=depth_encoder.embed_dims, enc_name="SwiftFormer_XS")
    elif args.model_name == 'swiftdepth':
        depth_encoder = SwiftFormer_S()
        depth_decoder = DepthDecoderReLU(num_ch_enc=depth_encoder.embed_dims, enc_name="SwiftFormer_S")
    elif args.model_name == 'litemono_s':
        depth_encoder = LiteMono(model="lite-mono-small")
        depth_decoder = LiteMonoDecoderReLU(depth_encoder.num_ch_enc, scales=range(3))
    elif args.model_name == 'litemono':
        depth_encoder = LiteMono(model="lite-mono")
        depth_decoder = LiteMonoDecoderReLU(depth_encoder.num_ch_enc, scales=range(3))
    elif args.model_name == 'litemono_8m':
        depth_encoder = LiteMono(model="lite-mono-8m")  
        depth_decoder = LiteMonoDecoderReLU(depth_encoder.num_ch_enc, scales=range(3))
    else:
        raise ValueError('Invalid model name')

    assert os.path.isdir(args.weights_dir), f"Cannot find weights_dir at {args.weights_dir}"

    encoder_path = os.path.join(args.weights_dir, "depth_encoder.pth")
    decoder_path = os.path.join(args.weights_dir, "depth_decoder.pth")

    depth_encoder = load_weights(depth_encoder, encoder_path)
    depth_decoder = load_weights(depth_decoder, decoder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depth_encoder.to(device)
    depth_decoder.to(device)

    depth_encoder.eval()
    depth_decoder.eval()

    dataset_size = len(dataset)

    print(f"Evaluating on NYUv2")
    print(f"\tSize of the set: {dataset_size}")
    print(f"\tAlignment: {args.align}")

    pred_disps = []
    gt_depths = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            # read ground truth
            gt_depth = batch["depth"][:, 0].numpy()
            gt_depths.append(gt_depth)

            frame = batch["image"].to(device)
            frame = torch.nn.functional.interpolate(frame, size=(288, 384), mode="bilinear", align_corners=False)

            pred_disp = depth_decoder(depth_encoder(frame))[("disp", 0)]
            pred_disp = pred_disp.data.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)
    gt_depths = np.concatenate(gt_depths)

    metrics_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    metrics = np.zeros(len(metrics_names))

    print(f"Size of predicted disparities: {pred_disps.shape[-2:]}")

    for i in range(dataset_size):
        # get ground truth depths
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        # get predicted depths
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = to_inv(pred_disp)

        # min-max mask for ground truth
        mask = (gt_depth > args.min_depth) & (gt_depth < args.max_depth)
        # crop
        crop_mask = np.zeros_like(mask, dtype=bool)
        crop_mask[45:471, 41:601] = True
        # combine crop and min-max masks
        mask = mask & crop_mask
   
        # mask pred and ground truth depths
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # align predictions to ground truth
        if args.align == "median":
            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio
        else:
            disp, gt_disp = to_inv(pred_depth), to_inv(gt_depth)
            scale, shift = align_lsqr(disp, gt_disp)
            disp = disp * scale + shift
            pred_depth = to_inv(disp)

        # clip predicted values to be within min-max range 
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        metrics += compute_eigen_metrics(pred_depth, gt_depth)

    metrics /= dataset_size

    metrics_string = ""

    for metrics_name in metrics_names:
        metrics_string += f"{metrics_name:<{10}}"
    metrics_string += "\n"  
    for metric in metrics:
        metrics_string += f"{f'{metric:.3f}':<{10}}"

    return metrics_string


def eval_nyuv2(data_dir, 
               weights_dir, 
               model_name, 
               align, 
               batch_size=16, 
               num_workers=8, 
               min_depth=1e-3, 
               max_depth=10.0):
    args = argparse.Namespace(
        data_dir=data_dir,
        weights_dir=weights_dir,
        model_name=model_name,
        align=align,
        batch_size=batch_size,
        num_workers=num_workers,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on NYUv2 dataset")
    parser.add_argument("--data_dir", 
                        type=str, 
                        help="Path to the KITTI data directory with .png files",
                        required=True)
    parser.add_argument("--weights_dir", 
                        type=str, 
                        help="Path to the weight directory",
                        required=True)
    parser.add_argument("--model_name", 
                        type=str, 
                        choices=[
                            "nimbled_md2_r18",
                            "nimbled_md2_r50",
                            "nimbled_swiftdepth_s", 
                            "nimbled_swiftdepth",
                            "nimbled_litemono_s",
                            "nimbled_litemono",
                            "nimbled_litemono_8m",
                        ],
                        required=True,
                        help="Name of the depth model to evaluate")
    parser.add_argument('--align', 
                    type=str,
                    choices=[
                        "median", 
                        "lsqr"
                    ], 
                    required=True,
                    help="Median or least squares alignment")
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=16, 
                        help='Batch size')
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=8, 
                        help='Number of workers')
    parser.add_argument('--min_depth', 
                        type=float, 
                        default=1e-3, 
                        help='Minimum depth')
    parser.add_argument('--max_depth',
                        type=float, 
                        default=10.0, 
                        help='Maximum depth')
    
    args = parser.parse_args()
    nimbled_to_model_name = {"nimbled_md2_r18": "md2_r18",
                             "nimbled_md2_r50": "md2_r50",
                             "nimbled_swiftdepth_s": "swiftdepth_s",
                             "nimbled_swiftdepth": "swiftdepth",
                             "nimbled_litemono_s": "litemono_s",
                             "nimbled_litemono": "litemono",
                             "nimbled_litemono_8m": "litemono_8m",
                             }
    args.model_name = nimbled_to_model_name[args.model_name]
    results = eval(args)
    print(results)
