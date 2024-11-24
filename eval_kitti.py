import os
import cv2
import numpy as np
import argparse
import torch
import warnings
from torch.utils.data import DataLoader

from networks.resnet_encoder import ResnetEncoder
from networks.monodepth_decoder import MonoDepthDecoderReLU
from networks.swiftdepth_encoder import SwiftFormer_XS, SwiftFormer_S
from networks.swiftdepth_decoder import DepthDecoderReLU
from networks.litemono_encoder import LiteMono
from networks.litemono_decoder import LiteMonoDecoderReLU
from datasets.kitti.kitti_dataset import KITTIRAWDataset
from datasets.kitti.kitti_utils import readlines
from utils import *


warnings.filterwarnings("ignore", category=UserWarning)


def eval(args):
    # Prepare dataset
    splits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'kitti', 'splits', args.eval_split)
    filenames = readlines(os.path.join(splits_dir, "test_files.txt"))
    dataset = KITTIRAWDataset(args.data_dir, 
                              filenames, 
                              args.height, 
                              args.width, 
                              [0],
                              4, 
                              is_train=False, 
                              img_ext='.png')
    dataloader = DataLoader(dataset, 
                            args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers, 
                            pin_memory=True, 
                            drop_last=False)

    # Prepare trained models
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

    print(f"Evaluating on {args.eval_split}")
    print(f"\tSize of the set: {dataset_size}")
    print(f"\tInput resolution: {args.height} x {args.width}")
    print(f"\tAlignment: {args.align}")

    # Predict disparities
    pred_disps = []

    with torch.inference_mode():
        for batch in dataloader:
            frame = batch[("color", 0, 0)].to(device)

            pred_disp = depth_decoder(depth_encoder(frame))[("disp", 0)]

            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    # Load ground truth
    gt_path = os.path.join(splits_dir, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    metrics_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    metrics = np.zeros(len(metrics_names))

    for i in range(dataset_size):
        # get ground truth depths
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        # get predicted depths
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = to_inv(pred_disp)

        if args.eval_split == "eigen":
            # min-max mask for ground truth
            mask = (gt_depth > args.min_depth) & (gt_depth < args.max_depth)
            # Garg crop
            crop_mask = np.zeros_like(mask, dtype=bool)
            crop_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = True
            # combine Garg crop and min-max masks
            mask = mask & crop_mask
        else:
            mask = gt_depth > 0

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


def eval_kitti(data_dir, 
               weights_dir, 
               model_name, 
               eval_split, 
               align, 
               height=192, 
               width=640, 
               batch_size=16, 
               num_workers=8, 
               min_depth=1e-3, 
               max_depth=80):
    args = argparse.Namespace(
        data_dir=data_dir,
        weights_dir=weights_dir,
        model_name=model_name,
        eval_split=eval_split,
        align=align,
        height=height,
        width=width,
        batch_size=batch_size,
        num_workers=num_workers,
        min_depth=min_depth,
        max_depth=max_depth
    )
    return eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on KITTI dataset")
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
    parser.add_argument("--eval_split", 
                        type=str, 
                        choices=[
                            "eigen", 
                            "eigen_benchmark"
                        ], 
                        required=True,
                        help="Splits directory (eigen or eigen_benchmark)")
    parser.add_argument('--align', 
                    type=str,
                    choices=[
                        "median", 
                        "lsqr"
                    ], 
                    required=True,
                    help="Median or least squares alignment")
    parser.add_argument('--height', 
                        type=int, 
                        default=192, 
                        help='Frame height')
    parser.add_argument('--width', 
                        type=int, 
                        default=640, 
                        help='Frame width')
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
                        default=80, 
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