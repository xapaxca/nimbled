import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import warnings

from utils import *
from datasets.make3d.make3d_dataset import Make3DDataset 
from networks.resnet_encoder import ResnetEncoder
from networks.monodepth_decoder import MonoDepthDecoderReLU
from networks.swiftdepth_encoder import SwiftFormer_XS, SwiftFormer_S
from networks.swiftdepth_decoder import DepthDecoderReLU
from networks.litemono_encoder import LiteMono
from networks.litemono_decoder import LiteMonoDecoderReLU


warnings.filterwarnings("ignore", category=UserWarning)


def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def compute_errors_make3d(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log

def eval(args):
    # Prepare dataset
    splits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'make3d', 'splits')
    filenames = readlines(os.path.join(splits_dir, "test_files.txt"))

    input_resolution = (args.height, args.width)
        
    dataset = Make3DDataset(args.data_dir, filenames, input_resolution)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)

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

    depth_encoder_params = sum(p.numel() for p in depth_encoder.parameters())
    print(f"Number of parameters of the depth_encoder: {depth_encoder_params / 1_000_000:.2f} M")

    depth_decoder_params = sum(p.numel() for p in depth_decoder.parameters())
    print(f"Number of parameters of the depth_decoder: {depth_decoder_params / 1_000_000:.2f} M")

    print(f"Total number of parameters: {(depth_encoder_params + depth_decoder_params) / 1_000_000:.2f} M")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depth_encoder.to(device)
    depth_decoder.to(device)

    depth_encoder.eval()
    depth_decoder.eval()

    dataset_size = len(dataset)

    print(f"Evaluating on make3d")
    print(f"\tSize of the set: {dataset_size}")
    print(f"\tInput resolution: {args.height} x {args.width}")
    print(f"\tAlignment: median")

    pred_depths = []
    gt_depths = []
    
    for data in dataloader:
        input_color = data["color"].cuda()

        output = depth_decoder(depth_encoder(input_color))
        pred_disp = output[("disp", 0)]
        pred_disp = pred_disp[:, 0]

        gt_depth = data["depth"]
        _, h, w = gt_depth.shape
        pred_depth = 1 / pred_disp
        pred_depth = F.interpolate(pred_depth.unsqueeze(0), (h, w), mode="nearest")[0]
        pred_depths.append(pred_depth)
        gt_depths.append(gt_depth)
    pred_depths = torch.cat(pred_depths, dim=0)
    gt_depths = torch.cat(gt_depths, dim=0).cuda()

    errors = []
    ratios = []
    for i in range(pred_depths.shape[0]):    
        pred_depth = pred_depths[i]
        gt_depth = gt_depths[i]
        mask = (gt_depth > 0) & (gt_depth < 70)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        ratio = torch.median(gt_depth) / torch.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio         
        pred_depth[pred_depth > 70] = 70
        errors.append(compute_errors_make3d(gt_depth, pred_depth))

    ratios = torch.tensor(ratios)
    med = torch.median(ratios)
    std = torch.std(ratios / med)
    mean_errors = torch.tensor(errors).mean(0)

    metrics_names = ["abs_rel", "sq_rel", "rmse", "rmse_log"]
    metrics_string = ""

    for metric_name in metrics_names:
        metrics_string += f"{metric_name:<10}"
    metrics_string += "\n"

    for metric in mean_errors:
        metrics_string += f"{metric.item():<10.3f}"

    return metrics_string


def eval_make3d(data_dir, 
                weights_dir, 
                model_name, 
                height=192, 
                width=640, 
                batch_size=16, 
                num_workers=8):
    args = argparse.Namespace(
        data_dir=data_dir,
        weights_dir=weights_dir,
        model_name=model_name,
        height=height,
        width=width,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on Make3D dataset")
    parser.add_argument("--data_dir", 
                        type=str, 
                        help="Path to the Make3D data directory",
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
